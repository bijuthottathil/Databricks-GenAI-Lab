# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.7 – Select the best LLM based on the attributes of the application
# MAGIC
# MAGIC **The big idea for juniors:** "Which model is best?" has no single answer. The best model is the one that fits
# MAGIC **your application's needs**. A model that is perfect for a legal research assistant may be far too slow and
# MAGIC expensive for classifying a million support tickets.
# MAGIC
# MAGIC **Analogy:** choosing a vehicle. A sports car, a delivery van and a bus are all "good" – for different jobs.
# MAGIC
# MAGIC **Attributes to consider**
# MAGIC
# MAGIC | Attribute | Question to ask | Usually favours |
# MAGIC |---|---|---|
# MAGIC | **Quality / reasoning** | How hard is the task? Multi-step reasoning, nuanced writing? | Larger models |
# MAGIC | **Latency** | Does a user wait for the answer (chat) or is it a batch job? | Smaller models |
# MAGIC | **Cost / volume** | Thousands or millions of calls per day? | Smaller models |
# MAGIC | **Context window** | How much text per request (long documents, many chunks)? | Long-context models |
# MAGIC | **Output structure** | Must it return valid JSON / call tools reliably? | Instruction-tuned models with tool support |
# MAGIC | **Language / domain** | Non-English, code, medical? | Models trained for that |
# MAGIC | **License & hosting** | Can we use it commercially? Must data stay in our cloud? | Open models served on Databricks |
# MAGIC
# MAGIC **In this notebook:** we measure 4 real models on the same tasks, then pick a winner for **two different
# MAGIC applications** using weighted scores – and see that the winner changes.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Endpoint** | The URL/name you call to use a model on Databricks (e.g. `databricks-gemma-3-12b`) |
# MAGIC | **Parameters (70B, 8B)** | The model's size in billions of learned numbers – bigger is usually smarter but slower and pricier |
# MAGIC | **Latency** | Seconds from sending a request to getting the answer |
# MAGIC | **Tokens** | Pieces of words; you pay per token and longer outputs take longer |
# MAGIC | **Normalise** | Convert different units (seconds, %, size) to the same 0–1 scale |
# MAGIC | **Weight** | How much an app cares about a metric; weights add up to 1.0 |
# MAGIC | **Mixture-of-experts (MoE)** | A big model that only uses a small part of itself per token – large capacity, lower cost |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 candidates (4 models)      Step 2 classification_set + reasoning_question
# MAGIC         └──────────────┬───────────────┘
# MAGIC Step 3  call every model on every task → results (raw measurements per model)
# MAGIC Step 4  add speed_score and cost_score to each row of results (0–1 scale)
# MAGIC Step 5  apply each app's weights → ranking per app
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The candidate models (all available as Databricks Foundation Model endpoints)
# MAGIC **What:** a mix of sizes. Parameter count ("B" = billions of parameters) is a rough proxy for capability and cost.
# MAGIC Always confirm context length, pricing and license on the model's documentation page.

# COMMAND ----------

candidates = {
    "databricks-meta-llama-3-3-70b-instruct": "Llama 3.3, 70B – strong general model",
    "databricks-meta-llama-3-1-8b-instruct": "Llama 3.1, 8B – small and fast",
    "databricks-gemma-3-12b": "Gemma 3, 12B – small/medium",
    "databricks-qwen3-next-80b-a3b-instruct": "Qwen3-Next, 80B total / ~3B active (mixture-of-experts)",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Test tasks with known correct answers
# MAGIC **What:** 6 ticket-classification items (the label is the answer key) and 1 reasoning question.
# MAGIC **Why both:** classification tests *accuracy & JSON reliability* (typical high-volume work); the reasoning question
# MAGIC tests *thinking ability* (typical assistant work).
# MAGIC
# MAGIC **The maths answer worked out:** $1200 × 1.08 = $1296 with tax, minus $100 credit = **$1196**. A model that applies
# MAGIC the credit *before* tax gets $1188 – a typical reasoning mistake this question is designed to catch.

# COMMAND ----------

classification_set = [
    ("My order is a week late.", "Shipping"),
    ("I was billed twice this month.", "Payment"),
    ("The screen flickers after two days.", "Product Defect"),
    ("I'd like to send these shoes back, they don't fit.", "Return"),
    ("Package says delivered but nothing at my door.", "Shipping"),
    ("Refund the duplicate charge please.", "Payment"),
]
reasoning_question = ("A laptop costs $1200. Tax is 8%. The customer has a $100 store credit applied after tax. "
                      "What is the final amount to pay? Reply with only the number.")
reasoning_answer = "1196"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Measure every model: accuracy, JSON validity, latency, output size
# MAGIC **What the code measures per model:**
# MAGIC - `accuracy` – share of tickets classified correctly.
# MAGIC - `json_valid` – share of replies that were valid JSON (important if code reads the output).
# MAGIC - `reasoning_correct` – did it get the maths question right?
# MAGIC - `avg_latency_s` – average seconds per call (what a user would wait).
# MAGIC - `avg_output_tokens` – how much text it produced (more tokens = more cost and time).
# MAGIC
# MAGIC **Why `extract_text`:** some models return the answer as a list of parts (e.g. reasoning + text). This small
# MAGIC function handles both shapes so every model is measured the same way.
# MAGIC
# MAGIC **Code walk-through:**
# MAGIC 1. `call()` starts a timer, calls the model, stops the timer → latency. It reads `completion_tokens` from the
# MAGIC    response's `usage` (falls back to a word count if the endpoint doesn't report it).
# MAGIC 2. For each model, loop over the 6 tickets: ask for JSON, try to parse it. Parsed OK → `valid += 1`; category
# MAGIC    equals the label → `correct += 1`. Not JSON → counts as wrong.
# MAGIC 3. Ask the maths question once and check whether `1196` appears in the reply.
# MAGIC 4. Save one row per model in `results`.
# MAGIC
# MAGIC **Model calls:** 4 models × 7 calls = 28 calls; it prints `done: <model>` as each finishes.
# MAGIC
# MAGIC **How to read the table:** one row per model. Accuracy and json_valid are 0–1 (1.0 = 6/6). Latency depends on
# MAGIC load at that moment, so small differences aren't meaningful – look for big gaps.

# COMMAND ----------

import json, time
import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def extract_text(message):
    content = message["content"]
    if isinstance(content, list):                                   # e.g. [{"type": "reasoning", ...}, {"type": "text", "text": "..."}]
        content = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content.strip()

def call(model, prompt, max_tokens=100):
    start = time.time()
    response = client.predict(endpoint=model, inputs={"messages": [{"role": "user", "content": prompt}],
                                                      "temperature": 0.0, "max_tokens": max_tokens})
    latency = time.time() - start
    text = extract_text(response["choices"][0]["message"])
    output_tokens = response.get("usage", {}).get("completion_tokens", len(text.split()))
    return text, latency, output_tokens

results = []
for model in candidates:
    correct = valid = 0
    latencies, tokens = [], []
    for ticket, label in classification_set:
        prompt = ('Classify the ticket. Return ONLY JSON {"category": "Shipping|Payment|Product Defect|Return"}.\n'
                  f"Ticket: {ticket}")
        text, latency, out_tokens = call(model, prompt)
        latencies.append(latency); tokens.append(out_tokens)
        try:
            category = json.loads(text[text.find("{"): text.rfind("}") + 1])["category"]
            valid += 1
            correct += int(category == label)
        except Exception:
            pass                                                     # invalid JSON counts as wrong
    text, latency, out_tokens = call(model, reasoning_question, max_tokens=50)
    reasoning_ok = reasoning_answer in text.replace(",", "")
    results.append({"model": model, "accuracy": correct / len(classification_set),
                    "json_valid": valid / len(classification_set), "reasoning_correct": float(reasoning_ok),
                    "avg_latency_s": float(np.mean(latencies)), "avg_output_tokens": float(np.mean(tokens))})
    print(f"done: {model}")

display(spark.createDataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Turn the raw measurements into 0–1 scores
# MAGIC **What:** for "higher is better" metrics we keep the value; for latency, smaller is better, so the fastest model
# MAGIC gets 1.0 and the slowest 0.0. We also add a **relative cost** score from model size (assumption: smaller = cheaper;
# MAGIC check real prices on the pricing page for production decisions).
# MAGIC **Why normalise:** you can't add "seconds" to "accuracy". Putting everything on 0–1 makes them comparable.
# MAGIC
# MAGIC **Worked example of the formula** `score = 1 - (value - min) / (max - min)`:
# MAGIC latencies 0.5 s, 1.0 s, 2.5 s → min 0.5, max 2.5:
# MAGIC - 0.5 s → 1 − 0/2 = **1.0** (fastest)
# MAGIC - 1.0 s → 1 − 0.5/2 = **0.75**
# MAGIC - 2.5 s → 1 − 2/2 = **0.0** (slowest)
# MAGIC
# MAGIC The `or 1` avoids dividing by zero if all values are equal. This cell adds two new keys to every row of `results`
# MAGIC and prints nothing – the next step uses them.

# COMMAND ----------

size_billions = {"databricks-meta-llama-3-3-70b-instruct": 70, "databricks-meta-llama-3-1-8b-instruct": 8,
                 "databricks-gemma-3-12b": 12, "databricks-qwen3-next-80b-a3b-instruct": 80}

lat = [r["avg_latency_s"] for r in results]
sizes = [size_billions[r["model"]] for r in results]
for r in results:
    r["speed_score"] = 1 - (r["avg_latency_s"] - min(lat)) / ((max(lat) - min(lat)) or 1)
    r["cost_score"] = 1 - (size_billions[r["model"]] - min(sizes)) / ((max(sizes) - min(sizes)) or 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Weight the scores by what each application cares about
# MAGIC **What:** two example applications with different priorities (weights add up to 1.0):
# MAGIC - **App A – high-volume ticket router:** millions of tickets per day, no human waiting on a single result, output
# MAGIC   must be valid JSON. → speed, cost and JSON matter most.
# MAGIC - **App B – customer-facing assistant for tricky questions:** fewer requests, needs to reason correctly.
# MAGIC   → reasoning and accuracy matter most.
# MAGIC
# MAGIC **How to read:** the model with the highest weighted score is the best fit **for that app**. It's normal for the
# MAGIC winners to differ – that's the whole point.
# MAGIC
# MAGIC **Worked example of a weighted score (App B):** a model with accuracy 1.0, json_valid 1.0, speed 0.4,
# MAGIC cost 0.2, reasoning 1.0 →
# MAGIC `0.3×1.0 + 0.1×1.0 + 0.1×0.4 + 0.0×0.2 + 0.5×1.0 = 0.94`.
# MAGIC A model that got the maths wrong loses 0.5 straight away – so for App B, reasoning decides the winner.
# MAGIC
# MAGIC **Try it:** change the weights to match an app you know and re-run this cell – no new model calls needed.

# COMMAND ----------

apps = {
    "App A – high-volume ticket router": {"accuracy": 0.25, "json_valid": 0.25, "speed_score": 0.25, "cost_score": 0.25, "reasoning_correct": 0.0},
    "App B – reasoning assistant":       {"accuracy": 0.3, "json_valid": 0.1, "speed_score": 0.1, "cost_score": 0.0, "reasoning_correct": 0.5},
}

rows = []
for app, weights in apps.items():
    ranked = sorted(results, key=lambda r: -sum(r[m] * w for m, w in weights.items()))
    for rank, r in enumerate(ranked, 1):
        rows.append((app, rank, r["model"], round(sum(r[m] * w for m, w in weights.items()), 3)))
display(spark.createDataFrame(rows, "application string, rank int, model string, weighted_score double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Things the scores don't show (check them too)
# MAGIC - **Context window** – will your longest prompt (instructions + retrieved chunks + history) fit?
# MAGIC - **Tool calling support** – needed for agents.
# MAGIC - **License** – some open models have usage restrictions.
# MAGIC - **Throughput** – for very high volume, use **provisioned throughput** endpoints (pay-per-token endpoints are rate limited).
# MAGIC - **Test on your own data** – public benchmarks are a starting point, not a decision.
# MAGIC
# MAGIC **Exam takeaways:** match the model to the app's **requirements** (quality, latency, cost, context length, structured
# MAGIC output/tool use, language, license). Start with the **smallest model that meets the quality bar**, measure on your
# MAGIC own examples, and weigh attributes explicitly.
