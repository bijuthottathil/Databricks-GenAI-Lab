# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.1 – Select an LLM (size and architecture) from quantitative evaluation metrics
# MAGIC
# MAGIC **The big idea for juniors:** bigger models are usually smarter but slower and more expensive. Some models use a
# MAGIC **Mixture-of-Experts (MoE)** architecture: they have many parameters in total but only use a small part for each
# MAGIC token, so they can be fast for their size. You don't pick by name or hype – you **measure** each candidate on your
# MAGIC task and choose with a clear rule.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Parameters (B)** | Size of the model in billions of numbers. More ≈ more knowledge and reasoning, more cost |
# MAGIC | **Dense model** | Every parameter is used for every token (Llama 8B, Gemma 12B, Llama 70B) |
# MAGIC | **Mixture of Experts (MoE)** | Many "expert" sub-networks; a router picks a few per token. **Total** params drive memory, **active** params drive speed |
# MAGIC | **Accuracy** | Share of answers that match the answer key |
# MAGIC | **Format validity** | Share of answers in the required format (valid JSON) |
# MAGIC | **Latency p50 / p95** | Typical / slow-case response time |
# MAGIC | **Output tokens per second** | How fast text is generated |
# MAGIC | **Quality gate** | Minimum quality a model must reach before we even compare cost and speed |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 candidates (size + architecture)   Step 2 test sets: classification, JSON extraction, reasoning
# MAGIC Step 3 run every model on every test → per-call results
# MAGIC Step 4 aggregate metrics per model + log to MLflow
# MAGIC Step 5 decision rule per scenario: quality gate first, then efficiency
# MAGIC Step 6 what size/architecture tells you before you measure
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 0 – Install MLflow 3
# MAGIC Run first; `%restart_python` clears variables.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Candidate models
# MAGIC **What:** four pay-per-token endpoints of different **size** and **architecture**. `relative_cost` is an
# MAGIC **illustrative** cost index (8B = 1) for teaching – check current Databricks pricing for real numbers.
# MAGIC If an endpoint doesn't exist in your workspace it is skipped automatically.

# COMMAND ----------

import time, json, re
import numpy as np
import pandas as pd
import mlflow
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

candidates = pd.DataFrame([
    # endpoint,                                   total_B, active_B, architecture, relative_cost
    ("databricks-meta-llama-3-1-8b-instruct",      8,    8,   "dense", 1.0),
    ("databricks-gemma-3-12b",                     12,   12,  "dense", 1.5),
    ("databricks-qwen3-next-80b-a3b-instruct",     80,   3,   "MoE",   2.0),
    ("databricks-meta-llama-3-3-70b-instruct",     70,   70,  "dense", 5.0),
], columns=["endpoint", "total_params_b", "active_params_b", "architecture", "relative_cost"])
display(candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Three small test sets with answer keys
# MAGIC | Test | Why it matters | Scored by |
# MAGIC |---|---|---|
# MAGIC | **classification** (8) | high-volume ticket routing | exact category match |
# MAGIC | **extraction** (4) | structured output for downstream code | valid JSON **and** correct fields |
# MAGIC | **reasoning** (4) | multi-step questions (refund maths, dates) | exact final number |
# MAGIC
# MAGIC Small models are usually fine at classification, weaker at multi-step reasoning – the tests are designed to show that.

# COMMAND ----------

CATS = ["Shipping", "Payment", "Product Defect", "Return", "Account"]
classification = [
    ("My parcel is a week late.", "Shipping"), ("I was charged twice.", "Payment"),
    ("The screen cracked on day one.", "Product Defect"), ("I'd like to send the jacket back.", "Return"),
    ("I can't log in to my profile.", "Account"), ("Tracking hasn't moved in 5 days.", "Shipping"),
    ("Refund the duplicate payment.", "Payment"), ("The kettle stopped heating.", "Product Defect"),
]
extraction = [
    ("Order 1002 for 2 laptops arrived damaged.", {"order_id": "1002", "quantity": 2}),
    ("I ordered 3 mugs in order 5531 and got none.", {"order_id": "5531", "quantity": 3}),
    ("Order number 7781: one pair of headphones, wrong colour.", {"order_id": "7781", "quantity": 1}),
    ("Five chairs, order 9001, still not shipped.", {"order_id": "9001", "quantity": 5}),
]
reasoning = [
    ("An order had 3 items at $40 each. The customer returns 2 items and pays a $5 return fee. How many dollars is the refund?", 75),
    ("Shipping takes 4 business days. An order is shipped on a Thursday. Weekends don't count. How many calendar days until delivery?", 6),
    ("A customer bought 5 items at $12 with a 20% discount on the total, then returned 1 item at its discounted price. What did they pay in dollars in the end?", 38.4),
    ("A refund of $90 is split equally across 3 cards, then each card is charged a $2 fee. How many dollars land on each card?", 28),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Run every model on every test
# MAGIC **Code walk-through:**
# MAGIC - `call()` sends one chat request with `temperature 0.0`, times it, and reads `usage.completion_tokens` (output
# MAGIC   token count) from the response – needed for tokens/second.
# MAGIC - Reasoning answers: we ask for the final number on the last line as `ANSWER: <number>` and compare with a small
# MAGIC   tolerance.
# MAGIC - Endpoints that fail on the first call are marked unavailable and skipped.
# MAGIC
# MAGIC This is ~64 model calls; expect a couple of minutes.

# COMMAND ----------

def call(endpoint, system, user, max_tokens):
    start = time.perf_counter()
    r = client.predict(endpoint=endpoint, inputs={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": max_tokens})
    seconds = time.perf_counter() - start
    out_tokens = (r.get("usage") or {}).get("completion_tokens") or 0
    return r["choices"][0]["message"]["content"].strip(), seconds, out_tokens

def last_number(text):
    m = re.findall(r"ANSWER:\s*\$?(-?\d+(?:\.\d+)?)", text)
    nums = m or re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None

records = []
for endpoint in candidates.endpoint:
    try:
        for text, label in classification:
            out, s, tok = call(endpoint, f"Classify into one of {CATS}. Reply with the category only.", text, 10)
            records.append((endpoint, "classification", out.strip(".") == label, True, s, tok))
        for text, expected in extraction:
            out, s, tok = call(endpoint, 'Return ONLY JSON {"order_id": "<string>", "quantity": <int>}.', text, 40)
            try:
                parsed = json.loads(out[out.find("{"): out.rfind("}") + 1])
                valid = True
                correct = str(parsed.get("order_id")) == expected["order_id"] and int(parsed.get("quantity", -1)) == expected["quantity"]
            except Exception:
                valid, correct = False, False
            records.append((endpoint, "extraction", correct, valid, s, tok))
        for question, expected in reasoning:
            out, s, tok = call(endpoint, "Think step by step, then give the final number on the last line as 'ANSWER: <number>'.",
                               question, 400)
            got = last_number(out)
            records.append((endpoint, "reasoning", got is not None and abs(got - expected) < 0.01, True, s, tok))
        print("done:", endpoint)
    except Exception as e:
        print("skipped (unavailable):", endpoint, str(e)[:120])

results = pd.DataFrame(records, columns=["endpoint", "test", "correct", "format_valid", "seconds", "output_tokens"])
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Metrics per model (and log them to MLflow)
# MAGIC **What:** one row per model with accuracy per test, JSON validity, latency p50/p95 and output tokens per second.
# MAGIC Each model also becomes an **MLflow run**, so you can compare them in the Experiments UI.
# MAGIC
# MAGIC **How to read the output (typical pattern – your numbers will differ):**
# MAGIC - All models do well on **classification**.
# MAGIC - **Reasoning** separates them: 8B makes arithmetic/date mistakes; 70B and the MoE model do better.
# MAGIC - The **MoE** model (3B active) is often much faster per token than the 70B dense model at similar quality.
# MAGIC - The 8B model has the lowest latency and cost.

# COMMAND ----------

def summarise(g):
    return pd.Series({
        "classification_acc": g[g.test == "classification"].correct.mean(),
        "extraction_acc": g[g.test == "extraction"].correct.mean(),
        "json_valid_rate": g[g.test == "extraction"].format_valid.mean(),
        "reasoning_acc": g[g.test == "reasoning"].correct.mean(),
        "latency_p50_s": g.seconds.quantile(0.5),
        "latency_p95_s": g.seconds.quantile(0.95),
        "tokens_per_s": g.output_tokens.sum() / g.seconds.sum(),
    })

metrics = results.groupby("endpoint").apply(summarise).reset_index().merge(candidates, on="endpoint")
display(metrics)

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_llm_selection")
for _, row in metrics.iterrows():
    with mlflow.start_run(run_name=row.endpoint):
        mlflow.log_params({"endpoint": row.endpoint, "architecture": row.architecture,
                           "total_params_b": row.total_params_b, "active_params_b": row.active_params_b})
        mlflow.log_metrics({k: float(row[k]) for k in ["classification_acc", "extraction_acc", "json_valid_rate",
                                                         "reasoning_acc", "latency_p50_s", "latency_p95_s",
                                                         "tokens_per_s", "relative_cost"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Choose per scenario: quality gate first, then efficiency
# MAGIC **The rule:**
# MAGIC 1. Drop every model below the scenario's **quality gate**.
# MAGIC 2. Among the rest, pick the one with the best **efficiency score** for that scenario (cost and/or latency).
# MAGIC
# MAGIC | Scenario | Quality gate | Then prefer |
# MAGIC |---|---|---|
# MAGIC | A. Route 1M tickets/day | classification ≥ 0.85 **and** JSON valid ≥ 0.95 | lowest cost |
# MAGIC | B. Refund-calculation assistant | reasoning ≥ 0.75 **and** extraction ≥ 0.75 | lowest p95 latency |
# MAGIC | C. Live chat with 2 s budget | classification ≥ 0.85, p95 ≤ 2 s | cheapest |
# MAGIC
# MAGIC **How to read the output:** the chosen model per scenario, or "no model passes" – which itself is useful information
# MAGIC (improve the prompt, add few-shot examples, or consider a larger model / fine-tuning).

# COMMAND ----------

scenarios = {
    "A. high-volume routing":  (lambda m: (m.classification_acc >= 0.85) & (m.json_valid_rate >= 0.95), "relative_cost"),
    "B. refund reasoning":     (lambda m: (m.reasoning_acc >= 0.75) & (m.extraction_acc >= 0.75), "latency_p95_s"),
    "C. live chat, 2 s budget": (lambda m: (m.classification_acc >= 0.85) & (m.latency_p95_s <= 2.0), "relative_cost"),
}
choices = []
for name, (gate, prefer) in scenarios.items():
    passing = metrics[gate(metrics)].sort_values(prefer)
    chosen = passing.endpoint.iloc[0] if len(passing) else "no model passes the gate"
    choices.append((name, ", ".join(passing.endpoint) or "-", prefer, chosen))
display(spark.createDataFrame(choices, "scenario string, passed_gate string, tie_breaker string, chosen string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – What size and architecture tell you (before measuring)
# MAGIC | Property | Effect |
# MAGIC |---|---|
# MAGIC | More **total** parameters | more knowledge and reasoning; more GPU memory → higher cost to host |
# MAGIC | More **active** parameters per token | slower generation and higher compute cost per token |
# MAGIC | **MoE** (large total, small active) | quality close to big dense models, speed closer to small ones |
# MAGIC | Small dense (≤ 12B) | cheapest and fastest; good for classification/extraction with clear prompts |
# MAGIC | Large dense (70B+) | strongest on complex reasoning, long instructions, nuanced writing |
# MAGIC | Context window | must fit your prompt + retrieved documents |
# MAGIC | Instruction-tuned vs base | use instruction-tuned ("-instruct") models for chat/apps |
# MAGIC
# MAGIC **Exam takeaways:** compare candidates on **task-specific metrics** (accuracy, format validity, reasoning) **and**
# MAGIC operational metrics (latency p95, throughput, cost); apply a **quality gate**, then choose the **smallest/cheapest**
# MAGIC model that passes. MoE models trade memory for speed – judge them on measured latency and quality.
