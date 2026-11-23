# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.10 – Select the best model for a task based on metrics generated in experiments
# MAGIC
# MAGIC **The big idea for juniors:** after reading model cards (3.9) you have a shortlist. Now you **run an experiment**:
# MAGIC give every candidate the same test data, compute the same metrics, log everything in **MLflow**, and pick the model
# MAGIC with the best numbers *for your decision rule*.
# MAGIC
# MAGIC **Common metrics – what they mean in plain English**
# MAGIC
# MAGIC | Metric | Task | Plain-English meaning | Better |
# MAGIC |---|---|---|---|
# MAGIC | **Accuracy** | Classification | Share of answers that were exactly right | Higher |
# MAGIC | **F1 (macro)** | Classification | Balance of precision & recall, averaged over classes – fair when classes are uneven | Higher |
# MAGIC | **Exact match** | QA / extraction | Answer identical to the expected answer | Higher |
# MAGIC | **ROUGE-1** | Summarisation | Share of the reference summary's words that appear in the model's summary | Higher |
# MAGIC | **ROUGE-L** | Summarisation | Longest run of words in the same order as the reference (rewards good phrasing) | Higher |
# MAGIC | **BLEU** | Translation | Overlap of word sequences with a reference translation | Higher |
# MAGIC | **Perplexity** | Language modelling | How "surprised" the model is by real text | Lower |
# MAGIC | **LLM-judge score** | Any open-ended text | A strong model grades correctness/helpfulness with a rubric | Higher |
# MAGIC | **Latency (p50 / p95)** | All | Typical / worst-case seconds per request | Lower |
# MAGIC | **Tokens / cost** | All | How much output (and money) each request uses | Lower |
# MAGIC
# MAGIC **Our task:** a support app needs a model that (1) classifies tickets and (2) summarises them. We compare 3 models.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Experiment** (MLflow) | A folder that groups related runs – here `fde_model_selection` |
# MAGIC | **Run** | One try of one setup – here one run per model |
# MAGIC | **Params** | The *settings* of a run (model name, temperature) |
# MAGIC | **Metrics** | The *numbers measured* (F1, ROUGE-L, latency) |
# MAGIC | **Artifact** | A file saved with the run (our table of every prediction) |
# MAGIC | **Reference / label** | The correct answer written by a human, used to grade the model |
# MAGIC | **p50 / p95** | Median / "95% of calls were faster than this" (worst-case-ish) |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1  test_data (ticket, correct label, reference summary), MODELS
# MAGIC Step 2  metric functions: macro_f1, rouge_1, rouge_l    (pure Python, no model calls)
# MAGIC Step 3  for each model: classify + summarise all tickets → compute metrics → log one MLflow run
# MAGIC Step 4  read the runs BACK from MLflow → latest (one row per model)
# MAGIC Step 5  decision rule on latest → selected model
# MAGIC ```
# MAGIC Step 5 reads from MLflow, not from Python variables – so the decision is based on what's *recorded*, which anyone
# MAGIC can re-check later on the Experiments page.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Test data with known answers (labels and reference summaries)
# MAGIC **What:** each row is `(ticket text, correct category, human-written reference summary)`. The category grades the
# MAGIC classification task; the reference summary grades the summarisation task.
# MAGIC
# MAGIC **Note the uneven classes:** 2 Shipping, 2 Product Defect, 1 Payment, 1 Return – realistic, and why we use macro F1.

# COMMAND ----------

test_data = [
    ("My laptop arrived two weeks late and the box was damaged. I need it for work.", "Shipping",
     "Laptop delivered two weeks late with a damaged box."),
    ("I see two identical charges of $150 on my card for one pair of headphones.", "Payment",
     "Customer was charged twice for one pair of headphones."),
    ("After three days the keyboard keys started sticking and some do not work.", "Product Defect",
     "Keyboard keys started sticking after three days."),
    ("The jacket does not fit, I want to send it back and get my money.", "Return",
     "Customer wants to return a jacket that does not fit."),
    ("Tracking says delivered but there is no package at my door.", "Shipping",
     "Package marked delivered but not received."),
    ("Monitor has dead pixels out of the box, please help.", "Product Defect",
     "New monitor has dead pixels."),
]
CATEGORIES = ["Shipping", "Payment", "Product Defect", "Return"]
MODELS = ["databricks-meta-llama-3-3-70b-instruct", "databricks-meta-llama-3-1-8b-instruct", "databricks-gemma-3-12b"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Metric functions, written out so you can see how they work
# MAGIC **What:** `macro_f1` computes F1 per category and averages; `rouge_1` and `rouge_l` compare word overlap with the
# MAGIC reference summary (`rouge_l` uses the *longest common subsequence* = longest run of shared words in order).
# MAGIC
# MAGIC **macro F1 in plain words** – for each category:
# MAGIC - **tp** (true positive) = said "Shipping" and it *was* Shipping
# MAGIC - **fp** (false positive) = said "Shipping" but it wasn't
# MAGIC - **fn** (false negative) = it was Shipping but the model said something else
# MAGIC - precision = tp / (tp + fp) "when it says Shipping, how often is it right?"; recall = tp / (tp + fn) "of all real
# MAGIC   Shipping tickets, how many did it find?"; F1 = balance of the two.
# MAGIC
# MAGIC Then average F1 over the 4 categories, so the single Return ticket counts as much as the two Shipping ones.
# MAGIC
# MAGIC **ROUGE worked example:** reference "Laptop delivered two weeks late with a damaged box." vs model
# MAGIC "Laptop arrived 2 weeks late, box damaged". Shared words in order: *laptop … weeks late … box* → decent ROUGE-L even
# MAGIC though "2" ≠ "two" and "arrived" ≠ "delivered". That's the limitation: ROUGE counts words, not meaning.
# MAGIC
# MAGIC **How to read the output:** one number around 0.5 – the score for that example pair.

# COMMAND ----------

import re
import numpy as np

def macro_f1(true_labels, predicted):
    f1s = []
    for c in CATEGORIES:
        tp = sum(t == c and p == c for t, p in zip(true_labels, predicted))       # correctly predicted c
        fp = sum(t != c and p == c for t, p in zip(true_labels, predicted))       # predicted c but wrong
        fn = sum(t == c and p != c for t, p in zip(true_labels, predicted))       # missed a real c
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))

def words(text):
    return re.findall(r"[a-z0-9$]+", text.lower())

def rouge_1(candidate, reference):
    cand, ref = words(candidate), words(reference)
    overlap = sum(min(cand.count(w), ref.count(w)) for w in set(ref))
    return overlap / len(ref) if ref else 0.0                                     # recall-style ROUGE-1

def rouge_l(candidate, reference):
    a, b = words(candidate), words(reference)
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]                       # dynamic-programming table for LCS
    for i in range(len(a)):
        for j in range(len(b)):
            table[i + 1][j + 1] = table[i][j] + 1 if a[i] == b[j] else max(table[i][j + 1], table[i + 1][j])
    lcs = table[-1][-1]
    if not lcs:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)

print("example ROUGE-L:", round(rouge_l("Laptop arrived 2 weeks late, box damaged", test_data[0][2]), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Run the experiment: one MLflow run per model
# MAGIC **What:** for each model, classify and summarise every ticket, time each call, compute all metrics, and log
# MAGIC params + metrics to MLflow. We also save a table of every prediction as an **artifact** so you can inspect mistakes.
# MAGIC **Why MLflow:** results are stored permanently and side by side; anyone on the team can see how the decision was made.
# MAGIC
# MAGIC **Code walk-through:**
# MAGIC 1. `mlflow.set_experiment(...)` – create/choose the experiment folder in your user directory.
# MAGIC 2. `call()` – plain model call that also returns latency and `total_tokens`.
# MAGIC 3. For each model and each ticket: one classification call and one summary call (2 calls × 6 tickets × 3 models
# MAGIC    = **36 calls**). The classifier's reply is mapped to a category by looking for the category name in the text;
# MAGIC    anything else becomes `"Unknown"` (counts as wrong).
# MAGIC 4. Compute all metrics with the Step 2 functions.
# MAGIC 5. `with mlflow.start_run(run_name=model):` – everything inside is saved to that run: `log_params` (settings),
# MAGIC    `log_metrics` (numbers), `log_table` (every ticket with prediction and summary).
# MAGIC
# MAGIC **How to read the output:** one printed line per model with its metrics. Open **Experiments → fde_model_selection**
# MAGIC in the left sidebar to see the same runs, and click a run → *Artifacts* → `predictions.json` to see each mistake.

# COMMAND ----------

import time, json
import mlflow
import pandas as pd
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_model_selection")

def call(model, prompt, max_tokens=80):
    start = time.time()
    response = client.predict(endpoint=model, inputs={"messages": [{"role": "user", "content": prompt}],
                                                      "temperature": 0.0, "max_tokens": max_tokens})
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):                                       # some models return a list of parts
        content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content.strip(), time.time() - start, response.get("usage", {}).get("total_tokens", 0)

for model in MODELS:
    predictions, summaries, latencies, tokens = [], [], [], []
    for ticket, label, reference in test_data:
        text, sec, tok = call(model, f"Classify the ticket into exactly one of {CATEGORIES}. Reply with the category only.\n\n{ticket}")
        predictions.append(next((c for c in CATEGORIES if c.lower() in text.lower()), "Unknown"))
        latencies.append(sec); tokens.append(tok)
        text, sec, tok = call(model, f"Summarise this ticket in one short sentence (max 12 words).\n\n{ticket}")
        summaries.append(text); latencies.append(sec); tokens.append(tok)

    labels = [t[1] for t in test_data]
    metrics = {
        "accuracy": float(np.mean([p == l for p, l in zip(predictions, labels)])),
        "macro_f1": macro_f1(labels, predictions),
        "rouge_1": float(np.mean([rouge_1(s, t[2]) for s, t in zip(summaries, test_data)])),
        "rouge_l": float(np.mean([rouge_l(s, t[2]) for s, t in zip(summaries, test_data)])),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "avg_tokens_per_call": float(np.mean(tokens)),
    }
    with mlflow.start_run(run_name=model):
        mlflow.log_params({"model": model, "temperature": 0.0, "num_examples": len(test_data)})
        mlflow.log_metrics(metrics)
        mlflow.log_table(pd.DataFrame({"ticket": [t[0] for t in test_data], "label": labels, "predicted": predictions,
                                       "reference_summary": [t[2] for t in test_data], "summary": summaries}),
                         artifact_file="predictions.json")
    print(model, {k: round(v, 3) for k, v in metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Compare runs straight from MLflow
# MAGIC **What:** `mlflow.search_runs` returns every run in the experiment as a table (`metrics.*` and `params.*` columns).
# MAGIC **Why:** this is how you compare dozens of runs – or open the **Experiments** page and use the chart view.
# MAGIC
# MAGIC **Code walk-through:** newest runs first; keep the `params.model` column plus every `metrics.*` column;
# MAGIC `drop_duplicates("params.model")` keeps only the newest run per model – so if you run Step 3 twice, older runs
# MAGIC don't get mixed in.

# COMMAND ----------

runs = mlflow.search_runs(order_by=["start_time DESC"])
cols = ["params.model"] + [c for c in runs.columns if c.startswith("metrics.")]
latest = runs[cols].drop_duplicates("params.model")                  # newest run per model
display(latest)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Apply a decision rule (agreed *before* looking at the numbers)
# MAGIC **Rule:**
# MAGIC 1. **Quality gate** – must reach macro F1 ≥ 0.8 **and** ROUGE-L ≥ 0.3 (otherwise it's not good enough, whatever the speed).
# MAGIC 2. Among models that pass, choose the **lowest p95 latency** (users hate slow worst cases).
# MAGIC 3. Tie-break on fewer tokens (cheaper).
# MAGIC
# MAGIC **Why a gate first:** a fast model that gets answers wrong is not "efficient", it's broken.
# MAGIC
# MAGIC **Why agree the rule *before* looking:** otherwise it's tempting to pick your favourite model and then choose the
# MAGIC metric that makes it win.
# MAGIC
# MAGIC **How to read the output:** either `Selected model: …` with its key numbers, or a message that no model passed –
# MAGIC which is a valid result too (it tells you to improve prompts or try other models).

# COMMAND ----------

passed = latest[(latest["metrics.macro_f1"] >= 0.8) & (latest["metrics.rouge_l"] >= 0.3)]
if passed.empty:
    print("No model passed the quality gate – improve prompts, try other models, or revisit the thresholds.")
else:
    winner = passed.sort_values(["metrics.latency_p95_s", "metrics.avg_tokens_per_call"]).iloc[0]
    print(f"Selected model: {winner['params.model']}")
    print(f"  macro F1 {winner['metrics.macro_f1']:.2f} | ROUGE-L {winner['metrics.rouge_l']:.2f} | "
          f"p95 latency {winner['metrics.latency_p95_s']:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to read the results – and their limits
# MAGIC - **Accuracy vs F1:** with uneven classes, a model that always says "Shipping" can look accurate; macro F1 exposes it.
# MAGIC - **ROUGE** rewards word overlap, not meaning – a correct summary in different words scores lower. For open-ended
# MAGIC   text, add an **LLM judge** (e.g. MLflow's `Correctness` scorer) or human review.
# MAGIC - **6 examples is a demo.** Real decisions need 50–500 representative examples and repeated runs.
# MAGIC - **p95 latency** varies with endpoint load – measure at realistic times.
# MAGIC
# MAGIC **Exam takeaways:** run every candidate on the **same** dataset, log **params + metrics** to MLflow, pick task-relevant
# MAGIC metrics (F1 for classification, ROUGE/BLEU for summarisation/translation, LLM judges for open-ended, latency & cost
# MAGIC for all), and choose with a clear rule: **quality gate first, then efficiency**.
