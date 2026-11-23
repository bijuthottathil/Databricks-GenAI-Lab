# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.2 – Select key metrics to monitor for a specific LLM deployment scenario
# MAGIC
# MAGIC **The big idea for juniors:** you can measure dozens of things about a live LLM app. Monitoring everything creates
# MAGIC noise; monitoring the wrong things misses real problems. The metrics you pick depend on **what the app does** and
# MAGIC **what failure would hurt most**:
# MAGIC - a customer chatbot → wrong answers and slow replies hurt → groundedness, latency, thumbs-down rate
# MAGIC - a nightly batch job → nobody waits → throughput, failure rate, cost per run
# MAGIC - an agent that issues refunds → wrong actions hurt → tool-call errors, policy violations
# MAGIC
# MAGIC ### Metric families
# MAGIC | Family | Examples | Needs |
# MAGIC |---|---|---|
# MAGIC | **Quality** | groundedness, answer relevance, correctness | LLM judges on sampled traffic; correctness needs ground truth |
# MAGIC | **Retrieval** | empty-retrieval rate, retrieval relevance, docs per request | retriever outputs in logs/traces |
# MAGIC | **Safety** | toxicity / safety flags, PII in output, injection attempts | guardrail logs, safety judge |
# MAGIC | **Operational** | latency p50/p95, time-to-first-token, error rate (4xx/5xx), throughput | inference tables / endpoint metrics |
# MAGIC | **Cost** | input/output tokens per request, cost per day, cost per resolved ticket | usage tables |
# MAGIC | **User / business** | thumbs-down rate, escalation to human, resolution rate | app feedback |
# MAGIC | **Agent-specific** | tool-call success rate, steps per request, loops | traces |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 four deployment scenarios described by their features
# MAGIC Step 2 rules: feature → metrics (+ why)        Step 3 recommended metric set per scenario
# MAGIC Step 4 7 days of simulated logs for the support chatbot
# MAGIC Step 5 compute the chosen metrics per day      Step 6 thresholds → alerts
# MAGIC ```
# MAGIC No model calls here – logs are simulated so the numbers are repeatable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Four deployment scenarios
# MAGIC Each scenario is described by yes/no **features**. Features, not names, decide which metrics matter.

# COMMAND ----------

scenarios = {
    "Customer support chatbot (RAG)": {"user_facing": True,  "realtime": True,  "retrieval": True,  "tools": False, "regulated": False, "batch": False},
    "Nightly review summarisation":   {"user_facing": False, "realtime": False, "retrieval": False, "tools": False, "regulated": False, "batch": True},
    "Refund agent with tools":        {"user_facing": True,  "realtime": True,  "retrieval": True,  "tools": True,  "regulated": False, "batch": False},
    "Health-insurance policy Q&A":    {"user_facing": True,  "realtime": True,  "retrieval": True,  "tools": False, "regulated": True,  "batch": False},
}
display(spark.createDataFrame([(k, *v.values()) for k, v in scenarios.items()],
        "scenario string, user_facing boolean, realtime boolean, retrieval boolean, tools boolean, regulated boolean, batch boolean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Rules: which feature needs which metrics
# MAGIC **What:** a list of `(feature, metric, why)`. `always` applies to every LLM deployment.
# MAGIC **Code walk-through:** for each scenario, keep the rules whose feature is `True` (or `always`).

# COMMAND ----------

RULES = [
    ("always",      "error rate (4xx/5xx)",                 "broken requests are invisible without it"),
    ("always",      "tokens & cost per day",                "LLM cost grows with traffic and prompt size"),
    ("realtime",    "latency p95 + time to first token",    "users leave if replies are slow"),
    ("realtime",    "requests per minute vs rate limit",    "spot traffic spikes and 429 throttling"),
    ("user_facing", "safety / toxicity flag rate",          "harmful output reaches customers directly"),
    ("user_facing", "thumbs-down & escalation rate",        "the users' own verdict on quality"),
    ("retrieval",   "groundedness (LLM judge, sampled)",    "answers must come from retrieved docs, not hallucination"),
    ("retrieval",   "empty / low-relevance retrieval rate", "missing knowledge shows up here first"),
    ("tools",       "tool-call error rate",                 "failed tools mean failed actions"),
    ("tools",       "policy-violating actions",             "e.g. refund above limit – must be zero"),
    ("tools",       "steps per request (loop detection)",   "agents can loop and burn tokens"),
    ("regulated",   "PII in output + audit log coverage",   "compliance requires proof"),
    ("regulated",   "correctness on a golden set (weekly)", "ground-truth checks for high-stakes answers"),
    ("batch",       "job success & rows failed (failOnError)", "partial failures must be retried"),
    ("batch",       "throughput (rows/hour) & cost per run", "the job must finish in its window"),
    ("batch",       "output drift (label distribution)",    "silent quality change between runs"),
]

def metrics_for(features):
    return [(m, why) for feature, m, why in RULES if feature == "always" or features.get(feature)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – The recommended metric set per scenario
# MAGIC **How to read the output:**
# MAGIC - The **batch** job has no latency or thumbs-down metrics – nobody is waiting or clicking.
# MAGIC - The **agent** adds tool-call errors, policy violations and loop detection.
# MAGIC - The **regulated** Q&A adds PII checks, audit coverage and golden-set correctness.

# COMMAND ----------

rows = [(s, m, why) for s, f in scenarios.items() for m, why in metrics_for(f)]
display(spark.createDataFrame(rows, "scenario string, metric string, why string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Simulated production logs for the support chatbot
# MAGIC **What:** 7 days × 300 requests. Columns look like what you get from **inference tables** + **LLM-judge
# MAGIC assessments** + **app feedback**. We deliberately make **day 5** bad (a document sync failed → many empty
# MAGIC retrievals → ungrounded answers → more thumbs-down) and **day 6** slow.
# MAGIC `np.random.default_rng(7)` makes the "random" data identical on every run.

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
frames = []
for day in range(1, 8):
    n = 300
    empty_p = 0.30 if day == 5 else 0.04
    empty = rng.random(n) < empty_p
    grounded = np.where(empty, rng.random(n) < 0.35, rng.random(n) < 0.93)
    latency = rng.gamma(4, 300, n) * (2.2 if day == 6 else 1.0)
    frames.append(pd.DataFrame({
        "day": day,
        "status_code": np.where(rng.random(n) < 0.01, 500, 200),
        "latency_ms": latency.round(),
        "ttft_ms": (latency * 0.25).round(),
        "input_tokens": rng.integers(600, 1400, n),
        "output_tokens": rng.integers(80, 250, n),
        "retrieved_docs": np.where(empty, 0, rng.integers(2, 6, n)),
        "grounded": grounded,
        "safety_flag": rng.random(n) < 0.003,
        "thumbs_down": np.where(grounded, rng.random(n) < 0.05, rng.random(n) < 0.40),
    }))
logs = pd.concat(frames, ignore_index=True)
display(logs.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Compute the chosen metrics per day
# MAGIC **Code walk-through:** one `groupby("day")` that turns raw rows into the chatbot's metric set from Step 3.
# MAGIC Cost uses an **illustrative** price of $0.50 / 1M input and $1.50 / 1M output tokens.
# MAGIC **How to read the output:** day 5 – empty retrieval ≈ 30 %, groundedness drops, thumbs-down rises.
# MAGIC Day 6 – p95 latency roughly doubles while quality is normal.

# COMMAND ----------

PRICE_IN, PRICE_OUT = 0.50 / 1e6, 1.50 / 1e6
daily = logs.groupby("day").apply(lambda g: pd.Series({
    "requests": len(g),
    "error_rate": (g.status_code >= 500).mean(),
    "latency_p95_ms": g.latency_ms.quantile(0.95),
    "ttft_p95_ms": g.ttft_ms.quantile(0.95),
    "empty_retrieval_rate": (g.retrieved_docs == 0).mean(),
    "groundedness": g.grounded.mean(),
    "safety_flag_rate": g.safety_flag.mean(),
    "thumbs_down_rate": g.thumbs_down.mean(),
    "cost_usd": (g.input_tokens * PRICE_IN + g.output_tokens * PRICE_OUT).sum(),
})).reset_index().round(4)
display(daily)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Thresholds and alerts
# MAGIC **What:** each metric gets a threshold based on the SLA / normal behaviour. A day that breaks one creates an alert
# MAGIC with the **likely cause** – so the on-call engineer knows where to look.
# MAGIC **How to read the output:** day 5 alerts on empty retrieval + groundedness + thumbs-down (→ check the index sync);
# MAGIC day 6 alerts on latency (→ check endpoint load / provisioned throughput).
# MAGIC In Databricks you would run this as a **SQL alert** or job on the inference table / monitoring results.

# COMMAND ----------

THRESHOLDS = [
    # metric,               direction, limit, likely cause
    ("error_rate",           ">", 0.02, "endpoint or dependency failing"),
    ("latency_p95_ms",       ">", 2500, "endpoint overloaded – scale up / provisioned throughput"),
    ("empty_retrieval_rate", ">", 0.10, "vector index stale or sync failed"),
    ("groundedness",         "<", 0.85, "retrieval or prompt problem – hallucinations"),
    ("safety_flag_rate",     ">", 0.01, "abuse wave or guardrail broken"),
    ("thumbs_down_rate",     ">", 0.12, "users unhappy – inspect sampled traces"),
]
alerts = []
for _, row in daily.iterrows():
    for metric, op, limit, cause in THRESHOLDS:
        value = row[metric]
        if (op == ">" and value > limit) or (op == "<" and value < limit):
            alerts.append((int(row.day), metric, float(value), f"{op} {limit}", cause))
display(spark.createDataFrame(alerts, "day int, metric string, value double, threshold string, likely_cause string"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Start from the **scenario**: user-facing? real-time? retrieval? tools? regulated? batch?
# MAGIC - Always: errors, tokens/cost. Real-time: latency p95/TTFT. RAG: groundedness + retrieval health.
# MAGIC   Agents: tool errors, policy violations, steps. Batch: throughput, failures, drift. Regulated: PII, audit, golden set.
# MAGIC - Metrics need **thresholds** and a **likely cause**, otherwise nobody acts on them.
# MAGIC - Sources on Databricks: **inference tables** (payloads, latency, status), **usage system tables** (tokens),
# MAGIC   **MLflow traces + production monitoring scorers** (quality, safety), app feedback.
