# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.4 – Use inference logging to assess deployed RAG application performance
# MAGIC
# MAGIC **The big idea for juniors:** once a RAG app is deployed, real users ask questions you never tested. **Inference
# MAGIC logging** saves every request and response of a serving endpoint into a Delta table (an **inference table**). By
# MAGIC querying that table you can answer: Is it fast? Is it failing? Does retrieval find documents? Are answers grounded?
# MAGIC Which questions does it handle badly?
# MAGIC
# MAGIC ### What an inference table row looks like (AI Gateway inference tables)
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `databricks_request_id` | unique ID of the request |
# MAGIC | `request_time` | when it arrived |
# MAGIC | `status_code` | HTTP status: 200 ok, 4xx client error, 5xx server error |
# MAGIC | `execution_duration_ms` | how long the model took |
# MAGIC | `request` | the raw JSON sent by the client (e.g. chat messages) |
# MAGIC | `response` | the raw JSON returned (answer – and for our RAG app, the retrieved documents) |
# MAGIC | `requester` | who called the endpoint |
# MAGIC
# MAGIC Because `request`/`response` are JSON **strings**, analysis starts with parsing them.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 simulated inference table (same columns as the real one) – or point the widget at your real table
# MAGIC Step 2 parse JSON → question, answer, retrieved docs
# MAGIC Step 3 operational health: volume, error rate, latency p50/p95
# MAGIC Step 4 retrieval health: empty retrievals, docs per request
# MAGIC Step 5 answer quality: mlflow.genai.evaluate on the LOGGED rows (no re-running the app)
# MAGIC Step 6 turn findings into actions: knowledge gaps, eval-set candidates
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The inference table
# MAGIC **What:** if you have a real endpoint with inference tables enabled, put its table name in the widget
# MAGIC `inference_table` – its `request`/`response` must follow the same JSON shape as below, or adapt Step 2. Otherwise
# MAGIC we create `workspace.fde_eval_monitor.rag_inference_log` with 14 rows that look exactly like real logs, including
# MAGIC a 500 error, slow requests, empty retrievals and one hallucinated answer.
# MAGIC
# MAGIC Our RAG app returns `{"choices": [...], "retrieved_docs": [...]}` – logging retrieved docs in the response is what
# MAGIC makes retrieval analysis possible later. (Agent Framework endpoints log the full **trace** instead.)

# COMMAND ----------

import json
from datetime import datetime, timedelta

dbutils.widgets.text("inference_table", "", "Real inference table (optional)")
REAL_TABLE = dbutils.widgets.get("inference_table").strip()

CATALOG, SCHEMA = "workspace", "fde_eval_monitor"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

def req(q):
    return json.dumps({"messages": [{"role": "user", "content": q}]})

def resp(answer, docs):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": answer}}], "retrieved_docs": docs})

RET = "Items can be returned within 30 days of delivery for a full refund."
REF = "Refunds are paid to the original payment method within 5-7 business days."
SHP = "Express shipping takes 2 business days."
logged = [
    # question, answer, docs, status, duration_ms
    ("How long do I have to return shoes?", "You can return them within 30 days of delivery.", [RET], 200, 1100),
    ("When do refunds arrive?", "Refunds reach your original payment method in 5-7 business days.", [REF], 200, 1250),
    ("How fast is express shipping?", "Express shipping takes 2 business days.", [SHP], 200, 980),
    ("Can I return a used mattress?", "Yes, all items including used mattresses can be returned within 90 days.", [RET], 200, 1400),
    ("Do you ship to Canada?", "I don't have information about shipping to Canada.", [], 200, 900),
    ("Do you ship to Mexico?", "I don't have information about international shipping.", [], 200, 870),
    ("What is the warranty on headphones?", "I couldn't find warranty information.", [], 200, 950),
    ("How long do I have to return a jacket?", "Jackets can be returned within 30 days of delivery.", [RET], 200, 1180),
    ("Refund to a different card?", "Refunds always go to the original payment method.", [REF], 200, 4200),
    ("Is express shipping available on weekends?", "Express takes 2 business days; weekends are not business days.", [SHP], 200, 3900),
    ("Where is order 1002?", "", [], 500, 30000),
    ("Return window for electronics?", "Electronics can be returned within 30 days of delivery.", [RET], 200, 1300),
    ("How do refunds work?", "Refunds go to your original payment method within 5-7 business days.", [REF], 200, 1150),
    ("What is the warranty on laptops?", "I couldn't find warranty information.", [], 200, 990),
]
start = datetime(2024, 6, 10, 9, 0)
rows = [(f"req-{i:03d}", start + timedelta(minutes=17 * i), status, dur, req(q),
         resp(a, d) if status == 200 else json.dumps({"error": "upstream timeout"}), "app-service-principal")
        for i, (q, a, d, status, dur) in enumerate(logged)]
spark.createDataFrame(rows, "databricks_request_id string, request_time timestamp, status_code int, "
                            "execution_duration_ms long, request string, response string, requester string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.rag_inference_log")

TABLE = REAL_TABLE or f"{CATALOG}.{SCHEMA}.rag_inference_log"
print("Analysing:", TABLE)
display(spark.table(TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Parse the JSON payloads
# MAGIC **SQL walk-through:**
# MAGIC - `get_json_object(request, '$.messages[0].content')` – the user's question.
# MAGIC - `get_json_object(response, '$.choices[0].message.content')` – the answer.
# MAGIC - `from_json(get_json_object(response, '$.retrieved_docs'), 'array<string>')` – the documents as a real array.
# MAGIC
# MAGIC We save this as a view `rag_requests` so every following step is simple SQL.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW rag_requests AS
SELECT databricks_request_id, request_time, status_code, execution_duration_ms,
       get_json_object(request, '$.messages[0].content')              AS question,
       get_json_object(response, '$.choices[0].message.content')      AS answer,
       from_json(get_json_object(response, '$.retrieved_docs'), 'array<string>') AS docs
FROM {TABLE}
""")
display(spark.sql("SELECT databricks_request_id, status_code, question, answer, size(docs) AS n_docs FROM rag_requests"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Operational health
# MAGIC **How to read the output:** 14 requests, 1 error (≈ 7 %) – the 30-second timeout. p95 latency is pulled up by the
# MAGIC timeout and two ~4 s requests; p50 ≈ 1.2 s. In production you'd group by hour/day and alert on these.

# COMMAND ----------

display(spark.sql("""
SELECT count(*)                                                   AS requests,
       round(avg(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 3) AS error_rate,
       percentile(execution_duration_ms, 0.5)                     AS latency_p50_ms,
       percentile(execution_duration_ms, 0.95)                    AS latency_p95_ms,
       max(execution_duration_ms)                                 AS latency_max_ms
FROM rag_requests
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Retrieval health
# MAGIC **What:** among successful requests, how often did retrieval return **nothing**, and for which questions?
# MAGIC **How to read the output:** ~4 of 13 successful requests had no documents – all about **international shipping**
# MAGIC and **warranty**. That's not a model problem: the knowledge base simply has no documents on those topics.

# COMMAND ----------

display(spark.sql("""
SELECT round(avg(CASE WHEN size(docs) = 0 THEN 1 ELSE 0 END), 3) AS empty_retrieval_rate,
       round(avg(size(docs)), 2)                                  AS avg_docs_per_request
FROM rag_requests WHERE status_code = 200
"""))
display(spark.sql("SELECT question, answer FROM rag_requests WHERE status_code = 200 AND size(docs) = 0"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Answer quality on the logged traffic
# MAGIC **What:** `mlflow.genai.evaluate` **without** `predict_fn` scores rows that already have `outputs` – exactly the
# MAGIC logged answers. Nothing is re-run; we grade what users actually saw.
# MAGIC **Scorers:**
# MAGIC - `RelevanceToQuery()` – built-in judge: does the answer address the question?
# MAGIC - `grounded_in_docs` – our custom LLM judge: is every claim supported by the retrieved docs? (Section 6.9 explains
# MAGIC   custom scorers.) For requests with no docs, "I don't have information" counts as grounded.
# MAGIC
# MAGIC **How to read the output:** the mattress answer ("90 days, used mattresses too") should be **not grounded** – the
# MAGIC doc says 30 days. That's a hallucination you'd never find from latency metrics.

# COMMAND ----------

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer, RelevanceToQuery
from mlflow.deployments import get_deploy_client

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_inference_logging")
client = get_deploy_client("databricks")

ok = spark.sql("SELECT databricks_request_id, question, answer, docs FROM rag_requests WHERE status_code = 200").toPandas()
eval_rows = [{"inputs": {"question": r.question, "retrieved_docs": list(r.docs)}, "outputs": r.answer} for r in ok.itertuples()]

@scorer
def grounded_in_docs(inputs, outputs) -> Feedback:
    docs = inputs["retrieved_docs"]
    r = client.predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
        "messages": [{"role": "system", "content":
                      "Is every factual claim in the ANSWER supported by the DOCS? If DOCS are empty, an answer that only "
                      "says the information is unavailable is supported. Reply 'yes' or 'no', then a short reason."},
                     {"role": "user", "content": f"DOCS: {docs}\nANSWER: {outputs}"}],
        "temperature": 0.0, "max_tokens": 60})
    text = r["choices"][0]["message"]["content"].strip()
    return Feedback(value=text.lower().startswith("yes"), rationale=text)

results = mlflow.genai.evaluate(data=eval_rows, scorers=[RelevanceToQuery(), grounded_in_docs])
print({k: round(v, 3) for k, v in results.metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC Per-row verdicts of our groundedness judge (re-computed directly so they're easy to read next to the question):

# COMMAND ----------

per_row = [(r["inputs"]["question"], r["outputs"], grounded_in_docs(inputs=r["inputs"], outputs=r["outputs"]).value)
           for r in eval_rows]
display(spark.createDataFrame(per_row, "question string, answer string, grounded boolean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Turn findings into actions
# MAGIC | Finding | Action |
# MAGIC |---|---|
# MAGIC | 1 timeout (500) | check endpoint logs / scale; add retries and a timeout in the client |
# MAGIC | ~4 s requests | inspect traces – long prompts? slow retrieval? |
# MAGIC | Empty retrieval for international shipping + warranty | **knowledge gap** → add documents (Section 2.5) and re-sync the index |
# MAGIC | Ungrounded mattress answer | tighten the prompt ("only answer from documents"), add a groundedness guardrail, add this question to the **eval set** |
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - **Inference tables** log raw requests/responses + status + latency for every call to a serving endpoint.
# MAGIC - Parse the JSON, then measure **operational** (errors, latency), **retrieval** (empty/low-relevance) and
# MAGIC   **quality** (LLM judges on logged outputs with `mlflow.genai.evaluate(data=...)` and no `predict_fn`).
# MAGIC - Log retrieved context (or full traces) – without it you can't assess a RAG app's retrieval or groundedness.
# MAGIC - Feed bad production examples back into the evaluation dataset.
