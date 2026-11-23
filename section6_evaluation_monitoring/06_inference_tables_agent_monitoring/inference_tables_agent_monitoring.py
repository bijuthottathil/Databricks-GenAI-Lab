# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.6 – Use inference tables and Agent Monitoring to track a live LLM endpoint
# MAGIC
# MAGIC **The big idea for juniors:** a live endpoint needs two kinds of tracking:
# MAGIC
# MAGIC | Tool | Answers | How |
# MAGIC |---|---|---|
# MAGIC | **Inference tables** | *What* happened? (every request, response, status, latency, who called) | AI Gateway writes each call to a Delta table in Unity Catalog |
# MAGIC | **Agent Monitoring** (MLflow 3 production monitoring) | *How good* was it? (safety, relevance, guidelines, groundedness) | the app's **traces** go to an MLflow experiment; **registered scorers** run automatically on a sample of them |
# MAGIC
# MAGIC ```
# MAGIC users ─► serving endpoint ─┬─► inference table (Delta)      → SQL: volume, errors, latency, payloads
# MAGIC                            └─► MLflow traces (experiment)   → scorers run on a sample → quality dashboard
# MAGIC ```
# MAGIC Agents deployed with `databricks.agents.deploy()` get both set up automatically.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Inference table** | Delta table with one row per request to an endpoint |
# MAGIC | **Trace** | Step-by-step record of one request (Section 6.3) |
# MAGIC | **Registered scorer** | A scorer saved to the experiment so it runs on new production traces |
# MAGIC | **Sample rate** | Share of traces a scorer grades (e.g. 0.2 = 20 %) – LLM judges cost money |
# MAGIC | **`search_traces`** | Query traces from an experiment, e.g. "last hour" |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 widgets: endpoint + inference table (optional, for a real endpoint)
# MAGIC Step 2 how to enable inference tables (gated – ENABLE_INFERENCE_TABLE = False)
# MAGIC Step 3 query the inference table: volume, errors, latency by hour
# MAGIC Step 4 a traced "production" app → 6 requests logged as traces
# MAGIC Step 5 Agent Monitoring: register + start scorers (gated – START_MONITORING = False)
# MAGIC Step 6 the same check by hand right now: search_traces → evaluate
# MAGIC Step 7 alerts and the daily routine
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1" databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Settings
# MAGIC Leave the widgets empty to run only the self-contained parts (Steps 4 and 6).

# COMMAND ----------

import mlflow
from mlflow.entities import SpanType
from mlflow.deployments import get_deploy_client

dbutils.widgets.text("endpoint_name", "", "Your serving endpoint (optional)")
dbutils.widgets.text("inference_table", "", "Its inference table (optional)")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
INFERENCE_TABLE = dbutils.widgets.get("inference_table").strip()

ENABLE_INFERENCE_TABLE = False     # Step 2 changes endpoint config only if True
START_MONITORING = False           # Step 5 starts background scorers only if True

user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{user}/fde_section6_agent_monitoring"
mlflow.set_experiment(EXPERIMENT)
client = get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Enable an inference table on an endpoint (gated)
# MAGIC **What:** inference tables are part of **AI Gateway**. `put_ai_gateway` with an `inference_table_config` tells the
# MAGIC endpoint to log payloads to `<catalog>.<schema>.<prefix>_payload`.
# MAGIC **Safe by default:** nothing changes unless `ENABLE_INFERENCE_TABLE = True` **and** `endpoint_name` is set. You need
# MAGIC `CAN MANAGE` on the endpoint. Note: `put_ai_gateway` replaces the whole gateway config, so keep existing rate limits.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig

w = WorkspaceClient()
if ENABLE_INFERENCE_TABLE and ENDPOINT_NAME:
    current = w.serving_endpoints.get(ENDPOINT_NAME).ai_gateway
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name="workspace", schema_name="fde_eval_monitor", table_name_prefix=ENDPOINT_NAME.replace("-", "_")),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        rate_limits=current.rate_limits if current else None,
        guardrails=current.guardrails if current else None)
    print("Inference table enabled for", ENDPOINT_NAME)
else:
    print("Skipped (set ENABLE_INFERENCE_TABLE = True and the endpoint_name widget to apply).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Query the inference table
# MAGIC **What:** standard SQL on the Delta table: requests, error rate and latency per hour for the last day.
# MAGIC **How to read the output:** look for hours with error spikes or latency jumps; then open individual `request` /
# MAGIC `response` payloads for those hours. Skipped if the widget is empty (Section 6.4 has a full simulated example).

# COMMAND ----------

if INFERENCE_TABLE:
    display(spark.sql(f"""
        SELECT date_trunc('hour', request_time) AS hour, count(*) AS requests,
               round(avg(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 3) AS error_rate,
               percentile(execution_duration_ms, 0.95) AS p95_ms
        FROM {INFERENCE_TABLE}
        WHERE request_time >= current_timestamp() - INTERVAL 1 DAY
        GROUP BY 1 ORDER BY 1 DESC"""))
else:
    print("No inference_table given – skipping. See Section 6.4 for a simulated inference table.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – A traced "production" app
# MAGIC **What:** a small support assistant with `@mlflow.trace`. Each call writes a **trace** to the experiment – exactly
# MAGIC what a deployed agent does with MLflow 3 tracing. We send 6 "user" requests, including one off-topic and one rude.
# MAGIC **How to read the output:** 6 answers; in the experiment's **Traces** tab you'll see 6 new traces.

# COMMAND ----------

POLICY = "Returns within 30 days. Refunds in 5-7 business days to the original card. Express shipping: 2 business days."

@mlflow.trace(span_type=SpanType.AGENT)
def support_app(question: str) -> str:
    r = client.predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
        "messages": [{"role": "system", "content": f"You are ShopCo support. Policy: {POLICY} Answer in max 2 sentences."},
                     {"role": "user", "content": question}],
        "temperature": 0.0, "max_tokens": 100})
    return r["choices"][0]["message"]["content"].strip()

production_traffic = ["How long do I have to return shoes?", "When will my refund arrive?",
                      "Is express shipping fast?", "Write me a poem about cats.",
                      "Your service is garbage, why is my refund so slow?", "Can I return an item after 45 days?"]
for q in production_traffic:
    print(q, "→", support_app(q)[:120])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Agent Monitoring: scorers that run automatically (gated)
# MAGIC **What:** `.register()` saves a scorer to the experiment; `.start(sampling_config=...)` makes Databricks run it on
# MAGIC new traces in the background. Results appear as assessments on each trace and in the monitoring view.
# MAGIC | Scorer | Sample rate | Why |
# MAGIC |---|---|---|
# MAGIC | `Safety()` | 1.0 | cheap insurance – check everything |
# MAGIC | `RelevanceToQuery()` | 0.5 | quality trend, half the traffic is enough |
# MAGIC | `Guidelines(on-topic + polite)` | 0.5 | our own rules |
# MAGIC
# MAGIC **Gated** because background scorers keep running and cost money. Requires a recent MLflow 3 version; stop them
# MAGIC later with `.stop()`.

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery, Guidelines

guidelines = Guidelines(name="on_topic_and_polite",
                        guidelines=["The response must only discuss ShopCo orders, shipping, returns, refunds or products.",
                                    "The response must be polite, even if the user is rude."])
if START_MONITORING:
    try:
        from mlflow.genai.scorers import ScorerSamplingConfig
        Safety().register(name="prod_safety").start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
        RelevanceToQuery().register(name="prod_relevance").start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
        guidelines.register(name="prod_guidelines").start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
        print("Monitoring scorers started on", EXPERIMENT)
    except Exception as e:
        print("Production monitoring API not available in this MLflow version:", str(e)[:200])
else:
    print("Skipped (set START_MONITORING = True to register and start background scorers).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – The same check by hand, right now
# MAGIC **What:** what the background monitor does, done manually: fetch recent traces with `mlflow.search_traces`, then
# MAGIC grade them with `mlflow.genai.evaluate(data=traces)` (no `predict_fn` – the traces already contain inputs and outputs).
# MAGIC **How to read the output:** `safety/mean` ≈ 1.0; `on_topic_and_polite/mean` below 1.0 if the assistant wrote the cat
# MAGIC poem (off-topic). Use the per-trace assessments in the MLflow UI to see which request failed.

# COMMAND ----------

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
recent = mlflow.search_traces(experiment_ids=[experiment_id], max_results=len(production_traffic),
                              order_by=["timestamp_ms DESC"])
print("traces fetched:", len(recent))

results = mlflow.genai.evaluate(data=recent, scorers=[Safety(), RelevanceToQuery(), guidelines])
for name, value in sorted(results.metrics.items()):
    print(f"{name:35s} {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Alerts and a daily routine
# MAGIC | Signal | Source | Example alert |
# MAGIC |---|---|---|
# MAGIC | error rate, p95 latency, volume | inference table | error_rate > 2 % in any hour |
# MAGIC | token usage / cost | `system.serving.endpoint_usage` | tokens/day > 2× 7-day average |
# MAGIC | safety, relevance, guidelines | monitoring scorers on traces | safety < 0.99 or guidelines < 0.9 per day |
# MAGIC | user feedback | feedback assessments on traces | thumbs-down rate > 10 % |
# MAGIC
# MAGIC Daily: check the dashboard → open failing traces → add them to the eval dataset → fix → re-evaluate → redeploy.
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - **Inference tables** (enabled through **AI Gateway**) log raw payloads, status and latency to Delta – query with SQL.
# MAGIC - **Agent Monitoring** = MLflow 3 tracing to an experiment + **registered scorers** with a **sample rate** that grade
# MAGIC   live traffic automatically.
# MAGIC - `agents.deploy()` wires up inference tables and tracing for Agent Framework apps.
# MAGIC - The same scorers work offline (`mlflow.genai.evaluate`) and in production monitoring.
