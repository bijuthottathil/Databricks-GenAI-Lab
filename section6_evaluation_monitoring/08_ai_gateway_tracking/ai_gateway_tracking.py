# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.8 – Use AI Gateway (inference tables, usage tables, rate limiting) to track an LLM or agent
# MAGIC
# MAGIC **The big idea for juniors:** **Mosaic AI Gateway** is a control layer in front of serving endpoints – foundation
# MAGIC models, external models, and **agents deployed with Agent Framework** (`agents.deploy()` creates a normal serving
# MAGIC endpoint). You switch features on per endpoint; nothing changes in the app code.
# MAGIC
# MAGIC | AI Gateway feature | What you get | Where it lands |
# MAGIC |---|---|---|
# MAGIC | **Inference tables** | every request + response payload, status, latency, requester | `<catalog>.<schema>.<prefix>_payload` Delta table |
# MAGIC | **Usage tracking** | tokens and request counts per call | `system.serving.endpoint_usage` (+ `system.serving.served_entities`) |
# MAGIC | **Rate limits** | max calls (or tokens) per minute per **user**, **endpoint** (or group/service principal) | requests over the limit get HTTP **429** |
# MAGIC | Guardrails | safety / PII filtering on input and output (FMAPI & external models) | blocked requests |
# MAGIC | Fallbacks / traffic split | route to another model on errors | endpoint config |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 audit: which endpoints have which AI Gateway features on? (read-only)
# MAGIC Step 2 configure inference table + usage tracking + rate limits (gated – APPLY = False)
# MAGIC Step 3 usage tables: tokens per endpoint / requester / day (real, or simulated fallback)
# MAGIC Step 4 rate limiting: simulate a per-user limit, then handle 429s with retry + backoff
# MAGIC Step 5 inference table of an Agent Framework endpoint: requests + traces
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Audit AI Gateway settings on all endpoints (read-only)
# MAGIC **What:** list serving endpoints and read each one's `ai_gateway` config.
# MAGIC **How to read the output:** one row per endpoint: is an inference table on, is usage tracking on, how many rate
# MAGIC limits. Pay-per-token `databricks-*` endpoints often show nothing configured – that's normal until you set it.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
rows = []
for ep in w.serving_endpoints.list():
    gw = None
    try:
        gw = w.serving_endpoints.get(ep.name).ai_gateway
    except Exception:
        pass
    rows.append((ep.name,
                 bool(gw and gw.inference_table_config and gw.inference_table_config.enabled),
                 bool(gw and gw.usage_tracking_config and gw.usage_tracking_config.enabled),
                 len(gw.rate_limits) if gw and gw.rate_limits else 0,
                 bool(gw and gw.guardrails)))
display(spark.createDataFrame(rows, "endpoint string, inference_table boolean, usage_tracking boolean, "
                                    "rate_limits int, guardrails boolean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Configure AI Gateway on an endpoint (gated)
# MAGIC **What:** one `put_ai_gateway` call enables all three tracking features:
# MAGIC - `inference_table_config` → payload table in `workspace.fde_eval_monitor`
# MAGIC - `usage_tracking_config` → rows in `system.serving.endpoint_usage`
# MAGIC - `rate_limits` → **20 calls/minute per user** and **200 calls/minute for the whole endpoint**
# MAGIC
# MAGIC **Safe by default:** runs only if `APPLY = True` and you set `TARGET_ENDPOINT` to an endpoint you manage (e.g. your
# MAGIC agent endpoint `agents_workspace-...`). The same settings are available in the UI: **Serving → endpoint → Edit AI Gateway**.

# COMMAND ----------

from databricks.sdk.service.serving import (AiGatewayInferenceTableConfig, AiGatewayUsageTrackingConfig,
                                            AiGatewayRateLimit, AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod)

APPLY = False
TARGET_ENDPOINT = ""          # e.g. "agents_workspace-fde_deploy-support_agent"

if APPLY and TARGET_ENDPOINT:
    w.serving_endpoints.put_ai_gateway(
        name=TARGET_ENDPOINT,
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name="workspace", schema_name="fde_eval_monitor",
            table_name_prefix=TARGET_ENDPOINT.replace("-", "_")),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
        rate_limits=[
            AiGatewayRateLimit(calls=20,  key=AiGatewayRateLimitKey.USER,     renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE),
            AiGatewayRateLimit(calls=200, key=AiGatewayRateLimitKey.ENDPOINT, renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE),
        ])
    print("AI Gateway configured on", TARGET_ENDPOINT)
else:
    print("Skipped – set APPLY = True and TARGET_ENDPOINT to configure a real endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Usage tables: who uses how many tokens?
# MAGIC **What:** join `endpoint_usage` (one row per request: requester, tokens, status) with `served_entities` (which
# MAGIC endpoint/model) and aggregate per endpoint, requester and day. If the system tables aren't available, a small
# MAGIC **simulated** table with the same columns is used so you can still see the analysis.
# MAGIC **How to read the output:** the heaviest users/endpoints by tokens, and the share of requests rejected with 429
# MAGIC (rate-limited). A user with many 429s is either abusing the app or needs a higher limit.

# COMMAND ----------

USAGE_SQL = """
SELECT e.endpoint_name, u.requester, date(u.request_time) AS day, count(*) AS requests,
       sum(u.input_token_count) AS input_tokens, sum(u.output_token_count) AS output_tokens,
       round(avg(CASE WHEN u.status_code = 429 THEN 1 ELSE 0 END), 3) AS rate_limited_share
FROM {usage} u JOIN {entities} e ON u.served_entity_id = e.served_entity_id
WHERE u.request_time >= current_date() - INTERVAL 7 DAYS
GROUP BY 1, 2, 3 ORDER BY output_tokens DESC
"""
try:
    usage = spark.sql(USAGE_SQL.format(usage="system.serving.endpoint_usage", entities="system.serving.served_entities"))
    usage.limit(1).collect()
    source = "system tables"
except Exception as e:
    print("System tables not available – using simulated usage data.", str(e)[:120])
    import random
    from datetime import datetime, timedelta
    random.seed(3)
    now = datetime.now()
    sim = []
    for i in range(400):
        user = random.choice(["app-sp", "app-sp", "app-sp", "analyst@corp.com", "intern@corp.com"])
        status = 429 if user == "intern@corp.com" and random.random() < 0.4 else 200
        sim.append(("se-1", now - timedelta(hours=random.randint(0, 150)), user, status,
                    random.randint(500, 1500) if status == 200 else 0, random.randint(50, 300) if status == 200 else 0))
    spark.createDataFrame(sim, "served_entity_id string, request_time timestamp, requester string, status_code int, "
                               "input_token_count long, output_token_count long").createOrReplaceTempView("sim_usage")
    spark.createDataFrame([("se-1", "agents_support_agent")], "served_entity_id string, endpoint_name string") \
        .createOrReplaceTempView("sim_entities")
    usage = spark.sql(USAGE_SQL.format(usage="sim_usage", entities="sim_entities"))
    source = "simulated"
print("source:", source)
display(usage)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Rate limiting: what it does and how clients should react
# MAGIC **Part A – simulate a limit of 5 calls per minute per user.** 3 users send bursts; the gateway counts calls per
# MAGIC user in the current minute and rejects the rest with 429.
# MAGIC **How to read the output:** `heavy_user` sends 12 calls → 5 allowed, 7 rejected; the others stay under the limit and
# MAGIC are not affected – that's the point of **per-user** limits.

# COMMAND ----------

from collections import Counter

LIMIT_PER_MINUTE = 5
burst = ["heavy_user"] * 12 + ["normal_user_a"] * 3 + ["normal_user_b"] * 4
calls_this_minute, outcome = Counter(), Counter()
for u in burst:
    calls_this_minute[u] += 1
    outcome[(u, 200 if calls_this_minute[u] <= LIMIT_PER_MINUTE else 429)] += 1
for (u, code), n in sorted(outcome.items()):
    print(f"{u:15s} HTTP {code}: {n}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Part B – handle 429 in the client: retry with exponential backoff.**
# MAGIC **Code walk-through:** `call_with_retry` tries the call; on a rate-limit error it waits 1 s, 2 s, 4 s… and tries
# MAGIC again, up to `max_retries`. `fake_endpoint` returns 429 twice, then succeeds – so the output shows two waits and
# MAGIC then the answer. (Waits are shortened with `base_delay=0.2` for the demo.) Wrap real `client.predict` calls the
# MAGIC same way.

# COMMAND ----------

import time

class RateLimited(Exception):
    pass

attempts = {"n": 0}
def fake_endpoint(question):
    attempts["n"] += 1
    if attempts["n"] <= 2:
        raise RateLimited("429 REQUEST_LIMIT_EXCEEDED")
    return f"answer to: {question}"

def call_with_retry(fn, *args, max_retries=4, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return fn(*args)
        except Exception as e:
            if "429" not in str(e) or attempt == max_retries:
                raise
            delay = base_delay * 2 ** attempt
            print(f"rate limited (attempt {attempt + 1}) – waiting {delay:.1f}s")
            time.sleep(delay)

print(call_with_retry(fake_endpoint, "where is my order?", base_delay=0.2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Inference tables for an Agent Framework endpoint
# MAGIC Agents deployed with `agents.deploy()` are served by a normal endpoint, so AI Gateway works the same. Their payload
# MAGIC table also contains the agent's **trace** in the response, which you can parse for tool calls and latency per step:
# MAGIC ```sql
# MAGIC SELECT request_time, status_code, execution_duration_ms,
# MAGIC        get_json_object(request,  '$.input[0].content')         AS question,      -- ResponsesAgent input
# MAGIC        get_json_object(response, '$.output[0].content[0].text') AS answer,
# MAGIC        get_json_object(response, '$.databricks_output.trace')   AS trace_json
# MAGIC FROM workspace.fde_eval_monitor.<prefix>_payload
# MAGIC WHERE request_time >= current_date() - INTERVAL 1 DAY
# MAGIC ```
# MAGIC (Exact JSON paths depend on the agent's input/output schema – check one row first.)
# MAGIC
# MAGIC ### Which feature answers which question?
# MAGIC | Question | Use |
# MAGIC |---|---|
# MAGIC | What did users ask, what did the agent answer, how long did it take? | **inference table** |
# MAGIC | Who is using how many tokens? What does it cost? | **usage tables** (+ `system.billing.usage`) |
# MAGIC | How do I stop one user/app from overloading or overspending? | **rate limits** |
# MAGIC | Is the answer quality OK? | inference table / traces + **judges** (Sections 6.4, 6.6) |
# MAGIC
# MAGIC **Exam takeaways:** AI Gateway is configured **per endpoint** (UI or `put_ai_gateway`), works for FMAPI, external
# MAGIC models and **Agent Framework** endpoints; **inference tables** = payload logs in UC, **usage tracking** =
# MAGIC `system.serving.endpoint_usage`, **rate limits** = calls per minute per user/endpoint → HTTP 429, so clients should
# MAGIC retry with backoff.
