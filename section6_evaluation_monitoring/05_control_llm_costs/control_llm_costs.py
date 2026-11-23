# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.5 – Use Databricks features to control LLM costs
# MAGIC
# MAGIC **The big idea for juniors:** LLM cost ≈ **number of requests × tokens per request × price per token** (pay-per-token)
# MAGIC or **hours the endpoint runs × its size** (provisioned / custom endpoints). So you control cost by:
# MAGIC 1. sending **fewer tokens** (shorter prompts, `max_tokens`, fewer retrieved chunks),
# MAGIC 2. using a **cheaper model** where quality allows,
# MAGIC 3. making **fewer calls** (caching, batching),
# MAGIC 4. choosing the right **serving mode** (pay-per-token vs provisioned throughput, scale to zero),
# MAGIC 5. **limiting and tracking** usage (AI Gateway rate limits, usage tables, tags, budgets).
# MAGIC
# MAGIC ### Databricks features for cost control
# MAGIC | Feature | What it does |
# MAGIC |---|---|
# MAGIC | **Foundation Model APIs pay-per-token** | no idle cost – pay only for tokens; good for low/variable traffic |
# MAGIC | **Provisioned throughput** | reserved capacity for high, steady traffic; cheaper per token at scale, predictable |
# MAGIC | **Scale to zero** (custom/provisioned endpoints) | no compute cost when idle (cold start on first request) |
# MAGIC | **AI Gateway rate limits** | caps requests/tokens per user or endpoint – stops runaway costs |
# MAGIC | **AI Gateway usage tracking** → `system.serving.endpoint_usage` | token counts per request, user, endpoint |
# MAGIC | **`system.billing.usage`** + custom **tags** / **budget policies** | cost per endpoint, team, project; budget alerts |
# MAGIC | **Batch inference with `ai_query`** | efficient bulk processing instead of many real-time calls |
# MAGIC | Smaller / MoE models, prompt and retrieval tuning | fewer or cheaper tokens |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 measure tokens from a real response (usage field)
# MAGIC Step 2 lever: shorter prompt          Step 3 lever: max_tokens + concise output
# MAGIC Step 4 lever: smaller model (same accuracy?)   Step 5 lever: caching repeated questions
# MAGIC Step 6 monthly projection: baseline vs optimised
# MAGIC Step 7 track real usage in system tables     Step 8 serving-mode decision + checklist
# MAGIC ```
# MAGIC Prices in this notebook are **illustrative** – look up current Databricks pricing (DBUs) for real numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Every response tells you its token count
# MAGIC **What:** the response's `usage` field has `prompt_tokens` (input) and `completion_tokens` (output).
# MAGIC `cost()` turns them into dollars with illustrative prices (big model: $0.50 in / $1.50 out per 1M tokens; small model:
# MAGIC $0.15 / $0.45).

# COMMAND ----------

import time
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
BIG, SMALL = "databricks-meta-llama-3-3-70b-instruct", "databricks-meta-llama-3-1-8b-instruct"
PRICES = {BIG: (0.50, 1.50), SMALL: (0.15, 0.45)}   # $ per 1M input / output tokens (illustrative)

def run(endpoint, system, user, max_tokens=300):
    r = client.predict(endpoint=endpoint, inputs={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": max_tokens})
    u = r.get("usage") or {}
    return r["choices"][0]["message"]["content"].strip(), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

def cost(endpoint, tin, tout):
    pin, pout = PRICES[endpoint]
    return (tin * pin + tout * pout) / 1e6

ticket = "I was charged twice for order 1002, please refund the extra payment."
answer, tin, tout = run(BIG, "You are a support assistant.", ticket)
print(answer, f"\n\ninput tokens={tin}, output tokens={tout}, cost=${cost(BIG, tin, tout):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Lever: a shorter prompt
# MAGIC **What:** the same classification task with a long, chatty system prompt (repeated rules, pasted policy, 10
# MAGIC examples) vs. a tight one. Input tokens are paid on **every** request.
# MAGIC **How to read the output:** the long prompt uses several times more input tokens for the same answer.

# COMMAND ----------

POLICY_DUMP = ("Our company values every customer. " * 30) + "Returns within 30 days. Refunds in 5-7 days. " * 5
EXAMPLES = "\n".join(f"Example {i}: 'my order is late' -> Shipping" for i in range(10))
LONG_PROMPT = (f"You are a very helpful, friendly and professional assistant. {POLICY_DUMP}\n{EXAMPLES}\n"
               "Please classify the ticket into Shipping, Payment, Product Defect, Return or Other. Reply with the category only.")
SHORT_PROMPT = "Classify the ticket: Shipping, Payment, Product Defect, Return or Other. Reply with the category only."

for name, prompt in [("long", LONG_PROMPT), ("short", SHORT_PROMPT)]:
    out, tin, tout = run(BIG, prompt, ticket, max_tokens=10)
    print(f"{name:6s} answer={out:10s} input_tokens={tin:5d} cost=${cost(BIG, tin, tout):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Lever: limit output tokens
# MAGIC **What:** output tokens are the most expensive ones (3× input here). Ask for a concise answer **and** set
# MAGIC `max_tokens` as a hard cap.
# MAGIC **How to read the output:** the open-ended reply uses far more output tokens than the "max 2 sentences" one.

# COMMAND ----------

for name, system, cap in [("open-ended", "You are a support assistant.", 500),
                          ("concise", "You are a support assistant. Answer in max 2 sentences.", 80)]:
    out, tin, tout = run(BIG, system, ticket, max_tokens=cap)
    print(f"{name:10s} output_tokens={tout:4d} cost=${cost(BIG, tin, tout):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Lever: a smaller model when it's good enough
# MAGIC **What:** classify 8 tickets with the 70B and the 8B model. If accuracy is equal, the small model is the cheaper
# MAGIC (and faster) choice for this task. (Section 6.1 does this comparison in depth.)
# MAGIC **How to read the output:** accuracy and total cost per model.

# COMMAND ----------

labelled = [("My parcel is a week late.", "Shipping"), ("I was charged twice.", "Payment"),
            ("The screen cracked on day one.", "Product Defect"), ("I'd like to send the jacket back.", "Return"),
            ("Tracking hasn't moved in 5 days.", "Shipping"), ("Refund the duplicate payment.", "Payment"),
            ("The kettle stopped heating.", "Product Defect"), ("How do I return shoes?", "Return")]
for endpoint in [BIG, SMALL]:
    try:
        correct, total = 0, 0.0
        for text, label in labelled:
            out, tin, tout = run(endpoint, SHORT_PROMPT, text, max_tokens=10)
            correct += out.strip(".") == label
            total += cost(endpoint, tin, tout)
        print(f"{endpoint:45s} accuracy={correct}/{len(labelled)} cost=${total:.6f}")
    except Exception as e:
        print(endpoint, "unavailable:", str(e)[:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Lever: cache repeated questions
# MAGIC **What:** many users ask the same thing ("where is my refund?"). A cache stores the answer for a **normalised**
# MAGIC question (lower-case, no punctuation) and skips the model call on a hit. In production use a shared store (e.g. a
# MAGIC Lakebase/Delta table or Redis) with an expiry time, and **never cache personalised answers** (order status).
# MAGIC **How to read the output:** 8 questions, 3 distinct → 5 cache hits, 3 model calls (~62 % fewer calls).

# COMMAND ----------

import re
cache, calls = {}, 0
def cached_answer(question):
    global calls
    key = re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()
    if key not in cache:
        calls += 1
        cache[key] = run(BIG, "Answer in one sentence. Refunds take 5-7 business days. Returns: 30 days.", question, 60)[0]
    return cache[key]

questions = ["When do I get my refund?", "when do i get my refund", "How long do returns take?",
             "When do I get my refund?!", "How long do returns take", "Do you ship to Canada?",
             "do you ship to canada?", "When do I get my refund"]
for q in questions:
    cached_answer(q)
print(f"questions={len(questions)} model_calls={calls} cache_hits={len(questions) - calls} "
      f"saved={1 - calls / len(questions):.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Monthly projection: baseline vs. optimised
# MAGIC **What:** 100,000 requests per day for 30 days. Baseline = long prompt, open-ended output, 70B, no cache. Optimised =
# MAGIC short prompt, concise output, small model for classification, 40 % cache hit rate. Token counts are typical values
# MAGIC from the steps above.
# MAGIC **How to read the output:** the optimised setup is typically **>10× cheaper** – without changing what users see for
# MAGIC this simple task.

# COMMAND ----------

REQUESTS = 100_000 * 30
baseline = REQUESTS * cost(BIG, 900, 300)
optimised = REQUESTS * (1 - 0.40) * cost(SMALL, 120, 60)
print(f"baseline  ${baseline:,.0f} / month")
print(f"optimised ${optimised:,.0f} / month   ({baseline / optimised:.0f}x cheaper)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Track real usage in system tables
# MAGIC **What:** with AI Gateway **usage tracking** enabled, every call is recorded in `system.serving.endpoint_usage`
# MAGIC (tokens, requester, time) and `system.serving.served_entities` (which endpoint/model). Dollar cost is in
# MAGIC `system.billing.usage`.
# MAGIC **How to read the output:** tokens per endpoint for the last 7 days, and model-serving DBUs per day. If system tables
# MAGIC aren't enabled for you, the cells print a message.

# COMMAND ----------

try:
    display(spark.sql("""
        SELECT e.endpoint_name, date(u.request_time) AS day, count(*) AS requests,
               sum(u.input_token_count) AS input_tokens, sum(u.output_token_count) AS output_tokens
        FROM system.serving.endpoint_usage u
        JOIN system.serving.served_entities e ON u.served_entity_id = e.served_entity_id
        WHERE u.request_time >= current_date() - INTERVAL 7 DAYS
        GROUP BY 1, 2 ORDER BY day DESC, output_tokens DESC"""))
except Exception as e:
    print("system.serving tables not available:", str(e)[:200])

try:
    display(spark.sql("""
        SELECT usage_date, sku_name, round(sum(usage_quantity), 2) AS dbus
        FROM system.billing.usage
        WHERE billing_origin_product = 'MODEL_SERVING' AND usage_date >= current_date() - INTERVAL 30 DAYS
        GROUP BY 1, 2 ORDER BY usage_date DESC"""))
except Exception as e:
    print("system.billing.usage not available:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Choose the serving mode + cost checklist
# MAGIC | Traffic pattern | Choose |
# MAGIC |---|---|
# MAGIC | Low or spiky, prototype | **pay-per-token** FMAPI – no idle cost |
# MAGIC | High and steady, needs guaranteed throughput / latency | **provisioned throughput** |
# MAGIC | Custom model/agent with idle periods | custom endpoint with **scale to zero** (accept cold starts) |
# MAGIC | Bulk processing of tables | **batch `ai_query`** in a scheduled job |
# MAGIC
# MAGIC **Checklist**
# MAGIC - Short prompts, few-but-relevant chunks, `max_tokens`, concise answers.
# MAGIC - Smallest model that passes the quality gate; route only hard requests to big models.
# MAGIC - Cache non-personalised answers.
# MAGIC - AI Gateway **rate limits** per user/endpoint; **usage tracking** on; tag endpoints for chargeback; budget alerts.
# MAGIC - Delete idle endpoints, Vector Search endpoints and continuous pipelines you don't use.
# MAGIC
# MAGIC **Exam takeaways:** cost = tokens × price (or uptime × size). Reduce tokens, pick smaller models, cache, batch;
# MAGIC choose pay-per-token vs provisioned throughput by traffic; enforce **rate limits** and monitor **usage / billing
# MAGIC system tables** via AI Gateway.
