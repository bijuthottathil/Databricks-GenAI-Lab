# Databricks notebook source
# MAGIC %md
# MAGIC # 04 – Translate business goals into AI pipeline inputs and outputs
# MAGIC **Scenario:** automate classification and prioritization of 10,000 complaints.
# MAGIC
# MAGIC | Stage | Input | Output |
# MAGIC |---|---|---|
# MAGIC | Data ingestion | Customer messages | Raw ticket records |
# MAGIC | Preprocessing | Raw records | Cleaned messages |
# MAGIC | Classification | Cleaned message | Category |
# MAGIC | Sentiment | Customer message | Sentiment |
# MAGIC | Priority assignment | Category + sentiment | Priority |
# MAGIC | Policy retrieval | Category | Relevant policy |
# MAGIC | Response generation | Ticket + policy | Recommended response |
# MAGIC | Storage | AI results | Delta table |
# MAGIC
# MAGIC > Exam tip: always identify **source data**, **transformation / model task**, **expected output**, and **destination / consumer**.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plain-code equivalent of `ask_llm` (for understanding)
# MAGIC `ask_llm(prompt)` from `00_setup/config` is just this: build a chat `messages` list, send it to the
# MAGIC model serving endpoint, and pull the reply text out of the response. (`ask_llm` also retries on HTTP 429 rate limits.)
# MAGIC
# MAGIC This notebook's SQL `ai_query('<endpoint>', prompt)` is the same call, run in parallel over every row of a table.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")          # authenticated client for this workspace

prompt = "In one sentence, recommend the next action for this ticket: I was charged twice for the same order."

# 1. Build the chat messages (system = role/rules, user = the actual request)
messages = [
    # optional: {"role": "system", "content": "You are ..."},
    {"role": "user", "content": prompt},
]

# 2. Call the endpoint (OpenAI-compatible chat format)
response = client.predict(
    endpoint=ENDPOINT,                            # e.g. databricks-meta-llama-3-3-70b-instruct
    inputs={"messages": messages, "temperature": 0.0, "max_tokens": 500},
)

# 3. The raw response: the answer is in choices[0].message.content
print(json.dumps(response, indent=2, default=str)[:1500])

# 4. Extract just the text - this is what ask_llm returns
answer = response["choices"][0]["message"]["content"].strip()
print("\nAnswer:", answer)

# COMMAND ----------

# Same thing in one line with the helper
print(ask_llm(prompt))

# COMMAND ----------

# MAGIC %md ### Stage 1–2: ingestion + preprocessing

# COMMAND ----------

from pyspark.sql.functions import col, trim

tickets = spark.table("customer_support_tickets")
cleaned = (tickets.select("ticket_id", "customer_message", "product", "order_value")
                  .filter(col("customer_message").isNotNull())
                  .withColumn("customer_message", trim(col("customer_message"))))
cleaned.createOrReplaceTempView("cleaned_tickets")
display(cleaned)

# COMMAND ----------

# MAGIC %md ### Target output contract

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

output_schema = StructType([
    StructField("ticket_id", IntegerType()),
    StructField("category", StringType()),
    StructField("sentiment", StringType()),
    StructField("priority", StringType()),
    StructField("policy_text", StringType()),
    StructField("recommended_action", StringType()),
])
print(output_schema.simpleString())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stages 3–7 as one scalable SQL pipeline (AI Functions run in parallel over all rows)
# MAGIC - Classification + sentiment: `ai_classify`, `ai_analyze_sentiment`
# MAGIC - Priority: deterministic business rule on category + sentiment (no LLM needed)
# MAGIC - Policy retrieval: join to `customer_support_policies`
# MAGIC - Response generation: `ai_query` with the ticket + policy

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE ticket_ai_results AS
WITH classified AS (
  SELECT ticket_id, customer_message, product, order_value,
         ai_classify(customer_message, ARRAY('Shipping','Return','Product Defect','Payment','Wrong Item','Positive Feedback','Other')) AS category,
         initcap(ai_analyze_sentiment(customer_message)) AS sentiment
  FROM cleaned_tickets
),
prioritized AS (
  SELECT *,
         CASE WHEN sentiment = 'Negative' AND category IN ('Shipping','Payment','Product Defect','Wrong Item') THEN 'High'
              WHEN sentiment = 'Positive' THEN 'Low'
              ELSE 'Medium' END AS priority
  FROM classified
)
SELECT CAST(p.ticket_id AS INT) AS ticket_id, p.category, p.sentiment, p.priority,
       coalesce(pol.policy_text, 'No matching policy found.') AS policy_text,
       ai_query('{ENDPOINT}',
         concat('You are a support agent. In one sentence, recommend the next action for this ticket. ',
                'Ticket: ', p.customer_message, ' Category: ', p.category,
                ' Policy: ', coalesce(pol.policy_text, 'none'))) AS recommended_action
FROM prioritized p
LEFT JOIN customer_support_policies pol ON p.category = pol.policy_category
""")

# COMMAND ----------

# MAGIC %md ### Stage 8: storage – results conform to the output contract

# COMMAND ----------

results = spark.table("ticket_ai_results")
assert [f.name for f in results.schema] == [f.name for f in output_schema], "Output does not match contract"
display(results.orderBy("ticket_id"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Consumer view: e.g. a support dashboard showing the high-priority queue
# MAGIC SELECT priority, category, count(*) AS tickets
# MAGIC FROM ticket_ai_results GROUP BY ALL ORDER BY priority, tickets DESC
