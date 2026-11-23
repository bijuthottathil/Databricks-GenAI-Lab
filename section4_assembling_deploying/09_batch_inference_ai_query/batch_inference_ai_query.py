# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.9 – Identify batch inference workloads and apply `ai_query()` appropriately
# MAGIC
# MAGIC **The big idea for juniors:** not every GenAI job is a chatbot. Many jobs are **"do this to every row of a table"**:
# MAGIC classify 1 million tickets, summarise last night's reviews, extract fields from all invoices. That's **batch
# MAGIC inference** – no user is waiting, the input is a table, the output is a table.
# MAGIC
# MAGIC **`ai_query()`** is a SQL function that calls a model serving endpoint **for every row**, in parallel, from SQL or
# MAGIC a DataFrame – no Python loop, no extra service.
# MAGIC
# MAGIC ### Batch or real-time?
# MAGIC | Question | Batch (`ai_query`) | Real-time (serving endpoint / app) |
# MAGIC |---|---|---|
# MAGIC | Is a person waiting for this answer right now? | No | Yes |
# MAGIC | Where is the input? | Already in a table / files | Arrives one request at a time |
# MAGIC | How many items? | Thousands to millions | One per request |
# MAGIC | When must it finish? | Minutes/hours (scheduled job) | Seconds |
# MAGIC | Examples | nightly ticket tagging, backfill sentiment, embed a document archive | chat assistant, live agent |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **`ai_query(endpoint, request)`** | Call a serving endpoint once per row; returns the model's output |
# MAGIC | **`modelParameters`** | Settings like `max_tokens`, `temperature` passed to the model |
# MAGIC | **`responseFormat`** | Ask for structured (JSON) output |
# MAGIC | **`failOnError => false`** | Don't stop the whole query if one row fails – return the error for that row instead |
# MAGIC | **Task-specific AI functions** | `ai_classify`, `ai_extract`, `ai_summarize`, `ai_analyze_sentiment` – simpler, fixed tasks |
# MAGIC | **CTAS** | `CREATE TABLE ... AS SELECT` – save the results as a new table |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 tickets table (12 rows) → Step 2 ai_query classify → Step 3 structured JSON output
# MAGIC Step 4 failOnError for robust jobs → Step 5 embeddings with ai_query → Step 6 save results (CTAS)
# MAGIC Step 7 task-specific functions vs ai_query → Step 8 scheduling + cost tips
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The input table
# MAGIC **What:** 12 support tickets in `workspace.fde_deploy.batch_tickets`. In real life this would be millions of rows
# MAGIC landing every day.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_deploy"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

tickets = [
    (1, "My order is two weeks late and nobody answers my emails."),
    (2, "I was charged twice for the same headphones."),
    (3, "The laptop screen flickers after two days of use."),
    (4, "I want to send back the shoes, they are too small."),
    (5, "Tracking says delivered but there is no package."),
    (6, "Great service, the replacement arrived quickly. Thanks!"),
    (7, "The keyboard has three keys that do not work."),
    (8, "Refund the duplicate payment please."),
    (9, "Can I return a jacket I bought 20 days ago?"),
    (10, "Package arrived damaged and the monitor is cracked."),
    (11, "Why was my card declined at checkout?"),
    (12, "Express shipping took five days instead of two."),
]
spark.createDataFrame(tickets, "ticket_id int, ticket_text string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.batch_tickets")
display(spark.table(f"{CATALOG}.{SCHEMA}.batch_tickets"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – `ai_query` to classify every row
# MAGIC **SQL walk-through:**
# MAGIC - `ai_query('<endpoint>', <prompt>)` – the prompt is built per row with `concat(...)`.
# MAGIC - `modelParameters => named_struct('max_tokens', 10, 'temperature', 0.0)` – short, repeatable answers.
# MAGIC - Databricks sends the rows to the endpoint **in parallel** and returns one answer per row.
# MAGIC
# MAGIC **How to read the output:** a `category` next to each ticket, e.g. ticket 2 → `Payment`, ticket 3 →
# MAGIC `Product Defect`. 12 rows = 12 model calls, but you wrote zero loops.

# COMMAND ----------

classified = spark.sql(f"""
SELECT ticket_id, ticket_text,
       ai_query(
         'databricks-meta-llama-3-3-70b-instruct',
         concat('Classify this support ticket into exactly one of: Shipping, Payment, Product Defect, Return, Other. ',
                'Reply with the category only.\\n\\nTicket: ', ticket_text),
         modelParameters => named_struct('max_tokens', 10, 'temperature', 0.0)
       ) AS category
FROM {CATALOG}.{SCHEMA}.batch_tickets
""")
display(classified)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Several fields at once with JSON output
# MAGIC **What:** ask for a JSON object, then turn it into real columns with `from_json`.
# MAGIC **Why:** one model call per row gives you category, sentiment and urgency together – cheaper than three calls.
# MAGIC **How to read the output:** proper columns `category`, `sentiment`, `urgent` you can filter and aggregate.

# COMMAND ----------

structured = spark.sql(f"""
WITH raw AS (
  SELECT ticket_id, ticket_text,
         ai_query(
           'databricks-meta-llama-3-3-70b-instruct',
           concat('Return ONLY JSON {{"category": "Shipping|Payment|Product Defect|Return|Other", ',
                  '"sentiment": "positive|neutral|negative", "urgent": true|false}} for this ticket: ', ticket_text),
           modelParameters => named_struct('max_tokens', 60, 'temperature', 0.0)
         ) AS answer
  FROM {CATALOG}.{SCHEMA}.batch_tickets
)
SELECT ticket_id, ticket_text, parsed.category, parsed.sentiment, parsed.urgent
FROM (
  SELECT *, from_json(substring(answer, instr(answer, '{{'), length(answer)),
                      'category STRING, sentiment STRING, urgent BOOLEAN') AS parsed
  FROM raw
)
""")
display(structured)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Robust batch jobs: `failOnError => false`
# MAGIC **What:** with `failOnError => false`, `ai_query` returns a struct `{result, errorMessage}` instead of failing the
# MAGIC whole query when one row has a problem (e.g. a timeout or an over-long input).
# MAGIC **Why:** in a 1-million-row job you don't want row 999,999 to throw away all the work. Save the errors and retry
# MAGIC only those rows later.
# MAGIC **How to read the output:** `result` filled and `errorMessage` empty (`null`) for successful rows.

# COMMAND ----------

display(spark.sql(f"""
SELECT ticket_id, response.result AS summary, response.errorMessage AS error
FROM (
  SELECT ticket_id,
         ai_query('databricks-meta-llama-3-3-70b-instruct',
                  concat('Summarise in max 8 words: ', ticket_text),
                  modelParameters => named_struct('max_tokens', 20, 'temperature', 0.0),
                  failOnError => false) AS response
  FROM {CATALOG}.{SCHEMA}.batch_tickets
)
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Batch embeddings with `ai_query`
# MAGIC **What:** the same function works with an **embedding** endpoint – it returns an `ARRAY<FLOAT>` per row.
# MAGIC **Why:** embedding a whole document archive (e.g. for a self-managed Vector Search index) is a classic batch job.
# MAGIC **How to read the output:** `dims = 1024` for every row.

# COMMAND ----------

display(spark.sql(f"""
SELECT ticket_id, size(embedding) AS dims, slice(embedding, 1, 3) AS first_3_values
FROM (SELECT ticket_id, ai_query('databricks-gte-large-en', ticket_text) AS embedding
      FROM {CATALOG}.{SCHEMA}.batch_tickets)
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Save the results as a table (CTAS)
# MAGIC **What:** `CREATE OR REPLACE TABLE ... AS SELECT ai_query(...)` – the standard pattern for a scheduled job.
# MAGIC Downstream dashboards then read the **table**; nobody calls the model again.
# MAGIC **How to read the output:** a count per category – the kind of summary a support manager wants each morning.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.batch_tickets_classified AS
SELECT ticket_id, ticket_text,
       trim(ai_query('databricks-meta-llama-3-3-70b-instruct',
                     concat('Classify into exactly one of: Shipping, Payment, Product Defect, Return, Other. ',
                            'Reply with the category only.\\n\\nTicket: ', ticket_text),
                     modelParameters => named_struct('max_tokens', 10, 'temperature', 0.0))) AS category,
       current_timestamp() AS classified_at
FROM {CATALOG}.{SCHEMA}.batch_tickets
""")
display(spark.sql(f"SELECT category, count(*) AS tickets FROM {CATALOG}.{SCHEMA}.batch_tickets_classified GROUP BY category ORDER BY tickets DESC"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Task-specific AI functions vs. `ai_query`
# MAGIC **What:** for common tasks Databricks has ready-made functions – no prompt or endpoint to choose.
# MAGIC | Use | When |
# MAGIC |---|---|
# MAGIC | `ai_classify`, `ai_analyze_sentiment`, `ai_summarize`, `ai_extract`, `ai_translate`, `ai_mask` | Standard task, default model is fine |
# MAGIC | `ai_query` | You need your own prompt, a specific model, structured output, embeddings, or **your own custom/agent endpoint** |
# MAGIC
# MAGIC **How to read the output:** a label per ticket from `ai_classify` and a sentiment from `ai_analyze_sentiment`.

# COMMAND ----------

try:
    display(spark.sql(f"""
    SELECT ticket_id, ticket_text,
           ai_classify(ticket_text, ARRAY('Shipping', 'Payment', 'Product Defect', 'Return', 'Other')) AS ai_classify_label,
           ai_analyze_sentiment(ticket_text) AS sentiment
    FROM {CATALOG}.{SCHEMA}.batch_tickets
    """))
except Exception as e:
    print("Task-specific AI functions not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Running it in production
# MAGIC - **Schedule** the CTAS/`INSERT ... SELECT ai_query(...)` as a **Lakeflow Job** or a SQL query schedule; process
# MAGIC   only **new** rows (e.g. `WHERE ticket_id NOT IN (SELECT ticket_id FROM classified)` or streaming tables).
# MAGIC - For very large jobs use **provisioned throughput** endpoints – pay-per-token endpoints have rate limits.
# MAGIC - Keep prompts short and set `max_tokens` – cost scales with tokens × rows.
# MAGIC - Test on `LIMIT 100` first, check the output, then run on everything.
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - **Batch inference** = table in, table out, no user waiting, large volume, scheduled → use **`ai_query()`** in
# MAGIC   SQL / DataFrames; real-time chat → serving endpoint.
# MAGIC - `ai_query(endpoint, request, modelParameters => ..., responseFormat => ..., failOnError => false)` works with
# MAGIC   chat, embedding and **custom** endpoints.
# MAGIC - Save results with CTAS; prefer task-specific AI functions for simple standard tasks.
