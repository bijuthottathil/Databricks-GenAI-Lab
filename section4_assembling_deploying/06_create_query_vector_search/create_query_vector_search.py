# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.6 – Create and query a Vector Search index
# MAGIC
# MAGIC **The big idea for juniors:** in 4.4 we searched 6 chunks in memory. Real apps have thousands or millions of
# MAGIC chunks, which change over time. **Mosaic AI Vector Search** stores the vectors in an **index** that stays in sync
# MAGIC with a Delta table and answers "find the most similar chunks" in milliseconds.
# MAGIC
# MAGIC ```
# MAGIC Delta table (chunks, CDF on) ──sync──► Vector Search INDEX (on an ENDPOINT) ◄── similarity_search("question")
# MAGIC        ▲ you insert/update rows            embeddings computed for you              returns top-k rows
# MAGIC ```
# MAGIC
# MAGIC > ⚠️ **Cost note:** a Vector Search **endpoint** is compute that keeps running (and is billed) until you delete it.
# MAGIC > The last cell deletes the index and endpoint – set `CLEANUP = True` there when you're done.
# MAGIC > Some workspace tiers (e.g. Free Edition) may limit or not offer Vector Search.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Vector Search endpoint** | The compute that hosts one or more indexes and answers queries |
# MAGIC | **Index** | The searchable copy of your vectors (+ the columns you want returned) |
# MAGIC | **Delta Sync index** | An index that automatically follows a Delta table |
# MAGIC | **Managed embeddings** | You give a text column + an embedding endpoint; Databricks computes the vectors |
# MAGIC | **Change Data Feed (CDF)** | A Delta table feature that records changed rows – required so the index can sync only changes |
# MAGIC | **Primary key** | A unique id per row, so updates/deletes find the right vector |
# MAGIC | **TRIGGERED / CONTINUOUS** | Sync when you ask (cheaper) vs. sync automatically within seconds (more expensive) |
# MAGIC | **Filter** | Restrict results by a column value, e.g. only `category = 'returns'` |
# MAGIC | **Hybrid search** | Combine vector similarity with keyword matching |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0 install → Step 1 source Delta table (CDF on) → Step 2 create/wait for endpoint
# MAGIC Step 3 create Delta Sync index with managed embeddings → wait until ready
# MAGIC Step 4 similarity search → Step 5 filters → Step 6 hybrid search
# MAGIC Step 7 add a row + sync → search finds it → Step 8 cleanup
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The source Delta table
# MAGIC **What:** policy chunks with `chunk_id` (primary key), `category` (for filtering), `source`, `chunk_text`.
# MAGIC **Must-have:** `delta.enableChangeDataFeed = true` – Delta Sync indexes require it.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_deploy"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.policy_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.policy_chunks_index"
VS_ENDPOINT = "fde_vs_endpoint"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
# CREATE ... IF NOT EXISTS + MERGE (not CREATE OR REPLACE) so re-running never breaks an index that already follows this table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
  chunk_id STRING NOT NULL, category STRING, source STRING, chunk_text STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
spark.sql(f"""
MERGE INTO {SOURCE_TABLE} t
USING (SELECT * FROM VALUES
 ('ret-1','returns','returns_policy.md','Most products can be returned within 30 days of delivery.'),
 ('ret-2','returns','returns_policy.md','Opened software and gift cards cannot be returned.'),
 ('shp-1','shipping','shipping_policy.md','Standard shipping takes 3 to 5 business days; express takes 1 to 2 business days.'),
 ('shp-2','shipping','shipping_policy.md','Orders delayed more than 3 days qualify for an investigation and a $10 store credit.'),
 ('war-1','warranty','warranty.md','Defective products are replaced free within 15 days of delivery.'),
 ('war-2','warranty','warranty.md','Accidental damage is covered only with the ShopEase Care protection plan.'),
 ('pay-1','payments','payments.md','Duplicate charges are refunded within 3 business days once confirmed.'),
 ('pay-2','payments','payments.md','We accept credit cards, debit cards and PayPal.')
 AS v(chunk_id, category, source, chunk_text)) s
ON t.chunk_id = s.chunk_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")
display(spark.table(SOURCE_TABLE).orderBy("chunk_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Create (or reuse) a Vector Search endpoint
# MAGIC **What:** `endpoint_type="STANDARD"` – low-latency endpoint (see 4.10 for when to use `STORAGE_OPTIMIZED`).
# MAGIC Creating an endpoint can take **several minutes**; the loop checks its state every 20 seconds.
# MAGIC **How to read the output:** `Endpoint state: ONLINE`.

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("Creating endpoint", VS_ENDPOINT, "...")

for _ in range(60):
    state = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state")
    if state == "ONLINE":
        break
    time.sleep(20)
print("Endpoint state:", state)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Create a Delta Sync index with managed embeddings
# MAGIC **Arguments explained:**
# MAGIC | Argument | Meaning |
# MAGIC |---|---|
# MAGIC | `source_table_name` | the Delta table to follow |
# MAGIC | `primary_key` | unique id column (`chunk_id`) |
# MAGIC | `embedding_source_column` | the **text** column to embed (`chunk_text`) |
# MAGIC | `embedding_model_endpoint_name` | which model computes the vectors – also used for query text |
# MAGIC | `pipeline_type="TRIGGERED"` | sync only when we call `sync()` – cheapest |
# MAGIC
# MAGIC The first sync embeds every row, which takes a few minutes. The loop waits until the index reports `ready`.

# COMMAND ----------

existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
if INDEX_NAME not in existing_indexes:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
        pipeline_type="TRIGGERED",
    )
    print("Creating index", INDEX_NAME, "...")
else:
    print("Index already exists – reusing it")

index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
for _ in range(60):
    status = index.describe().get("status", {})
    if status.get("ready"):
        break
    time.sleep(20)
print("Index ready:", status.get("ready"), "| state:", status.get("detailed_state"), "| rows:", status.get("indexed_row_count"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Similarity search
# MAGIC **What:** `similarity_search(query_text=..., columns=[...], num_results=k)`. Because the index uses managed
# MAGIC embeddings, we pass **plain text** – the index embeds the question with the same model.
# MAGIC **How to read the output:** rows ordered by `score` (higher = more similar). For "my order is late" the top row
# MAGIC should be `shp-2` (the $10 credit rule).
# MAGIC
# MAGIC **About the result format:** `manifest.columns` = column names, `result.data_array` = rows (the score is added as
# MAGIC the last column). The `show()` function turns that into a readable table.

# COMMAND ----------

def show(results):
    cols = [c["name"] for c in results["manifest"]["columns"]]
    rows = results.get("result", {}).get("data_array", [])
    display(spark.createDataFrame([[str(v) for v in r] for r in rows], ", ".join(f"`{c}` string" for c in cols)) if rows
            else spark.createDataFrame([("no results",)], "message string"))

results = index.similarity_search(query_text="My order is a week late, do I get anything?",
                                  columns=["chunk_id", "category", "chunk_text"], num_results=3)
show(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Search with a metadata filter
# MAGIC **What:** `filters={"category": "warranty"}` – only rows where category = warranty are considered.
# MAGIC **Why:** narrow the search when you already know the topic (from the user's choice, intent detection, or
# MAGIC permissions), which improves precision.
# MAGIC **How to read the output:** only `war-1` and `war-2` can appear.

# COMMAND ----------

show(index.similarity_search(query_text="I dropped my laptop, is that covered?",
                             columns=["chunk_id", "category", "chunk_text"], num_results=3,
                             filters={"category": "warranty"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Hybrid search (vector + keyword)
# MAGIC **What:** `query_type="hybrid"` combines meaning-based similarity with exact keyword matches.
# MAGIC **Why:** pure vectors can miss exact terms such as product codes or names like "PayPal".

# COMMAND ----------

try:
    show(index.similarity_search(query_text="PayPal", columns=["chunk_id", "chunk_text"], num_results=2, query_type="hybrid"))
except Exception as e:
    print("Hybrid search not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Update the source table and sync the index
# MAGIC **What:** add a new policy row → call `index.sync()` (TRIGGERED mode) → wait → search finds it.
# MAGIC **Why:** this is how the knowledge base stays current – you never edit the index directly, you edit the **table**.
# MAGIC Thanks to CDF, only the changed row is re-embedded.

# COMMAND ----------

spark.sql(f"""
MERGE INTO {SOURCE_TABLE} t
USING (SELECT 'shp-3' AS chunk_id, 'shipping' AS category, 'shipping_policy.md' AS source,
              'International orders take 7 to 14 business days and may incur customs fees.' AS chunk_text) s
ON t.chunk_id = s.chunk_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")
index.sync()
for _ in range(30):
    time.sleep(20)
    if index.describe().get("status", {}).get("indexed_row_count", 0) >= 9:
        break
show(index.similarity_search(query_text="How long does shipping to another country take?",
                             columns=["chunk_id", "chunk_text"], num_results=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Cleanup (stop paying for the endpoint)
# MAGIC Set `CLEANUP = True` and run this cell when you have finished **all** Section 4 notebooks that use the index
# MAGIC (4.12 can reuse it).

# COMMAND ----------

CLEANUP = False
if CLEANUP:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
    vsc.delete_endpoint(VS_ENDPOINT)
    print("Deleted index and endpoint")
else:
    print("Cleanup skipped – endpoint", VS_ENDPOINT, "is still running")

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Steps: Delta table with **CDF + primary key** → **endpoint** → **Delta Sync index** (managed embeddings: text
# MAGIC   column + embedding endpoint) → `similarity_search`.
# MAGIC - Query with `query_text` (managed embeddings) or `query_vector` (self-managed); add `filters`, `num_results`,
# MAGIC   `query_type="hybrid"`.
# MAGIC - Keep the index fresh by changing the **source table** and syncing (TRIGGERED: `index.sync()`; CONTINUOUS: automatic).
