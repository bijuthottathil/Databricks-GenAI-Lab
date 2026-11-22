# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.4 – Operations and sequence to write chunked text into Delta Lake tables in Unity Catalog
# MAGIC
# MAGIC **Plain English:** a RAG app needs its chunks stored in a proper table so they can be searched, governed
# MAGIC (who may see what), refreshed when documents change, and synced into a **Vector Search index**.
# MAGIC On Databricks that table is a **Delta table** registered in **Unity Catalog** (`catalog.schema.table`).
# MAGIC
# MAGIC **The sequence (memorise this order):**
# MAGIC 1. **Create the containers** – catalog → schema → volume (for raw files).
# MAGIC 2. **Land raw files** in the Volume.
# MAGIC 3. **Read / parse** files into a DataFrame (one row per document) – keep the file path.
# MAGIC 4. **Clean** the text.
# MAGIC 5. **Chunk** each document (one row per chunk) – with `split` + `posexplode`.
# MAGIC 6. **Add an ID and metadata** – a stable unique `chunk_id`, source path, section, position.
# MAGIC 7. **Create the target Delta table** with a primary key and **Change Data Feed** turned on
# MAGIC    (Vector Search uses Change Data Feed to pick up only what changed).
# MAGIC 8. **Write with MERGE** (upsert) so re-running the pipeline doesn't create duplicates.
# MAGIC 9. **Verify**, then (optionally) create a **Delta Sync Vector Search index** on the table.

# COMMAND ----------

# MAGIC %md ### Step 1 – Create the containers

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "fde_data_prep"
VOLUME = "handbook_files"
CHUNK_TABLE = f"{CATALOG}.{SCHEMA}.handbook_chunks"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
folder = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# COMMAND ----------

# MAGIC %md ### Step 2 – Land raw files in the Volume (here we create two small Markdown documents)

# COMMAND ----------

documents = {
    "shipping_and_returns.md": """# Shipping and Returns

## Shipping
Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs $15.
Orders delayed more than 3 days qualify for a shipping investigation.

## Returns
Customers can return most products within 30 days of delivery for a full refund.
Opened software and gift cards cannot be returned.
""",
    "warranty_and_payments.md": """# Warranty and Payments

## Warranty
Defective products qualify for a free replacement within 15 days of delivery.
After 15 days, the 1-year manufacturer warranty applies.

## Payments
We accept credit cards, debit cards and PayPal.
Duplicate payments are refunded within 3 business days if confirmed.
""",
}
for name, text in documents.items():
    dbutils.fs.put(f"{folder}/{name}", text, overwrite=True)
display(dbutils.fs.ls(folder))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Read the files into Spark (one row per document)
# MAGIC The `binaryFile` reader always gives **exactly one row per file** – the file's raw bytes in a `content` column
# MAGIC and its location in a `path` column. We convert the bytes to text with `cast("string")`.
# MAGIC We keep the path so answers can cite which file they came from.
# MAGIC
# MAGIC *(Why not `spark.read.text`? By default it returns one row per **line**, which would separate each `## heading`
# MAGIC from its paragraph and break the chunking below.)*

# COMMAND ----------

from pyspark.sql import functions as F

raw_docs = (spark.read.format("binaryFile").load(folder)
            .select(F.col("path").alias("source_path"),
                    F.col("content").cast("string").alias("raw_text")))      # bytes -> text
display(raw_docs)
print("Documents read:", raw_docs.count())      # expect 2 (one row per file)

# COMMAND ----------

# MAGIC %md ### Step 4 – Clean the text (squash extra spaces, trim)

# COMMAND ----------

clean_docs = raw_docs.withColumn("text", F.trim(F.regexp_replace("raw_text", r"[ \t]+", " "))).drop("raw_text")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Chunk each document (structure-aware: one chunk per `##` section)
# MAGIC We only use Spark's **built-in text functions** here (no custom Python function / UDF). Built-ins run inside
# MAGIC Spark's engine, so they are faster and behave the same everywhere (including serverless).
# MAGIC
# MAGIC How it works, for one document:
# MAGIC 1. `split(text, "\n## ")` cuts the document wherever a line starts with `## ` → a list of blocks.
# MAGIC    Block 0 is the document title (`# Shipping and Returns`); every other block is one section.
# MAGIC 2. `posexplode` turns that list into **one row per block** and also gives each row its position (`pos`).
# MAGIC 3. The first line of a block is the section name; the rest is the section's text.
# MAGIC 4. We glue them back together as `"Shipping: Standard shipping takes..."` so each chunk makes sense on its own.

# COMMAND ----------

blocks = clean_docs.select(
    "source_path",
    F.posexplode(F.split("text", "\n## ")).alias("pos", "block"),     # one row per "## " section
)

first_line = F.substring_index("block", "\n", 1)                       # text before the first line break
strip_whitespace = lambda c: F.regexp_replace(c, r"^\s+|\s+$", "")    # remove spaces AND line breaks at both ends

chunks = (blocks
    .filter(F.col("pos") > 0)                                           # skip block 0 = the "# Title" line
    .withColumn("chunk_index", (F.col("pos") - 1).cast("int"))          # 0, 1, 2... inside each document
    .withColumn("section", strip_whitespace(first_line))                # e.g. "Shipping"
    .withColumn("body", strip_whitespace(F.expr("substr(block, length(substring_index(block, '\\n', 1)) + 2)")))
    .withColumn("chunk_text", F.concat_ws(": ", "section", "body"))     # "Shipping: Standard shipping takes..."
    .select("source_path", "chunk_index", "section", "chunk_text"))

display(chunks)
print("Chunks created:", chunks.count())      # expect 4 (2 documents x 2 sections)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Add a stable ID and metadata
# MAGIC The `chunk_id` is a fingerprint (hash) of "file path + chunk position". Re-running the pipeline on the same files
# MAGIC produces the **same IDs**, which is what lets MERGE update rows instead of duplicating them.

# COMMAND ----------

chunks_with_ids = (chunks
    .withColumn("chunk_id", F.sha2(F.concat_ws("::", "source_path", F.col("chunk_index").cast("string")), 256))
    .withColumn("doc_name", F.element_at(F.split("source_path", "/"), -1))
    .withColumn("token_estimate", (F.size(F.split("chunk_text", r"\s+")) * 1.3).cast("int"))
    .withColumn("updated_at", F.current_timestamp())
    .select("chunk_id", "doc_name", "source_path", "section", "chunk_index", "chunk_text", "token_estimate", "updated_at"))
display(chunks_with_ids)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Create the target table (once)
# MAGIC - `chunk_id ... NOT NULL` + `PRIMARY KEY` – Vector Search requires a unique key column.
# MAGIC - `delta.enableChangeDataFeed = true` – records every insert/update/delete so a Vector Search index can sync
# MAGIC   only the changes instead of re-reading the whole table.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CHUNK_TABLE} (
  chunk_id        STRING NOT NULL COMMENT 'Stable hash of source_path + chunk_index',
  doc_name        STRING          COMMENT 'File name, used for citations',
  source_path     STRING          COMMENT 'Full path of the source file in the Volume',
  section         STRING          COMMENT 'Heading the chunk came from',
  chunk_index     INT             COMMENT 'Position of the chunk inside its document',
  chunk_text      STRING          COMMENT 'The text that will be embedded and retrieved',
  token_estimate  INT             COMMENT 'Rough token count (words x 1.3)',
  updated_at      TIMESTAMP,
  CONSTRAINT handbook_chunks_pk PRIMARY KEY (chunk_id)
)
COMMENT 'Chunked ShopEase handbook for RAG'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Write with MERGE (upsert)
# MAGIC "If this chunk_id already exists, update it; otherwise insert it." Run this cell twice – the row count stays the same.

# COMMAND ----------

chunks_with_ids.createOrReplaceTempView("new_chunks")
spark.sql(f"""
MERGE INTO {CHUNK_TABLE} AS target
USING new_chunks AS source
ON target.chunk_id = source.chunk_id
WHEN MATCHED AND target.chunk_text <> source.chunk_text THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE
""")
# "NOT MATCHED BY SOURCE THEN DELETE" removes chunks whose document section no longer exists.

# COMMAND ----------

# MAGIC %md ### Step 9 – Verify

# COMMAND ----------

display(spark.sql(f"SELECT doc_name, section, chunk_text, token_estimate FROM {CHUNK_TABLE} ORDER BY doc_name, chunk_index"))
print("Rows:", spark.table(CHUNK_TABLE).count())
print("Change Data Feed on:", spark.sql(f"SHOW TBLPROPERTIES {CHUNK_TABLE}('delta.enableChangeDataFeed')").first().value)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The table's history shows each write (CREATE, MERGE...) - useful for auditing and time travel
# MAGIC DESCRIBE HISTORY workspace.fde_data_prep.handbook_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional – Create a Vector Search index on the table
# MAGIC This needs a Vector Search endpoint (may not exist in Free Edition), so the code is shown but not run.
# MAGIC ```python
# MAGIC %pip install databricks-vectorsearch
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC vsc = VectorSearchClient()
# MAGIC vsc.create_delta_sync_index(
# MAGIC     endpoint_name="my_vs_endpoint",
# MAGIC     index_name="workspace.fde_data_prep.handbook_chunks_index",
# MAGIC     source_table_name="workspace.fde_data_prep.handbook_chunks",
# MAGIC     pipeline_type="TRIGGERED",                       # sync when you ask (or CONTINUOUS)
# MAGIC     primary_key="chunk_id",
# MAGIC     embedding_source_column="chunk_text",            # Databricks computes the embeddings for you
# MAGIC     embedding_model_endpoint_name="databricks-gte-large-en",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Exam takeaways:** volume → read/parse → clean → chunk (`split` + `posexplode`) → stable `chunk_id` + metadata →
# MAGIC Delta table with primary key and Change Data Feed → MERGE → Vector Search Delta Sync index.
