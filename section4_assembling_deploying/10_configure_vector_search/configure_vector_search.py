# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.10 – Configure Vector Search for a solution
# MAGIC ### based on number of embeddings, update frequency, latency and cost
# MAGIC
# MAGIC **The big idea for juniors:** there is no single "right" Vector Search setup. You choose a few settings based on
# MAGIC four questions about the use case – like choosing a car by passengers, distance, speed and budget.
# MAGIC
# MAGIC | Question | Setting it drives |
# MAGIC |---|---|
# MAGIC | **How many embeddings?** (vectors = chunks) | endpoint type, chunk size, embedding dimension |
# MAGIC | **How often does the data change, and how fast must changes appear?** | sync mode: `TRIGGERED` or `CONTINUOUS`; index type |
# MAGIC | **How fast must a search return?** | endpoint type, `num_results`, filters, reranking on/off |
# MAGIC | **What's the budget?** | all of the above – endpoints and continuous pipelines are billed while running |
# MAGIC
# MAGIC ### The settings you can choose
# MAGIC | Setting | Options | Trade-off |
# MAGIC |---|---|---|
# MAGIC | **Endpoint type** | `STANDARD` | lowest query latency; best for interactive apps with up to large (but not huge) vector counts |
# MAGIC | | `STORAGE_OPTIMIZED` | built for very large collections at lower cost per vector; somewhat higher latency |
# MAGIC | **Index type** | Delta Sync (managed embeddings) | easiest – Databricks embeds rows and queries |
# MAGIC | | Delta Sync (self-managed) | you bring vectors (custom model) |
# MAGIC | | Direct Vector Access | app writes vectors via API in real time; no Delta source |
# MAGIC | **Sync mode** | `TRIGGERED` | sync when you call it / on a schedule – cheapest; data can be hours old |
# MAGIC | | `CONTINUOUS` | changes appear within seconds – pipeline always running, costs more |
# MAGIC | **Embedding model / dimension** | e.g. 1024-dim GTE | more dimensions = more storage & slightly slower |
# MAGIC | **Chunk size** | small vs large | smaller = more vectors (precision ↑, storage ↑) |
# MAGIC | **Query options** | `num_results`, `filters`, hybrid, reranking | reranking improves quality but adds latency |
# MAGIC
# MAGIC > Exact limits (max vectors per endpoint, latency figures, prices) change over time – always confirm in the
# MAGIC > current Databricks documentation. This notebook teaches the **reasoning**.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 five example use cases → Step 2 estimate vectors & storage
# MAGIC Step 3 decision function (rules) → Step 4 recommendation table
# MAGIC Step 5 turn a recommendation into the create_* code → Step 6 cost levers checklist
# MAGIC ```
# MAGIC This notebook makes **no** model or Vector Search calls – it's pure reasoning and arithmetic.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Five example use cases
# MAGIC | Field | Meaning |
# MAGIC |---|---|
# MAGIC | `documents`, `avg_tokens_per_doc` | how much text |
# MAGIC | `chunk_tokens` | planned chunk size |
# MAGIC | `freshness` | how quickly changes must be searchable: `seconds`, `hours`, `daily` |
# MAGIC | `latency_ms` | maximum acceptable search time |
# MAGIC | `budget` | `low`, `medium`, `high` |
# MAGIC | `source` | `delta` (data lands in a table) or `app_events` (an app pushes vectors directly) |

# COMMAND ----------

use_cases = [
    # name, documents, avg_tokens_per_doc, chunk_tokens, freshness, latency_ms, budget, source
    ("HR policy assistant",         300,        2_000,  500, "daily",   500, "low",    "delta"),
    ("Customer support RAG",        50_000,     1_500,  400, "hours",   400, "medium", "delta"),
    ("Live product catalogue",      2_000_000,    300,  300, "seconds", 100, "high",   "delta"),
    ("Legal archive search",        40_000_000, 3_000,  500, "daily",  1000, "low",    "delta"),
    ("Personalised feed (real-time)", 5_000_000,  100,  100, "seconds", 100, "high",   "app_events"),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Estimate number of embeddings and storage
# MAGIC **Formulas:**
# MAGIC - vectors ≈ documents × (avg tokens per doc ÷ chunk tokens)
# MAGIC - storage ≈ vectors × dimensions × 4 bytes (float32)
# MAGIC
# MAGIC **How to read the output:** the HR assistant needs ~1,200 vectors (tiny); the legal archive ~240 million (huge).
# MAGIC That 200,000× difference is why one setup can't fit all.

# COMMAND ----------

import math

DIMENSIONS = 1024
estimates = []
for name, docs, avg_tok, chunk_tok, fresh, lat, budget, source in use_cases:
    vectors = docs * math.ceil(avg_tok / chunk_tok)
    storage_gb = vectors * DIMENSIONS * 4 / 1e9
    estimates.append((name, vectors, round(storage_gb, 2), fresh, lat, budget, source))

display(spark.createDataFrame(estimates, "use_case string, vectors long, storage_gb double, freshness string, "
                                         "latency_ms int, budget string, source string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Turn requirements into settings (simple rules)
# MAGIC **The rules, in plain words:**
# MAGIC 1. **Index type:** data pushed by an app in real time → *Direct Vector Access*; otherwise *Delta Sync with managed
# MAGIC    embeddings* (simplest).
# MAGIC 2. **Endpoint type:** very many vectors (here: > 100 million) and latency tolerance ≥ ~500 ms, or low budget with
# MAGIC    large volume → `STORAGE_OPTIMIZED`; otherwise `STANDARD` (fastest).
# MAGIC 3. **Sync mode:** freshness in seconds → `CONTINUOUS`; hours/daily → `TRIGGERED` on a schedule (cheaper).
# MAGIC 4. **Reranking:** only if the latency budget leaves room (≥ 300 ms).
# MAGIC 5. **Cost tip:** if storage is huge and budget is low → consider bigger chunks or a smaller embedding dimension.
# MAGIC
# MAGIC The 100-million threshold is an **example** for teaching – use the documented limits for real designs.

# COMMAND ----------

def recommend(vectors, freshness, latency_ms, budget, source):
    index_type = "Direct Vector Access" if source == "app_events" else "Delta Sync (managed embeddings)"
    if (vectors > 100_000_000 and latency_ms >= 500) or (budget == "low" and vectors > 10_000_000):
        endpoint = "STORAGE_OPTIMIZED"
    else:
        endpoint = "STANDARD"
    if index_type == "Direct Vector Access":
        sync = "n/a (app upserts vectors)"
    else:
        sync = "CONTINUOUS" if freshness == "seconds" else f"TRIGGERED ({'hourly' if freshness == 'hours' else 'daily'} job)"
    rerank = "yes" if latency_ms >= 300 else "no (latency budget too tight)"
    tip = ("bigger chunks / smaller dimension to cut storage" if vectors > 10_000_000 and budget == "low"
           else "filters on metadata to shrink the search space" if vectors > 1_000_000 else "-")
    return index_type, endpoint, sync, rerank, tip

rows = [(name, vectors, *recommend(vectors, fresh, lat, budget, source))
        for name, vectors, gb, fresh, lat, budget, source in estimates]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – The recommendations
# MAGIC **How to read the output (one row per use case):**
# MAGIC - **HR policy assistant** – tiny, daily updates → `STANDARD`, `TRIGGERED` daily. Cheapest sensible setup.
# MAGIC - **Customer support RAG** – medium size, updates every few hours → `STANDARD`, `TRIGGERED` hourly, reranking on.
# MAGIC - **Live product catalogue** – prices/stock change constantly, 100 ms target → `STANDARD` + `CONTINUOUS`, no reranking.
# MAGIC - **Legal archive** – hundreds of millions of vectors, latency not critical, low budget → `STORAGE_OPTIMIZED`,
# MAGIC   `TRIGGERED` daily.
# MAGIC - **Personalised feed** – events pushed by an app → **Direct Vector Access**.

# COMMAND ----------

display(spark.createDataFrame(rows, "use_case string, vectors long, index_type string, endpoint_type string, "
                                    "sync_mode string, reranking string, cost_tip string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – From recommendation to code (shown, not run)
# MAGIC **Customer support RAG** (`STANDARD`, Delta Sync managed, `TRIGGERED`):
# MAGIC ```python
# MAGIC vsc.create_endpoint(name="support_vs", endpoint_type="STANDARD")
# MAGIC idx = vsc.create_delta_sync_index(endpoint_name="support_vs", index_name="cat.sch.support_index",
# MAGIC         source_table_name="cat.sch.support_chunks", primary_key="chunk_id",
# MAGIC         embedding_source_column="chunk_text", embedding_model_endpoint_name="databricks-gte-large-en",
# MAGIC         pipeline_type="TRIGGERED")
# MAGIC # schedule a job every hour that runs:  idx.sync()
# MAGIC ```
# MAGIC **Live product catalogue** (`CONTINUOUS`): same call with `pipeline_type="CONTINUOUS"` – no sync job needed.
# MAGIC
# MAGIC **Legal archive** (`STORAGE_OPTIMIZED`): `vsc.create_endpoint(name="archive_vs", endpoint_type="STORAGE_OPTIMIZED")`.
# MAGIC
# MAGIC **Personalised feed** (Direct Vector Access):
# MAGIC ```python
# MAGIC idx = vsc.create_direct_access_index(endpoint_name="feed_vs", index_name="cat.sch.feed_index",
# MAGIC         primary_key="item_id", embedding_dimension=1024, embedding_vector_column="vector",
# MAGIC         schema={"item_id": "string", "vector": "array<float>", "category": "string"})
# MAGIC idx.upsert([{"item_id": "a1", "vector": [...], "category": "sport"}])
# MAGIC ```
# MAGIC
# MAGIC ### Step 6 – Cost levers checklist
# MAGIC | Lever | Effect |
# MAGIC |---|---|
# MAGIC | `TRIGGERED` instead of `CONTINUOUS` | no always-on pipeline |
# MAGIC | Bigger chunks / fewer duplicates / filter junk (Section 2) | fewer vectors to store and embed |
# MAGIC | Smaller embedding dimension | less storage per vector |
# MAGIC | `STORAGE_OPTIMIZED` for huge, latency-tolerant data | lower cost per vector |
# MAGIC | Share one endpoint across several small indexes | fewer endpoints running |
# MAGIC | Delete unused indexes/endpoints | stop paying for idle compute |
# MAGIC | Only return needed `columns`, small `num_results` | faster, cheaper queries |
# MAGIC
# MAGIC **Exam takeaways:** pick **endpoint type** from vector count + latency + budget (`STANDARD` = low latency,
# MAGIC `STORAGE_OPTIMIZED` = very large scale, lower cost), **sync mode** from update frequency/freshness (`TRIGGERED` cheap,
# MAGIC `CONTINUOUS` near-real-time), **index type** from where vectors come from, and tune chunk size, dimension, filters
# MAGIC and reranking for the latency/quality/cost balance.
