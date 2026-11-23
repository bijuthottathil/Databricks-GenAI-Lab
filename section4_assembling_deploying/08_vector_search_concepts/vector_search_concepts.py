# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.8 – Explain the key concepts and components of Mosaic AI Vector Search
# MAGIC
# MAGIC **The big idea for juniors:** Vector Search answers one question very fast: *"Which stored texts mean something
# MAGIC similar to this question?"* This notebook explains each part by building a **tiny version in plain Python**, so you
# MAGIC understand what the real service does for you. (No Vector Search endpoint is created here – no extra cost.)
# MAGIC
# MAGIC ### The components
# MAGIC ```
# MAGIC ┌─────────────── Unity Catalog ────────────────┐
# MAGIC │  Delta table (source)  ──sync──►  INDEX  ─────┼──► hosted on a VECTOR SEARCH ENDPOINT (compute)
# MAGIC │  chunk_id | text | category        vectors +  │         ▲
# MAGIC │                                     columns   │         │ similarity_search(query, filters, k)
# MAGIC └───────────────────────────────────────────────┘     your app / agent
# MAGIC         embedding model endpoint (e.g. databricks-gte-large-en) turns text → vectors
# MAGIC ```
# MAGIC
# MAGIC | Component | What it is |
# MAGIC |---|---|
# MAGIC | **Embedding model** | Turns text into a vector (a list of numbers). Same model for documents and queries |
# MAGIC | **Source Delta table** | Your chunks; must have a **primary key** and **Change Data Feed** for Delta Sync |
# MAGIC | **Index** | Vectors + selected columns, organised for fast search. Governed in **Unity Catalog** |
# MAGIC | **Endpoint** | The compute that hosts indexes and serves queries (`STANDARD` or `STORAGE_OPTIMIZED`) |
# MAGIC | **Sync pipeline** | Keeps the index up to date with the table (`TRIGGERED` or `CONTINUOUS`) |
# MAGIC | **Query** | `similarity_search` with text or a vector, `num_results`, `filters`, optional hybrid / reranking |
# MAGIC
# MAGIC ### The three index types
# MAGIC | Type | You provide | Databricks does | Use when |
# MAGIC |---|---|---|---|
# MAGIC | **Delta Sync – managed embeddings** | a text column + embedding endpoint | embeds rows *and* queries, syncs | Easiest; most RAG apps |
# MAGIC | **Delta Sync – self-managed embeddings** | a vector column you computed | syncs the vectors | You use your own/custom embedding model |
# MAGIC | **Direct Vector Access** | vectors via API (upsert/delete) | stores & searches; **no** table sync | Real-time writes from an app, no Delta source |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 documents with metadata → Step 2 embeddings (vectors) → Step 3 similarity = dot product
# MAGIC Step 4 exact (brute-force) search vs ANN idea → Step 5 filters → Step 6 hybrid (vector + keyword)
# MAGIC Step 7 sync: what happens when the table changes → Step 8 map each piece to the real API
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Documents with metadata
# MAGIC **What:** 8 chunks. `category` is **metadata** – extra columns we can filter on and return with results.

# COMMAND ----------

docs = [
    ("ret-1", "returns", "Most products can be returned within 30 days of delivery."),
    ("ret-2", "returns", "Opened software and gift cards cannot be returned."),
    ("shp-1", "shipping", "Standard shipping takes 3 to 5 business days; express takes 1 to 2."),
    ("shp-2", "shipping", "Orders delayed more than 3 days qualify for an investigation and a $10 store credit."),
    ("war-1", "warranty", "Defective products are replaced free within 15 days of delivery."),
    ("war-2", "warranty", "Accidental damage is covered only with the ShopEase Care protection plan."),
    ("pay-1", "payments", "Duplicate charges are refunded within 3 business days once confirmed."),
    ("pay-2", "payments", "We accept credit cards, debit cards and PayPal."),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Embeddings: text → vectors
# MAGIC **What:** call the embedding endpoint once for all chunks. Each vector has **1024** numbers.
# MAGIC **Why normalise:** after dividing each vector by its length, "how similar" becomes a simple dot product.
# MAGIC **How to read the output:** shape `(8, 1024)` and the first 5 numbers of one vector – meaningless to humans,
# MAGIC but similar texts get similar numbers.

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(texts):
    r = client.predict(endpoint="databricks-gte-large-en", inputs={"input": texts})
    v = np.array([d["embedding"] for d in r["data"]])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

doc_vectors = embed([d[2] for d in docs])
print("shape:", doc_vectors.shape)
print("first 5 numbers of 'ret-1':", np.round(doc_vectors[0][:5], 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Similarity: comparing a question with every chunk
# MAGIC **What:** cosine similarity between the question vector and each chunk vector. Scores near 1 = very similar.
# MAGIC **How to read the output:** the shipping-delay chunk (`shp-2`) should score highest for "my parcel is really late".
# MAGIC Notice it wins even though the words "parcel" and "late" don't both appear in it – that's **semantic** search.

# COMMAND ----------

question = "My parcel is really late, do I get compensation?"
q = embed([question])[0]
scores = doc_vectors @ q
for (doc_id, cat, text), s in sorted(zip(docs, scores), key=lambda x: -x[1]):
    print(f"{s:.3f}  {doc_id:6s} {text[:70]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Exact search vs. Approximate Nearest Neighbour (ANN)
# MAGIC **What we just did is exact (brute-force) search:** compare with *every* vector. Fine for 8 rows; with 100 million
# MAGIC it's far too slow.
# MAGIC
# MAGIC **What Vector Search does instead – ANN (e.g. the HNSW algorithm):** it builds a "map" of neighbours when indexing,
# MAGIC then hops through the map to the closest region – checking a tiny fraction of the vectors. The result is
# MAGIC *almost* always the same as exact search, in milliseconds.
# MAGIC
# MAGIC **How to read the output:** the time per query grows with the number of vectors for brute force (we simulate
# MAGIC bigger collections with random vectors). Roughly: 200× more vectors → ~200× more time. Imagine 100 million –
# MAGIC this growth is why an ANN index is needed.

# COMMAND ----------

import time

rng = np.random.default_rng(0)
for n in [1_000, 50_000, 200_000]:                       # kept small enough for notebook memory (~0.8 GB max)
    fake = rng.random((n, 1024), dtype=np.float32)
    start = time.time()
    _ = np.argsort(-(fake @ q.astype("float32")))[:3]
    print(f"brute force over {n:>9,} vectors: {time.time() - start:.3f} s per query")
    del fake

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Metadata filters
# MAGIC **What:** keep only rows that match a condition, *then* rank by similarity. In the real API:
# MAGIC `filters={"category": "warranty"}`.
# MAGIC **How to read the output:** only warranty chunks are ranked.

# COMMAND ----------

q2 = embed(["I dropped my phone, is that covered?"])[0]
filtered = [(d, float(v @ q2)) for d, v in zip(docs, doc_vectors) if d[1] == "warranty"]
for (doc_id, cat, text), s in sorted(filtered, key=lambda x: -x[1]):
    print(f"{s:.3f}  {doc_id}  {text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Hybrid search: vector + keyword
# MAGIC **What:** blend the vector score with a simple keyword-match score. In the real API: `query_type="hybrid"`.
# MAGIC **Why:** an exact word like "PayPal" or a product code may not dominate the vector; keywords rescue it.
# MAGIC **How to read the output:** with hybrid scoring, `pay-2` (contains "PayPal") is first.

# COMMAND ----------

query = "PayPal"
qv = embed([query])[0]
rows = []
for (doc_id, cat, text), v in zip(docs, doc_vectors):
    keyword = 1.0 if query.lower() in text.lower() else 0.0
    rows.append((doc_id, round(float(v @ qv), 3), keyword, round(0.5 * float(v @ qv) + 0.5 * keyword, 3)))
display(spark.createDataFrame(rows, "chunk_id string, vector_score double, keyword_score double, hybrid_score double")
        .orderBy("hybrid_score", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Sync: what happens when the table changes
# MAGIC | Change in the Delta table | What the sync does |
# MAGIC |---|---|
# MAGIC | new row | embeds it and adds the vector |
# MAGIC | updated `chunk_text` | re-embeds that row (found by **primary key**) |
# MAGIC | deleted row | removes its vector |
# MAGIC
# MAGIC **Change Data Feed** tells the pipeline *which* rows changed, so it doesn't re-embed everything.
# MAGIC **TRIGGERED** = runs when you call `index.sync()` (or on a schedule/job); **CONTINUOUS** = runs all the time,
# MAGIC changes appear within seconds, higher cost.
# MAGIC
# MAGIC ### Step 8 – From this demo to the real API
# MAGIC | Demo here | Real Vector Search |
# MAGIC |---|---|
# MAGIC | `embed(...)` on docs | managed embeddings: `embedding_source_column` + `embedding_model_endpoint_name` |
# MAGIC | `doc_vectors` array | the **index** (`vsc.create_delta_sync_index(...)`) |
# MAGIC | brute-force `@` | ANN search on the **endpoint** |
# MAGIC | Python `if d[1] == ...` | `filters={"category": "warranty"}` |
# MAGIC | blended score | `query_type="hybrid"` |
# MAGIC | re-running embed after changes | `index.sync()` / CONTINUOUS pipeline |
# MAGIC
# MAGIC See **4.6** for the real thing and **4.10** for choosing settings.
# MAGIC
# MAGIC **Exam takeaways:** know the components (**embedding model, source Delta table with PK + CDF, index, endpoint,
# MAGIC sync pipeline**), the **three index types**, **TRIGGERED vs CONTINUOUS** sync, ANN search, **filters**, **hybrid**
# MAGIC search, and that indexes are **Unity Catalog** objects with UC permissions.
