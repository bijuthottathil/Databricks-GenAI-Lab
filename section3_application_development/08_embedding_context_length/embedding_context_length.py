# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.8 – Select an embedding model context length based on source documents, queries and optimisation strategy
# MAGIC
# MAGIC **The big idea for juniors:** an embedding model turns text into a list of numbers (a *vector*) for searching.
# MAGIC Every embedding model has a **context length** – the maximum number of *tokens* (≈ ¾ of a word) it can read at once.
# MAGIC **Anything beyond the limit is cut off (truncated) and silently ignored.** If the answer is in the ignored part, the
# MAGIC search can never find it.
# MAGIC
# MAGIC | Model on Databricks | Context length (max tokens per input) | Vector size |
# MAGIC |---|---|---|
# MAGIC | `databricks-bge-large-en` | 512 | 1024 |
# MAGIC | `databricks-gte-large-en` | 8192 | 1024 |
# MAGIC
# MAGIC *(Check each model's documentation/model card for current limits – we also measure the vector size below.)*
# MAGIC
# MAGIC **How to decide – look at three things:**
# MAGIC 1. **Source documents** – how long are they, and how long will your chunks be?
# MAGIC 2. **Expected queries** – short keyword searches or long, detailed questions?
# MAGIC 3. **Optimisation strategy** – many small precise chunks (fine with a short-context model), or fewer big chunks
# MAGIC    (needs a long-context model)? Also: speed, cost and storage.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample source documents of very different lengths
# MAGIC **What:** a short FAQ answer, a medium policy page, and a long warranty contract where the key fact
# MAGIC (accidental damage cover) appears near the **end**.
# MAGIC **Why put the fact at the end:** that's exactly what gets lost when a model truncates.

# COMMAND ----------

filler = ("This section describes general terms and conditions that apply to all ShopEase customers, including "
          "definitions, eligibility, responsibilities of both parties, and the process for raising a claim. ")
documents = {
    "faq_shipping": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 days.",
    "returns_policy": "Most products can be returned within 30 days of delivery. " * 8 +
                      "Opened software and gift cards cannot be returned.",
    "warranty_contract": filler * 30 +                                                   # ~800+ words of general text…
                         "Section 14: Accidental damage, such as dropping the device or spilling liquid on it, "
                         "is covered only if the customer purchased the ShopEase Care protection plan.",  # …key fact at the end
}

def approx_tokens(text):
    return int(len(text.split()) * 1.3)          # rough rule: 1 word ≈ 1.3 tokens

for name, text in documents.items():
    print(f"{name:18s} ~{approx_tokens(text):5d} tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Sample expected queries
# MAGIC **What:** a short keyword query and a long, detailed one.
# MAGIC **Why:** queries must fit too. Short queries fit any model; very long queries (e.g. a pasted email) may not.

# COMMAND ----------

queries = {
    "short": "accidental damage coverage",
    "long": ("Hi, last week I dropped my new laptop while getting off the bus and the screen is now cracked. "
             "I bought it from you two months ago. Is this kind of accidental damage covered by any plan, "
             "and if so what do I need to have purchased for it to be covered?"),
}
for name, q in queries.items():
    print(f"{name:6s} query ~{approx_tokens(q)} tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Embed the long contract with each model and see the effect of truncation
# MAGIC **What:** compare how similar the question is to (a) the **whole** long contract and (b) **only its last
# MAGIC paragraph**, for each model.
# MAGIC **Why:** if a model truncates at 512 tokens, it never "reads" Section 14, so the whole-document vector looks like
# MAGIC generic legal text and scores **low** for our question. A long-context model reads everything and scores higher.
# MAGIC
# MAGIC **How to read:** look at the `whole_contract` column. A big gap between models = truncation is hurting the
# MAGIC short-context model.

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
MODELS = ["databricks-bge-large-en", "databricks-gte-large-en"]

def embed(model, texts):
    response = client.predict(endpoint=model, inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

last_paragraph = documents["warranty_contract"].split("Section 14:")[1]
rows = []
for model in MODELS:
    try:
        v = embed(model, [queries["long"], documents["warranty_contract"], last_paragraph])
        rows.append((model, v.shape[1], float(v[0] @ v[1]), float(v[0] @ v[2]), "ok"))
    except Exception as e:                                   # some endpoints reject over-long input instead of truncating
        rows.append((model, None, None, None, f"error: {str(e)[:120]}"))

display(spark.createDataFrame(rows, "model string, vector_size int, similarity_whole_contract double, "
                                    "similarity_last_paragraph_only double, note string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Retrieval test: which document does each model pick?
# MAGIC **What:** for both queries, rank the 3 documents by similarity with each model.
# MAGIC **Why:** this is what users experience. The correct document for both queries is `warranty_contract`.

# COMMAND ----------

names = list(documents)
rows = []
for model in MODELS:
    doc_vectors = embed(model, [documents[n] for n in names])
    for qname, q in queries.items():
        scores = doc_vectors @ embed(model, [q])[0]
        best = names[int(np.argmax(scores))]
        rows.append((model, qname, best, best == "warranty_contract",
                     ", ".join(f"{n}={s:.2f}" for n, s in zip(names, scores))))
display(spark.createDataFrame(rows, "model string, query string, top_document string, correct boolean, all_scores string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Optimisation strategy: chunk to fit, or use a long-context model?
# MAGIC **What:** split the long contract into chunks that fit in 512 tokens, then retrieve with the **short-context**
# MAGIC model again.
# MAGIC **Why:** this is the classic fix – if every chunk fits, nothing is truncated, and the small Section 14 chunk is a
# MAGIC sharp match for the question. Often you get *better* precision than embedding one giant document.

# COMMAND ----------

words = documents["warranty_contract"].split()
chunk_words = 300                                   # 300 words ≈ 390 tokens, safely under 512
contract_chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
print(f"{len(contract_chunks)} chunks, largest ≈ {max(approx_tokens(c) for c in contract_chunks)} tokens")

v_chunks = embed("databricks-bge-large-en", contract_chunks)
scores = v_chunks @ embed("databricks-bge-large-en", [queries["long"]])[0]
best = int(np.argmax(scores))
print(f"Best chunk #{best} (score {scores[best]:.2f}) contains Section 14: {'Section 14' in contract_chunks[best]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Cost & storage: more chunks = more vectors
# MAGIC **What:** estimate how many vectors and how much storage each strategy needs for a larger library.
# MAGIC **Why:** a long-context model with big chunks stores fewer vectors (cheaper index, faster search) but each vector
# MAGIC "averages" more topics (less precise). Small chunks are precise but multiply storage and embedding calls.

# COMMAND ----------

library_tokens = 50_000_000            # pretend our whole document library is 50 million tokens
for label, chunk_tokens in [("small chunks for a 512-token model", 400), ("big chunks for an 8k-token model", 4000)]:
    n_vectors = library_tokens // chunk_tokens
    storage_mb = n_vectors * 1024 * 4 / 1e6         # 1024 numbers per vector × 4 bytes each
    print(f"{label:38s}: {n_vectors:>9,} vectors ≈ {storage_mb:,.0f} MB of vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision guide
# MAGIC | Situation | Choose |
# MAGIC |---|---|
# MAGIC | Documents are short, or you will chunk them small (≤ ~400 tokens) | A **512-token** model is enough (often faster/cheaper) |
# MAGIC | You need big chunks (whole sections, long tables, contracts) | A **long-context** model (e.g. 8k) |
# MAGIC | Queries are long (pasted emails, tickets) | Make sure queries fit too – or summarise/trim the query first |
# MAGIC | Priority is precision | Smaller chunks (+ parent–child retrieval for context) |
# MAGIC | Priority is low storage / fewer vectors | Bigger chunks with a long-context model |
# MAGIC | Unsure | Measure both with an eval set (see 3.3) |
# MAGIC
# MAGIC **Exam takeaways:** the chunk (and query) must fit in the embedding model's **context length** or the extra text is
# MAGIC **truncated**. Pick the model's context length from your document/chunk sizes and query lengths, then balance
# MAGIC precision (small chunks) against cost and storage (fewer, larger chunks).
