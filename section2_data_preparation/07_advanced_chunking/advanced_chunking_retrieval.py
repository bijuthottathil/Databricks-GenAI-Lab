# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.7 – Design retrieval systems using advanced chunking strategies
# MAGIC
# MAGIC **Plain English:** basic chunking (cut every N characters) has two classic problems:
# MAGIC 1. **Small chunks are precise but lack context** – "It lasts 2 years." Which product? What lasts?
# MAGIC 2. **Big chunks have context but are fuzzy to search** – one chunk about five topics matches everything a little.
# MAGIC
# MAGIC Advanced strategies get the best of both:
# MAGIC
# MAGIC | Strategy | Idea in one line | Fixes |
# MAGIC |---|---|---|
# MAGIC | **Contextual headers / metadata enrichment** | Prefix each chunk with its document title and section | Chunks that don't make sense alone |
# MAGIC | **Semantic chunking** | Cut where the *topic changes* (measured with embeddings), not at a fixed size | Chunks mixing topics |
# MAGIC | **Parent–child ("small-to-big")** | *Search* small child chunks, but *return* their bigger parent section | Precision **and** context |
# MAGIC | **Sentence window** | Search single sentences, return the sentence plus its neighbours | Same idea, finer grained |
# MAGIC | **Metadata filtering** | Store product/category columns and filter before searching | Wrong product / wrong document |
# MAGIC | **Hierarchical / summary index** | Search document summaries first, then chunks inside the best documents | Very large collections |

# COMMAND ----------

# MAGIC %md ### Setup – an embedding function (turns text into numbers that capture meaning)

# COMMAND ----------

import re
import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(texts):
    response = client.predict(endpoint="databricks-gte-large-en", inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # unit length -> dot product = cosine similarity

def top_matches(question, chunks, k=2):
    vectors = embed([question] + chunks)
    scores = vectors[1:] @ vectors[0]
    order = np.argsort(-scores)[:k]
    return [(round(float(scores[i]), 3), chunks[i]) for i in order]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Contextual headers (metadata enrichment)
# MAGIC Two product manuals have almost identical sentences. Without context the retriever cannot tell them apart.

# COMMAND ----------

manuals = {
    "Laptop Pro 15 – User Manual": {
        "Warranty": "The warranty lasts 2 years and covers hardware failures.",
        "Battery": "The battery lasts up to 12 hours on a full charge.",
    },
    "SoundMax Headphones – User Manual": {
        "Warranty": "The warranty lasts 6 months and covers hardware failures.",
        "Battery": "The battery lasts up to 30 hours on a full charge.",
    },
}

plain_chunks = [text for doc in manuals.values() for text in doc.values()]
header_chunks = [f"Document: {title}\nSection: {section}\n{text}"
                 for title, doc in manuals.items() for section, text in doc.items()]

question = "How long is the warranty on the headphones?"
print("WITHOUT headers:")
for score, chunk in top_matches(question, plain_chunks):
    print(f"  {score}  {chunk}")
print("\nWITH headers:")
for score, chunk in top_matches(question, header_chunks):
    print(f"  {score}  {chunk.replace(chr(10), ' | ')}")

# COMMAND ----------

# MAGIC %md
# MAGIC Without headers, the two warranty sentences differ only by "2 years" vs "6 months", so the laptop one can win.
# MAGIC With the document title in the chunk, the headphones chunk is clearly the best match.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Semantic chunking – cut where the topic changes
# MAGIC 1. Split the text into sentences.
# MAGIC 2. Embed every sentence.
# MAGIC 3. Compare each sentence with the next one. A **drop in similarity** means "new topic" → start a new chunk there.

# COMMAND ----------

article = ("Standard shipping takes 3 to 5 business days. Express shipping arrives in 1 to 2 days. "
           "Shipping is free on orders over $50. "
           "You can return most items within 30 days. Returned items must be unused. Refunds take 5 to 7 days. "
           "We accept credit cards and PayPal. Duplicate charges are refunded within 3 days.")

sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", article) if s.strip()]
sentence_vectors = embed(sentences)

# similarity between sentence i and sentence i+1
neighbour_similarity = [float(sentence_vectors[i] @ sentence_vectors[i + 1]) for i in range(len(sentences) - 1)]
for i, sim in enumerate(neighbour_similarity):
    print(f"{sim:.3f}  between: '{sentences[i][:35]}...'  and  '{sentences[i + 1][:35]}...'")

# COMMAND ----------

# Break wherever the similarity is clearly lower than usual (below the 30th percentile of all neighbour scores)
threshold = float(np.percentile(neighbour_similarity, 30))
semantic_chunks, current = [], [sentences[0]]
for i, sim in enumerate(neighbour_similarity):
    if sim < threshold:
        semantic_chunks.append(" ".join(current))
        current = []
    current.append(sentences[i + 1])
semantic_chunks.append(" ".join(current))

print(f"Threshold = {threshold:.3f}\n")
for i, chunk in enumerate(semantic_chunks):
    print(f"chunk {i}: {chunk}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parent–child ("small-to-big") retrieval
# MAGIC - **Parents** = whole sections (good context for the AI model to read).
# MAGIC - **Children** = single sentences (precise to search).
# MAGIC - We **search the children**, then **return the parent** of the best child.
# MAGIC
# MAGIC In a real system you store both in Delta tables: a `child_chunks` table (indexed in Vector Search) with a
# MAGIC `parent_id` column pointing to a `parent_chunks` table.

# COMMAND ----------

parents = {
    "P1": "Returns: You can return most items within 30 days. Returned items must be unused and in original packaging. "
          "Laptops require the original box. Refunds take 5 to 7 business days.",
    "P2": "Warranty: Defective items are replaced free within 15 days. After 15 days the 1-year manufacturer warranty applies. "
          "Send a photo or video of the defect.",
}

children = []                                   # (parent_id, sentence)
for parent_id, text in parents.items():
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        children.append((parent_id, sentence))

question = "Do I need to keep the laptop box?"
child_texts = [c[1] for c in children]
best_score, best_child = top_matches(question, child_texts, k=1)[0]
parent_id = next(pid for pid, text in children if text == best_child)

print(f"Best CHILD (what we searched): {best_child}   score={best_score}")
print(f"Returned PARENT {parent_id} (what the AI model reads):\n{parents[parent_id]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sentence-window retrieval
# MAGIC Same idea, but instead of a fixed parent we return the matching sentence **plus one sentence on each side**.

# COMMAND ----------

all_sentences = [s for _, s in children]
best_score, best_sentence = top_matches("How do I prove my item is broken?", all_sentences, k=1)[0]
i = all_sentences.index(best_sentence)
window = " ".join(all_sentences[max(0, i - 1): i + 2])
print(f"Matched sentence: {best_sentence}\nWindow returned : {window}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metadata filtering
# MAGIC If we store the product name as a column, we can **filter first** ("only headphone chunks") and then rank by
# MAGIC similarity. Vector Search supports this with `filters={"product": "SoundMax Headphones"}`.

# COMMAND ----------

rows = [(title.split(" –")[0], section, text) for title, doc in manuals.items() for section, text in doc.items()]
chunk_df = spark.createDataFrame(rows, "product string, section string, chunk_text string")
display(chunk_df)

filtered = [r.chunk_text for r in chunk_df.filter("product = 'SoundMax Headphones'").collect()]
print(top_matches("How long does the battery last?", filtered, k=1))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Chunks that don't stand alone → add **contextual headers / metadata**.
# MAGIC - Topics drift inside long text → **semantic chunking**.
# MAGIC - Need precise search *and* rich context → **parent–child** or **sentence window**.
# MAGIC - Many similar documents (products, regions, versions) → **metadata filtering**.
# MAGIC - Huge collections → **hierarchical** (summary first, then chunks).
# MAGIC - Always measure with the retrieval metrics from 2.6 before and after changing the strategy.
