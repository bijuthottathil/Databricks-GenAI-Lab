# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.3 – Select a chunking strategy based on model & retrieval evaluation
# MAGIC
# MAGIC **The big idea for juniors:** there is no "best chunk size" in general. The right choice depends on **your
# MAGIC documents**, **your embedding model** and **your questions**. So we don't guess – we run a small **experiment**:
# MAGIC try several chunking strategies with each embedding model, measure retrieval quality, and pick the winner.
# MAGIC
# MAGIC **Analogy:** choosing running shoes. You don't pick by the box description; you try a few pairs on the same track
# MAGIC and time yourself.
# MAGIC
# MAGIC **The experiment grid**
# MAGIC
# MAGIC | | `databricks-gte-large-en` | `databricks-bge-large-en` |
# MAGIC |---|---|---|
# MAGIC | Small chunks (150 chars, overlap 30) | ? | ? |
# MAGIC | Medium chunks (400 chars, overlap 50) | ? | ? |
# MAGIC | One chunk per section (structure-aware) | ? | ? |
# MAGIC
# MAGIC For each cell we measure **Recall@3** (was the answer in the top 3?) and **MRR** (how high was it?), log the run in
# MAGIC **MLflow**, then choose. This builds on Section 2.6 (retrieval metrics) – here the focus is the *decision*.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The source document (a product & policy handbook)
# MAGIC **Why this document:** it has clear sections *and* some long paragraphs, so different strategies behave differently.

# COMMAND ----------

handbook = """## Laptop Pro 15 – Warranty
The Laptop Pro 15 includes a 2-year limited warranty covering hardware defects such as screen, keyboard and motherboard failures. Accidental damage like drops or liquid spills is not covered unless the customer bought ShopEase Care.

## Laptop Pro 15 – Battery
The battery lasts up to 12 hours of web browsing. Battery capacity below 80 percent within the first year is treated as a defect and replaced free of charge.

## SoundMax Headphones – Warranty
SoundMax Headphones have a 6-month warranty. Defective headphones reported within 15 days of delivery are replaced immediately; after that, they are repaired by the manufacturer.

## Returns
Most products can be returned within 30 days of delivery for a full refund. Opened software, gift cards and personalised items cannot be returned. Laptops must be returned with all accessories and the original box.

## Shipping
Standard shipping takes 3 to 5 business days and is free on orders over $50. Express shipping takes 1 to 2 business days and costs $15. Orders delayed more than 3 days beyond the promised date qualify for an investigation and a $10 store credit.

## ShopEase Care
ShopEase Care is an optional protection plan that costs $99 per year. It covers accidental damage, including drops and liquid spills, for laptops and headphones.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The evaluation questions (with evidence phrases as the answer key)
# MAGIC **What:** questions a real customer might ask, each with a phrase that a correct chunk must contain.
# MAGIC **Why these questions:** some need a *specific product* (laptop vs headphones warranty), some combine ideas
# MAGIC (accidental damage + ShopEase Care) – that's where chunking choices matter.

# COMMAND ----------

eval_set = [
    ("How long is the warranty on the laptop?", "2-year limited warranty"),
    ("Is a cracked screen from dropping my laptop covered?", "Accidental damage like drops"),
    ("My laptop battery is at 75% after 8 months, is that a defect?", "below 80 percent within the first year"),
    ("How long is the headphones warranty?", "6-month warranty"),
    ("Can I return a gift card?", "gift cards"),
    ("What must I include when returning a laptop?", "all accessories and the original box"),
    ("How much does express delivery cost?", "costs $15"),
    ("What do I get if my order is very late?", "$10 store credit"),
    ("How much is the protection plan per year?", "$99 per year"),
    ("Does anything cover spilling coffee on my headphones?", "liquid spills, for laptops and headphones"),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – The three chunking strategies
# MAGIC **What:**
# MAGIC - `fixed_with_overlap(size, overlap)` – cut every `size` characters, repeating `overlap` characters.
# MAGIC - `by_section` – one chunk per `##` heading, with the heading kept at the top (so "Laptop" vs "Headphones" is clear).
# MAGIC **Why keep the heading:** a chunk saying "has a 6-month warranty" is useless if you can't tell which product.

# COMMAND ----------

def fixed_with_overlap(text, size, overlap):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]

def by_section(text):
    return ["## " + block.strip() for block in text.split("## ") if block.strip()]

strategies = {
    "small_150_overlap30": fixed_with_overlap(handbook, 150, 30),
    "medium_400_overlap50": fixed_with_overlap(handbook, 400, 50),
    "by_section": by_section(handbook),
}
for name, chunks in strategies.items():
    print(f"{name:22s} {len(chunks):3d} chunks, average {sum(map(len, chunks)) // len(chunks)} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Retrieval with each embedding model
# MAGIC **What:** `embed(model, texts)` turns texts into vectors using the chosen model. `evaluate(...)` retrieves the top 3
# MAGIC chunks per question and computes Recall@3 and MRR.
# MAGIC **Why two models:** a chunk size that works for one model may not be best for another (different training data
# MAGIC and different input limits – `bge-large-en` reads at most 512 tokens per chunk).

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(model, texts):
    response = client.predict(endpoint=model, inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)      # unit length -> dot product = cosine similarity

def evaluate(model, chunks, k=3):
    chunk_vectors = embed(model, chunks)
    question_vectors = embed(model, [q for q, _ in eval_set])
    recalls, reciprocal_ranks = [], []
    for (question, evidence), q_vec in zip(eval_set, question_vectors):
        top = np.argsort(-(chunk_vectors @ q_vec))[:k]                     # indexes of the k best chunks
        hits = [evidence.lower() in chunks[i].lower() for i in top]        # True where the chunk holds the answer
        recalls.append(1.0 if any(hits) else 0.0)
        reciprocal_ranks.append(1.0 / (hits.index(True) + 1) if any(hits) else 0.0)
    return float(np.mean(recalls)), float(np.mean(reciprocal_ranks))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Run the experiment grid and log every combination to MLflow
# MAGIC **What:** 2 models × 3 strategies = 6 runs. Each run records its settings (params) and scores (metrics).
# MAGIC **Why MLflow:** you can sort and compare the runs in the **Experiments** page, and re-run later with new
# MAGIC documents to see whether the winner changes.

# COMMAND ----------

import mlflow

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_chunking_selection")

MODELS = ["databricks-gte-large-en", "databricks-bge-large-en"]
results = []
for model in MODELS:
    for name, chunks in strategies.items():
        recall, mrr = evaluate(model, chunks)
        avg_chars = sum(map(len, chunks)) / len(chunks)
        results.append((model, name, len(chunks), round(avg_chars), recall, mrr))
        with mlflow.start_run(run_name=f"{model}__{name}"):
            mlflow.log_params({"embedding_model": model, "chunking": name, "num_chunks": len(chunks)})
            mlflow.log_metrics({"recall_at_3": recall, "mrr": mrr})

grid = spark.createDataFrame(results, "embedding_model string, chunking string, num_chunks int, avg_chars int, recall_at_3 double, mrr double")
display(grid.orderBy("recall_at_3", "mrr", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Make the decision with a clear rule
# MAGIC **Rule (write it down before looking at results!):** highest Recall@3 first (the answer must be found at all);
# MAGIC break ties with MRR (higher = right chunk nearer the top); then prefer **fewer chunks** (cheaper index, less to store).

# COMMAND ----------

best = sorted(results, key=lambda r: (-r[4], -r[5], r[2]))[0]
print(f"Selected: chunking = {best[1]}, embedding model = {best[0]} "
      f"(Recall@3 = {best[4]:.2f}, MRR = {best[5]:.2f}, {best[2]} chunks)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to read the results
# MAGIC - **Small chunks** often split an answer from its product name ("6-month warranty" without "Headphones"),
# MAGIC   so they lose on product-specific questions.
# MAGIC - **Medium fixed chunks** may mix two products in one chunk – found, but noisy for the LLM.
# MAGIC - **By section** usually wins on structured documents because each chunk = one complete topic with its title.
# MAGIC - If two options tie, the cheaper one (fewer chunks, smaller model) is the better engineering choice.
# MAGIC
# MAGIC **Also evaluate the final answers:** good retrieval should lead to good answers. In practice, follow this with an
# MAGIC answer-quality evaluation (`mlflow.genai.evaluate` with correctness/groundedness judges) for the top 1–2 options.
# MAGIC
# MAGIC **Exam takeaways:** choose chunking **empirically** – build an eval set, try strategies × embedding models, compare
# MAGIC Recall@k / MRR (and answer quality) in MLflow, and apply a pre-agreed decision rule.
