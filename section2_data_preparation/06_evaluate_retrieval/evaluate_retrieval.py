# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.6 – Use tools and metrics to evaluate retrieval performance
# MAGIC
# MAGIC **Plain English:** in a RAG app, the *retriever* is the part that finds chunks for a question. If it brings back
# MAGIC the wrong chunks, even the best AI model will answer badly. So we **test the retriever on its own**, like a
# MAGIC search engine: give it questions whose correct answers we already know, and score what it returns.
# MAGIC
# MAGIC | Metric | Plain-English meaning | Range |
# MAGIC |---|---|---|
# MAGIC | **Hit rate @k** | Was *at least one* correct chunk in the top k results? | 0–1, higher is better |
# MAGIC | **Recall @k** | Of *all* correct chunks, what share did we find in the top k? | 0–1 |
# MAGIC | **Precision @k** | Of the k chunks returned, what share were correct? | 0–1 |
# MAGIC | **MRR** (Mean Reciprocal Rank) | How high up was the first correct chunk? 1st place = 1, 2nd = ½, 3rd = ⅓ … | 0–1 |
# MAGIC | **nDCG @k** | Like MRR but rewards *all* correct chunks being near the top | 0–1 |
# MAGIC | **Context relevance (LLM judge)** | An AI model grades whether each returned chunk is useful – no answer key needed | 0–1 |
# MAGIC
# MAGIC **Tools on Databricks:** a labelled evaluation set (a Delta table), **MLflow** to log metrics and compare runs,
# MAGIC and **Mosaic AI Agent Evaluation** (`mlflow.genai.evaluate`) with built-in LLM judges for retrieval relevance.
# MAGIC **Analogy:** we are grading a librarian. We ask 10 questions we already know the answers to and check whether
# MAGIC they bring back the right page.
# MAGIC
# MAGIC ```
# MAGIC question ──► RETRIEVER finds chunks ──► AI MODEL writes the answer from those chunks
# MAGIC               (this notebook tests        (if the chunks are wrong, the answer
# MAGIC                ONLY this half)             will be wrong too)
# MAGIC ```
# MAGIC
# MAGIC | Step | Role |
# MAGIC |---|---|
# MAGIC | 1–2 | **What** we test with: documents + an answer key |
# MAGIC | 3 | **What** we compare: two design choices (chunking strategies) |
# MAGIC | 4 | **The thing** being tested: the retriever |
# MAGIC | 5–6 | **Exact grading** with the answer key, recorded in MLflow |
# MAGIC | 7 | **AI grading** without an answer key, as in production |
# MAGIC
# MAGIC **In one sentence:** test the retriever on its own, with questions whose answers you know, measure it
# MAGIC several ways, change one thing at a time, and record every run so the better design is proven by numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The knowledge base (a short handbook)
# MAGIC **What:** a short handbook with six `##` sections (Shipping, Shipping Delays, Returns, Refunds, Warranty, Payments).
# MAGIC
# MAGIC **Why:** this is the "library" we will search. Keeping it small means you can check every result by eye.

# COMMAND ----------

handbook = """## Shipping
Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs $15.
We ship to the USA, Canada and Mexico only.

## Shipping Delays
Orders delayed more than 3 days qualify for a shipping investigation. A confirmed delay earns a $10 store credit.

## Returns
Customers can return most products within 30 days of delivery for a full refund.
Opened software and gift cards cannot be returned.

## Refunds
Refunds go back to the original payment method within 5 to 7 business days.

## Warranty
Defective products qualify for a free replacement within 15 days of delivery.
After 15 days, the 1-year manufacturer warranty applies.

## Payments
We accept credit cards, debit cards and PayPal. Duplicate charges are refunded within 3 business days.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The evaluation set ("answer key")
# MAGIC **What:** 10 questions, each with a short **evidence phrase** that a chunk must contain to count as correct,
# MAGIC e.g. `("Can I pay with PayPal?", ["PayPal"])`.
# MAGIC
# MAGIC **Why:**
# MAGIC - You can't grade without knowing the right answers.
# MAGIC - The questions are worded *differently* from the handbook ("send something back" vs "return"), which tests
# MAGIC   whether search understands **meaning**, not just matching words.
# MAGIC - **Why a phrase and not "chunk #3"?** The two strategies cut the text differently, so chunk #3 means
# MAGIC   different things in each. "Does the chunk contain this phrase?" works for both, so the comparison is fair.

# COMMAND ----------

eval_set = [
    ("How long does express delivery take?",              ["1 to 2 business days"]),
    ("Do you deliver to Europe?",                          ["USA, Canada and Mexico"]),
    ("My package is a week late, what do I get?",          ["$10 store credit"]),
    ("How long do I have to send something back?",         ["within 30 days of delivery"]),
    ("Can I return a gift card?",                          ["gift cards cannot be returned"]),
    ("When will my refund show up?",                       ["5 to 7 business days"]),
    ("My headphones broke after a week, what happens?",    ["free replacement within 15 days"]),
    ("Is there a warranty after the first two weeks?",     ["1-year manufacturer warranty"]),
    ("I was charged twice, when do I get my money?",       ["refunded within 3 business days"]),
    ("Can I pay with PayPal?",                             ["PayPal"]),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Two chunking strategies to compare
# MAGIC **What:**
# MAGIC - **A: fixed 120-character chunks** – ignores sentences, so it can cut an answer in half.
# MAGIC - **B: one chunk per `##` section** – each chunk is one complete topic.
# MAGIC
# MAGIC **Why:** evaluation is about **comparing options**. We change *one* thing (the chunking) and keep everything else
# MAGIC the same (questions, embedding model, k), so any difference in scores is caused by the chunking.

# COMMAND ----------

def fixed_chunks(text, size=120):
    return [text[i:i + size] for i in range(0, len(text), size)]

def section_chunks(text):
    return [block.strip() for block in text.split("## ") if block.strip()]

strategies = {"A_fixed_120_chars": fixed_chunks(handbook), "B_by_section": section_chunks(handbook)}
for name, chunks in strategies.items():
    print(name, "->", len(chunks), "chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – A simple retriever: embeddings + cosine similarity
# MAGIC **What:**
# MAGIC - `embed()` sends text to the `databricks-gte-large-en` model, which returns a list of numbers (an
# MAGIC   **embedding**) capturing its meaning. Texts with similar meaning get similar numbers.
# MAGIC - `retrieve()` compares the question's numbers with every chunk's numbers (**cosine similarity**) and returns
# MAGIC   the **top k = 3** closest chunks.
# MAGIC
# MAGIC **Why:** this is the search engine we are grading. In production, **Databricks Vector Search** does the same job
# MAGIC at scale; we write it by hand here so you can see how it works.
# MAGIC
# MAGIC ⚠️ Steps 6 and 7 both use `embed()`, `retrieve()` and `question_vectors` from this cell – run it first.

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(texts):
    response = client.predict(endpoint="databricks-gte-large-en", inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

questions = [q for q, _ in eval_set]
question_vectors = embed(questions)

def retrieve(chunks, chunk_vectors, question_vector, k):
    scores = chunk_vectors @ question_vector
    top = np.argsort(-scores)[:k]                    # indices of the k most similar chunks
    return [chunks[i] for i in top]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – The scoring formulas, written out in plain Python
# MAGIC **What:** for each question we mark the 3 returned chunks as correct or wrong, e.g. `[wrong, correct, wrong]`, then:
# MAGIC
# MAGIC | Metric | Plain-English question | Example `[wrong, correct, wrong]` |
# MAGIC |---|---|---|
# MAGIC | **Hit rate** | Did we find the answer at all? | 1 (yes) |
# MAGIC | **Recall** | Of all the correct chunks that exist, how many did we find? | 1 of 1 = 1.0 |
# MAGIC | **Precision** | Of the 3 we returned, how many were correct? | 1 of 3 = 0.33 |
# MAGIC | **MRR** | How high up was the first correct one? | 2nd place = ½ = 0.5 |
# MAGIC | **nDCG** | Like MRR, but rewards *all* correct chunks for being near the top | ≈ 0.63 |
# MAGIC
# MAGIC **Why several metrics?** Each one catches a different problem:
# MAGIC - **Low hit rate / recall** → the right chunk is never found. Fix the chunking or the embedding model.
# MAGIC - **Low MRR / nDCG** → it's found, but too far down. Add re-ranking (notebook 08).
# MAGIC - **Low precision** → too much irrelevant text goes to the AI model. Use a smaller k, or re-rank.
# MAGIC
# MAGIC Note: here each question has exactly **one** correct chunk and we fetch 3, so the **best possible precision is 0.33**.

# COMMAND ----------

def is_relevant(chunk, evidence_phrases):
    return any(phrase.lower() in chunk.lower() for phrase in evidence_phrases)

def score_one(retrieved, all_chunks, evidence):
    relevant_flags = [is_relevant(c, evidence) for c in retrieved]          # e.g. [False, True, False]
    total_relevant = max(1, sum(is_relevant(c, evidence) for c in all_chunks))
    hit = 1.0 if any(relevant_flags) else 0.0
    recall = sum(relevant_flags) / total_relevant
    precision = sum(relevant_flags) / len(retrieved)
    first = next((i for i, flag in enumerate(relevant_flags) if flag), None)
    reciprocal_rank = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(flag / np.log2(i + 2) for i, flag in enumerate(relevant_flags))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(total_relevant, len(retrieved))))
    ndcg = dcg / ideal if ideal else 0.0
    return hit, recall, precision, reciprocal_rank, ndcg

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Run the evaluation and log it to MLflow
# MAGIC **What:** for each strategy: embed its chunks → run the 10 questions → average the scores → log them to
# MAGIC **MLflow** (experiment `fde_retrieval_evaluation`) together with the settings (chunking, k, model).
# MAGIC
# MAGIC **Why:**
# MAGIC - **Averaging over many questions** shows overall quality, not one lucky or unlucky question.
# MAGIC - **MLflow** keeps a permanent, side-by-side record. When you later try k = 5 or another embedding model, you
# MAGIC   compare against these runs on the **Experiments** page instead of relying on memory.
# MAGIC
# MAGIC **Expected result:** B wins – hit rate and recall 1.00 vs 0.70, MRR 0.95 vs 0.70. Fixed 120-character cuts split
# MAGIC 3 of the 10 answer phrases across two chunks, so no single chunk contained the answer.

# COMMAND ----------

import mlflow

# Store the runs in an experiment in your home folder (visible under Experiments in the left menu)
me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_retrieval_evaluation")

K = 3
results = []
for name, chunks in strategies.items():
    chunk_vectors = embed(chunks)
    per_question = []
    for (question, evidence), q_vec in zip(eval_set, question_vectors):
        retrieved = retrieve(chunks, chunk_vectors, q_vec, K)
        per_question.append(score_one(retrieved, chunks, evidence))
    hit, recall, precision, mrr, ndcg = np.mean(per_question, axis=0)
    results.append((name, len(chunks), float(hit), float(recall), float(precision), float(mrr), float(ndcg)))

    with mlflow.start_run(run_name=f"retrieval_eval_{name}"):
        mlflow.log_params({"chunking": name, "k": K, "embedding_model": "databricks-gte-large-en"})
        mlflow.log_metrics({f"hit_rate_at_{K}": hit, f"recall_at_{K}": recall, f"precision_at_{K}": precision,
                            "mrr": mrr, f"ndcg_at_{K}": ndcg})

display(spark.createDataFrame(results, f"strategy string, num_chunks int, hit_rate_at_{K} double, recall_at_{K} double, "
                                       f"precision_at_{K} double, mrr double, ndcg_at_{K} double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Metric without an answer key: LLM-as-judge "context relevance"
# MAGIC **What:** for the first 5 questions, retrieve 3 chunks with strategy B, then ask
# MAGIC `databricks-meta-llama-3-3-70b-instruct`: *"Does this text help answer the question? YES or NO."*
# MAGIC The score is the share of YES answers.
# MAGIC
# MAGIC **Why:** real users ask questions nobody has an answer key for. An AI judge can grade results **without** one –
# MAGIC that's how you monitor quality in production, and it is what Databricks' built-in judges in
# MAGIC `mlflow.genai.evaluate` do automatically. Here it is written out so you can see how it works.
# MAGIC
# MAGIC **Its limits:**
# MAGIC - It costs one model call per chunk (slower and pricier than Step 6).
# MAGIC - The judge itself can be wrong or inconsistent.
# MAGIC - It only grades what *was* retrieved – it **cannot notice that the right chunk was never found** (that is
# MAGIC   what recall in Step 6 measures).

# COMMAND ----------

def judge_relevance(question, chunk):
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={
            "messages": [
                {"role": "system", "content": "You grade search results. Reply with only YES if the text helps answer the question, otherwise NO."},
                {"role": "user", "content": f"Question: {question}\n\nText: {chunk}"},
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        },
    )
    return response["choices"][0]["message"]["content"].strip().upper().startswith("YES")

chunks = strategies["B_by_section"]
chunk_vectors = embed(chunks)
judged = []
for question, q_vec in list(zip(questions, question_vectors))[:5]:      # first 5 questions keep it quick
    retrieved = retrieve(chunks, chunk_vectors, q_vec, K)
    grades = [judge_relevance(question, c) for c in retrieved]
    judged.append((question, sum(grades) / K, " || ".join(c[:40] for c in retrieved)))

display(spark.createDataFrame(judged, "question string, llm_context_relevance double, top_chunks string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Do I need Steps 5–6, or is Step 7 enough?
# MAGIC **Short answer: do both. Step 7 alone is not a proper analysis.**
# MAGIC
# MAGIC 1. **Code dependency:** Step 7 uses `embed()`, `retrieve()` and `question_vectors` from Step 4, and the chunks
# MAGIC    from Step 3. The minimum to run Step 7 is Steps 1–4 + 7 (Steps 5–6 can be skipped *technically*).
# MAGIC 2. **They answer different questions:**
# MAGIC
# MAGIC | | Steps 5–6 (answer key) | Step 7 (LLM judge) |
# MAGIC |---|---|---|
# MAGIC | Needs labelled questions? | Yes | No |
# MAGIC | Exact & repeatable? | Yes – same input, same score | No – the judge can vary or be wrong |
# MAGIC | Cost | Free (just maths) | One model call per chunk |
# MAGIC | Detects "right chunk never found"? | **Yes** (recall) | **No** – it only sees what was returned |
# MAGIC | Fair comparison of designs (A vs B)? | **Yes** – this is how we proved B wins | Weak |
# MAGIC | Works on real user traffic? | No (no answer key) | **Yes** |
# MAGIC
# MAGIC **Use them together:**
# MAGIC - **Designing / comparing** (chunking, embedding model, k, re-ranking) → a small labelled test set + Steps 5–6.
# MAGIC - **Monitoring in production** (real questions, no answer key) → an LLM judge like Step 7 / `mlflow.genai.evaluate`.
# MAGIC
# MAGIC **Exam phrasing:** "has ground truth / labelled evaluation set" → recall, precision, MRR, nDCG.
# MAGIC "No ground truth / production monitoring" → LLM-as-judge (context relevance).

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Evaluate the **retriever separately** from the generator.
# MAGIC - With an answer key: **hit rate / recall / precision @k, MRR, nDCG**. Without one: **LLM judges** (context relevance).
# MAGIC - Change one thing at a time (chunking, k, embedding model, re-ranking) and **log every run in MLflow** to compare.
# MAGIC - Low recall → the right chunk isn't being found (try better chunking, a different embedding model, hybrid search).
# MAGIC   Low precision → too much noise in the top k (try re-ranking or a smaller k).
