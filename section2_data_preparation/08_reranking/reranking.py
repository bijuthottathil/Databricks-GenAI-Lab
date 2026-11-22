# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Section 2.8 – The role of re-ranking in information retrieval
# MAGIC
# MAGIC **Analogy – hiring:** a recruiter has 10,000 CVs. First they **skim** every CV for keywords in a few seconds each and
# MAGIC keep a shortlist of 6 (fast, rough). Then a hiring manager **reads each shortlisted CV carefully** against the job
# MAGIC description and puts them in order (slow, accurate). Nobody could read all 10,000 carefully – but reading 6 is easy.
# MAGIC Retrieval works the same way:
# MAGIC
# MAGIC ```
# MAGIC             STAGE 1 – the "skim" (retriever)                    STAGE 2 – the "careful read" (re-ranker)
# MAGIC question ──► compare numbers with ALL chunks ──► top 6 ──► read question + each candidate TOGETHER ──► best 1–3
# MAGIC              (embeddings, milliseconds)        candidates    and score how well it really answers       chunks
# MAGIC                                                                                                          │
# MAGIC                                                                     AI model writes the answer  ◄────────┘
# MAGIC ```
# MAGIC
# MAGIC **Stage 1 – retriever ("bi-encoder")**
# MAGIC - Every chunk was turned into an **embedding** (a list of numbers that captures meaning) *ahead of time*.
# MAGIC - At question time we embed only the question and compare numbers – very fast, even over millions of chunks.
# MAGIC - Weakness: the question and the chunk are encoded **separately**, so the model never sees them side by side.
# MAGIC   A chunk that merely *sounds* similar ("software… returning customers…") can beat the one that actually answers.
# MAGIC
# MAGIC **Stage 2 – re-ranker ("cross-encoder" or an LLM)**
# MAGIC - Reads the question **and** one candidate **together**, so it can judge "does this actually answer it?".
# MAGIC - Much more accurate, but much slower – so it only runs on the handful of candidates from Stage 1.
# MAGIC
# MAGIC | | Stage 1: retriever (bi-encoder) | Stage 2: re-ranker (cross-encoder / LLM) |
# MAGIC |---|---|---|
# MAGIC | Reads | question and chunk **separately** | question and chunk **together** |
# MAGIC | Speed | milliseconds over millions of chunks | slower – only run on ~5–50 candidates |
# MAGIC | Good at | **recall** – getting the right chunk *somewhere* in the shortlist | **precision** – putting the right chunk **first** |
# MAGIC
# MAGIC **Why it matters:** the AI model only reads the top few chunks. If the right one is 4th and we send only 3, the model
# MAGIC never sees it and may make something up. Re-ranking puts the right chunk first, so you can send **fewer** chunks
# MAGIC (cheaper, less confusing) and get **fewer made-up answers**.
# MAGIC
# MAGIC **What this notebook does**
# MAGIC
# MAGIC | Notebook step | What happens | Why |
# MAGIC |---|---|---|
# MAGIC | 1 | Build a tiny knowledge base full of "look-alike" chunks + an answer key | Create a situation where fast search can be fooled |
# MAGIC | 2 | Stage 1: fast embedding search, keep top 6 | See what plain vector search returns |
# MAGIC | 3 | Stage 2: LLM scores each of the 6, re-sort | See how re-ranking changes the order |
# MAGIC | 4 | Measure Hit@1 and MRR before vs after on all questions | Prove (with numbers) whether re-ranking helped |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 1 – Knowledge base with deliberately "look-alike" chunks
# MAGIC **What:** 12 short chunks from a shop's help centre, plus an **answer key** (`eval_set`) saying which chunk
# MAGIC (by its position in the list, starting at 0) truly answers each question.
# MAGIC
# MAGIC **Why the chunks are tricky on purpose:** several chunks *share words* with a question but don't answer it:
# MAGIC
# MAGIC | Question | Look-alike chunk (shares words, wrong) | Real answer |
# MAGIC |---|---|---|
# MAGIC | "Can I return software after opening the box?" | "Our **software** store… **returning** customers get 10% off." | "**Opened software** and gift cards **cannot be returned**." (index 1) |
# MAGIC | "My headphones broke after a week…" | "**Headphones** come in black, white and blue." | "Headphones that **stop working within 15 days are replaced**." (index 7) |
# MAGIC | "My parcel is a week late…" | "Delivery drivers may leave **parcels** with a neighbour…" | "A confirmed shipping **delay earns a $10 store credit**." (index 11) |
# MAGIC
# MAGIC These traps are exactly where fast vector search can slip and where a re-ranker earns its keep.
# MAGIC
# MAGIC **Why an answer key:** without knowing the right chunk, we couldn't *measure* whether re-ranking helped (Step 4).

# COMMAND ----------

chunks = [
    "Software purchases can be downloaded instantly after payment from your account page.",
    "Opened software and gift cards cannot be returned.",
    "Customers can return most products within 30 days of delivery for a full refund.",
    "Our software store offers thousands of titles, and returning customers get 10% off.",
    "Return shipping labels are free for wrong or defective items.",
    "Defective products qualify for a free replacement within 15 days of delivery.",
    "Headphones come in black, white and blue.",
    "Headphones that stop working within 15 days are replaced for free.",
    "Express shipping takes 1 to 2 business days and costs $15.",
    "Orders delayed more than 3 days qualify for a shipping investigation.",
    "Delivery drivers may leave parcels with a neighbour if nobody is home.",
    "A confirmed shipping delay earns a $10 store credit.",
]

# Small answer key: which chunk index really answers each question
eval_set = [
    ("Can I return software after opening the box?", 1),
    ("My headphones broke after a week. What happens now?", 7),
    ("My parcel is a week late. Do I get anything?", 11),
    ("What is the time limit for sending a laptop back?", 2),
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2 – Stage 1: fast embedding search (keep the top 6)
# MAGIC **What the code does:**
# MAGIC 1. `embed()` sends texts to the `databricks-gte-large-en` model and gets back one list of numbers per text.
# MAGIC    We scale each list to length 1 so that a simple multiplication (dot product) gives the **cosine similarity**
# MAGIC    (1.0 = same meaning, 0 = unrelated).
# MAGIC 2. `chunk_vectors = embed(chunks)` – embed all chunks **once, ahead of time**. This is what a Vector Search index
# MAGIC    stores, and why Stage 1 is fast: at question time only the question needs embedding.
# MAGIC 3. `first_stage(question, k=6)` – embed the question, score every chunk, keep the **6 highest**.
# MAGIC
# MAGIC **Why 6 and not just 1?** Stage 1 is rough. Fetching a few extra candidates makes it very likely that the right
# MAGIC chunk is *somewhere* in the shortlist (good **recall**). Re-ranking can only reorder what Stage 1 found – it can't
# MAGIC rescue a chunk that was never retrieved. Typical real systems fetch 20–50 and keep the best 3–5 after re-ranking.
# MAGIC
# MAGIC **How to read the output:** each line is `rank. (similarity score) chunk text`. Look for where the true answer
# MAGIC ("Opened software and gift cards cannot be returned.") appears – is it 1st, or did a look-alike beat it?

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(texts):
    response = client.predict(endpoint="databricks-gte-large-en", inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

chunk_vectors = embed(chunks)          # done once, ahead of time (this is what a vector index stores)

def first_stage(question, k=6):
    q = embed([question])[0]
    scores = chunk_vectors @ q
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]

question = eval_set[0][0]
candidates = first_stage(question)
print(f"Question: {question}\n")
for rank, (i, score) in enumerate(candidates, 1):
    print(f"{rank}. ({score:.3f}) {chunks[i]}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 3 – Stage 2: an LLM re-ranker
# MAGIC **What the code does:**
# MAGIC 1. `rerank_score(question, chunk_text)` sends the question **and** one candidate **together** to
# MAGIC    `databricks-meta-llama-3-3-70b-instruct`, with instructions to reply with a single 0–10 score
# MAGIC    ("10 = answers it completely, 0 = unrelated").
# MAGIC    - `temperature = 0.0` → the model gives the same score every time (no creativity wanted when grading).
# MAGIC    - `max_tokens = 5` → we only want a number, so we stop it from writing an essay.
# MAGIC    - We then keep only the digits from the reply (in case it answers "Score: 8") and turn them into a number.
# MAGIC 2. `rerank(question, candidates)` scores each of the 6 candidates and sorts them **highest score first**.
# MAGIC
# MAGIC **Why this works better than Stage 1:** the LLM sees both texts side by side, so it can tell the difference between
# MAGIC "mentions software" and "says opened software cannot be returned". This is the same idea as a dedicated
# MAGIC **cross-encoder** re-ranking model – we just use a chat model because one is already available in the workspace.
# MAGIC
# MAGIC **How to read the output:** the left column is the Stage 1 order, the right column is the new order with the LLM's
# MAGIC score in brackets. The true answer should move to (or stay in) the top position.
# MAGIC
# MAGIC **Cost to keep in mind:** this makes **one model call per candidate** (6 calls per question here). That is why
# MAGIC re-ranking only runs on a short list, never on the whole knowledge base.

# COMMAND ----------

def rerank_score(question, chunk_text):
    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={
            "messages": [
                {"role": "system", "content": "You are a search relevance grader. Score from 0 to 10 how directly the text "
                                              "answers the question (10 = answers it completely, 0 = unrelated). "
                                              "Reply with the number only."},
                {"role": "user", "content": f"Question: {question}\n\nText: {chunk_text}"},
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        },
    )
    reply = response["choices"][0]["message"]["content"]
    digits = "".join(ch for ch in reply if ch.isdigit() or ch == ".")
    return float(digits) if digits else 0.0

def rerank(question, candidates):
    scored = [(i, rerank_score(question, chunks[i])) for i, _ in candidates]
    return sorted(scored, key=lambda x: -x[1])          # highest re-rank score first

reranked = rerank(question, candidates)
print(f"Question: {question}\n")
print("BEFORE re-ranking (embedding order)        AFTER re-ranking (LLM order)")
for (i_before, _), (i_after, s_after) in zip(candidates, reranked):
    print(f"  {chunks[i_before][:42]:42s}   ->  [{s_after:4.1f}] {chunks[i_after][:50]}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 4 – Measure the effect on every question (Hit@1 and MRR)
# MAGIC One example can be luck, so we repeat Stage 1 + Stage 2 for **every** question in the answer key and compare
# MAGIC where the correct chunk ended up **before** and **after** re-ranking.
# MAGIC
# MAGIC **What the code does:**
# MAGIC - `position_of(correct, ranked)` → the correct chunk's place in a ranked list (1 = first). `None` means it was not
# MAGIC   in the shortlist at all – a Stage 1 **recall** problem that re-ranking cannot fix.
# MAGIC - **Hit@1** = share of questions where the correct chunk is in **1st place**. This matters most when you send only
# MAGIC   one chunk to the AI model.
# MAGIC - **MRR** (Mean Reciprocal Rank) = average of 1 / position: 1st place = 1.0, 2nd = 0.5, 3rd = 0.33, missing = 0.
# MAGIC   1.0 means "always first".
# MAGIC
# MAGIC **How to read the results table:** `position_before` vs `position_after` per question. Improvements look like
# MAGIC `3 → 1`. If a value is empty (`null`) in *both* columns, Stage 1 missed the chunk – fetch more candidates (larger k).
# MAGIC
# MAGIC **What if "after" isn't better?** With a small, clean knowledge base, good embeddings may already rank everything
# MAGIC correctly, so re-ranking has little to fix. Its value grows with **large, messy collections full of look-alike text**.
# MAGIC Always measure on *your* data – re-ranking adds cost and time to every question, so it should earn its place.

# COMMAND ----------

def position_of(correct_index, ranked):
    ids = [i for i, _ in ranked]
    return ids.index(correct_index) + 1 if correct_index in ids else None

rows = []
for question, correct in eval_set:
    candidates = first_stage(question)
    reranked = rerank(question, candidates)
    rows.append((question, position_of(correct, candidates), position_of(correct, reranked)))

def mrr(positions):
    return float(np.mean([1 / p if p else 0 for p in positions]))

def hit_at_1(positions):
    return float(np.mean([1 if p == 1 else 0 for p in positions]))

display(spark.createDataFrame(rows, "question string, position_before int, position_after int"))
before = [r[1] for r in rows]
after = [r[2] for r in rows]
print(f"MRR    before: {mrr(before):.2f}   after: {mrr(after):.2f}")
print(f"Hit@1  before: {hit_at_1(before):.2f}   after: {hit_at_1(after):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Re-ranking options on Databricks
# MAGIC | Option | How | Trade-off |
# MAGIC |---|---|---|
# MAGIC | **Vector Search built-in re-ranker** | Recent versions of `databricks-vectorsearch` can re-rank inside `similarity_search` (check your workspace's docs for availability) | Least code |
# MAGIC | **Cross-encoder model** | An open-source re-ranker (e.g. an `ms-marco` cross-encoder) served on Model Serving | Fast and cheap per candidate; purpose-built |
# MAGIC | **LLM re-ranker** (this notebook) | A chat model scores each candidate | Easiest to try; slowest and priciest |
# MAGIC
# MAGIC **Don't confuse with hybrid search:** hybrid search (keyword + vector) improves **Stage 1 recall** – it finds more of
# MAGIC the right candidates. Re-ranking improves the **order** in Stage 2. They are often used together.
# MAGIC
# MAGIC **When to use re-ranking**
# MAGIC - ✅ Large or messy collections with many similar-looking chunks.
# MAGIC - ✅ You can only send a few chunks to the AI model (small context window, cost limits).
# MAGIC - ✅ Answer quality matters more than a little extra waiting time.
# MAGIC - ❌ Tiny, clean collections where Stage 1 is already right (measure first!).
# MAGIC - ❌ Very strict speed requirements, where every extra model call hurts.
# MAGIC
# MAGIC **Settings you can tune:** how many candidates Stage 1 fetches (recall vs cost), and how many chunks you keep after
# MAGIC re-ranking (context size vs completeness).
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - Re-ranking is a **second stage** after retrieval: retrieve many (**recall**), re-rank to keep the best few (**precision**).
# MAGIC - Re-rankers read **question + document together** (cross-encoder), so they are more accurate but too slow to run on everything.
# MAGIC - Benefits: better top results, fewer chunks in the prompt, lower cost and fewer hallucinations.
# MAGIC - Cost: extra latency (and model calls) per question.
# MAGIC - Re-ranking **cannot recover** a chunk that Stage 1 never retrieved.