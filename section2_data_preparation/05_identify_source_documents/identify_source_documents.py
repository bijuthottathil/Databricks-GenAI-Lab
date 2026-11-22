# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.5 – Identify the source documents that give a RAG app the knowledge and quality it needs
# MAGIC
# MAGIC **Plain English:** a RAG app can only be as good as the documents you give it. Before building anything, decide
# MAGIC **which documents to include** and **which to leave out**. Ask, for every candidate document:
# MAGIC
# MAGIC | Question | Why it matters |
# MAGIC |---|---|
# MAGIC | **Relevant?** Does it help answer the questions users will ask? | Irrelevant docs add noise and wrong matches |
# MAGIC | **Authoritative?** Is it the official source (not a forum post or a draft)? | Unofficial text can be wrong |
# MAGIC | **Current?** Is it the latest version? | Old versions contradict new ones |
# MAGIC | **Unique?** Is it a duplicate of another doc? | Duplicates waste space and skew results |
# MAGIC | **Safe?** Does it contain personal data (PII) or secrets? | The bot could leak them |
# MAGIC | **Complete?** Together, do the docs cover *every* expected question? | Gaps = "I don't know" or made-up answers |
# MAGIC
# MAGIC **Our app:** a *Customer Policy Assistant* that answers shoppers' questions about shipping, returns, warranty and payments.

# COMMAND ----------

# MAGIC %md ### Step 1 – Write down the questions the app must answer (this defines the knowledge we need)

# COMMAND ----------

expected_questions = [
    "How long does standard shipping take?",
    "How many days do I have to return a product?",
    "Can I return opened software?",
    "What happens if my headphones are defective?",
    "Which payment methods do you accept?",
    "Do you ship internationally?",
    "Can I get a price match if an item goes on sale?",
]

# COMMAND ----------

# MAGIC %md ### Step 2 – The candidate documents someone collected from around the company

# COMMAND ----------

candidates = [
    # (doc_id, title, owner, version_date, is_official, text)
    ("D1", "Shipping Policy v3", "Logistics", "2026-09-01", True,
     "Standard shipping takes 3 to 5 business days. Express takes 1 to 2 days. We ship to the USA, Canada and Mexico."),
    ("D2", "Returns Policy v2 (current)", "Customer Care", "2026-06-15", True,
     "Customers can return most products within 30 days of delivery. Opened software and gift cards cannot be returned."),
    ("D3", "Returns Policy v1 (old)", "Customer Care", "2023-01-10", True,
     "Customers can return products within 14 days of delivery."),
    ("D4", "Warranty Policy", "Product", "2026-03-20", True,
     "Defective products qualify for a free replacement within 15 days. After that, the 1-year manufacturer warranty applies."),
    ("D5", "Payments FAQ", "Finance", "2026-08-05", True,
     "We accept credit cards, debit cards and PayPal. Duplicate charges are refunded within 3 business days."),
    ("D6", "Payments FAQ (copy)", "Finance", "2026-08-05", True,
     "We accept credit cards, debit cards and PayPal. Duplicate charges are refunded within 3 business days."),
    ("D7", "Community forum: returns tips", "Unknown", "2025-11-30", False,
     "I think you can return stuff for like 60 days if you call and complain enough."),
    ("D8", "Escalation log", "Customer Care", "2026-10-02", True,
     "Ticket 4411: customer john.smith@example.com, phone 555-123-4567, card ending 4242, demanded refund."),
    ("D9", "Office holiday schedule", "HR", "2026-01-05", True,
     "The office is closed on New Year's Day, Independence Day and Thanksgiving."),
]
cand_df = spark.createDataFrame(candidates, "doc_id string, title string, owner string, version_date string, is_official boolean, text string")
display(cand_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Automatic quality checks
# MAGIC Simple, explainable rules – each one flags a reason to **exclude** a document.

# COMMAND ----------

import re

def find_pii(text):
    """Look for e-mail addresses, phone numbers and card numbers."""
    found = []
    if re.search(r"[\w.]+@[\w.]+\.\w+", text): found.append("email")
    if re.search(r"\b\d{3}-\d{3}-\d{4}\b", text): found.append("phone")
    if re.search(r"card ending \d{4}", text, re.I): found.append("card")
    return found

seen_texts = {}
review = []
for doc_id, title, owner, version_date, is_official, text in candidates:
    reasons = []
    if not is_official:
        reasons.append("not an official source")
    if "(old)" in title.lower() or version_date < "2024-01-01":
        reasons.append("outdated version")
    if text in seen_texts:
        reasons.append(f"duplicate of {seen_texts[text]}")
    else:
        seen_texts[text] = doc_id
    pii = find_pii(text)
    if pii:
        reasons.append("contains PII: " + ", ".join(pii))
    review.append([doc_id, title, reasons])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Relevance: is each document about *any* of our expected questions?
# MAGIC We use an **embedding model** to compare meanings. For each document we record its best similarity to any
# MAGIC expected question. Low scores (e.g. the holiday schedule) mean "not relevant to this app".

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed(texts):
    """Turn a list of texts into normalised vectors (so a dot product = cosine similarity)."""
    response = client.predict(endpoint="databricks-gte-large-en", inputs={"input": texts})
    vectors = np.array([item["embedding"] for item in response["data"]])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

doc_vectors = embed([c[5] for c in candidates])
question_vectors = embed(expected_questions)
similarity = question_vectors @ doc_vectors.T           # rows = questions, columns = documents

RELEVANCE_THRESHOLD = 0.70                               # tune by looking at the scores below
for row, best_score in zip(review, similarity.max(axis=0)):
    row.append(round(float(best_score), 3))
    if best_score < RELEVANCE_THRESHOLD:
        row[2].append("not relevant to expected questions")

summary = spark.createDataFrame(
    [(d, t, s, "EXCLUDE" if r else "INCLUDE", "; ".join(r)) for d, t, r, s in review],
    "doc_id string, title string, best_relevance double, decision string, reasons string")
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Coverage: does the *included* set answer every expected question?
# MAGIC For each question we take the best-matching included document and ask an LLM a yes/no question:
# MAGIC "does this text actually answer it?". A "NO" is a **knowledge gap** – a document we still need to find or write.

# COMMAND ----------

included_ids = [r.doc_id for r in summary.filter("decision = 'INCLUDE'").collect()]
included = [c for c in candidates if c[0] in included_ids]
included_vectors = embed([c[5] for c in included])

coverage = []
for question, q_vec in zip(expected_questions, question_vectors):
    best = int(np.argmax(included_vectors @ q_vec))
    doc_id, title, text = included[best][0], included[best][1], included[best][5]

    response = client.predict(
        endpoint="databricks-meta-llama-3-3-70b-instruct",
        inputs={
            "messages": [
                {"role": "system", "content": "You check whether a document answers a question. Reply with only YES or NO."},
                {"role": "user", "content": f"Question: {question}\n\nDocument: {text}"},
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        },
    )
    verdict = response["choices"][0]["message"]["content"].strip().upper()
    coverage.append((question, doc_id, title, "covered" if verdict.startswith("YES") else "GAP – need a source"))

display(spark.createDataFrame(coverage, "question string, best_doc string, title string, status string"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What you should see**
# MAGIC - Excluded: D3 (outdated, contradicts D2), D6 (duplicate), D7 (unofficial forum), D8 (PII), D9 (irrelevant).
# MAGIC - Gap: the **price-match** question – no document covers it, so the team must add a price-match policy (or the
# MAGIC   bot must be told to say "the policy does not specify this").
# MAGIC
# MAGIC **Exam takeaways:** start from the **questions users will ask**; include documents that are relevant,
# MAGIC authoritative, current, unique and safe; test **coverage** to find gaps before launch.
