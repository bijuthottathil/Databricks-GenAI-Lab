# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.7 – Identify evaluation judges that require ground truth
# MAGIC
# MAGIC **The big idea for juniors:** some judges can grade an answer just by looking at the question, the answer and the
# MAGIC retrieved documents ("is this relevant?", "is this safe?", "is this supported by the docs?"). Others need to know
# MAGIC **the right answer** ("is this correct?") – that right answer is called **ground truth** (in MLflow:
# MAGIC `expectations`, e.g. `expected_facts` or `expected_response`).
# MAGIC
# MAGIC Why it matters: ground truth must be written by people (SMEs) – it's expensive and **doesn't exist for live
# MAGIC production traffic**. So in **production monitoring** you can only use judges that *don't* need it; ground-truth
# MAGIC judges run **offline** on a curated evaluation set.
# MAGIC
# MAGIC ### The two groups (MLflow 3 built-in judges)
# MAGIC | Judge / scorer | Needs ground truth? | Needs | Checks |
# MAGIC |---|---|---|---|
# MAGIC | `Correctness()` | **Yes** | `expected_facts` or `expected_response` | answer matches the known right answer |
# MAGIC | `RetrievalSufficiency()` | **Yes** | `expected_facts`/`expected_response` + retriever span | retrieved docs contain everything needed |
# MAGIC | `ExpectationsGuidelines()` | **Yes** (per-row) | `guidelines` in each row's expectations | row-specific rules |
# MAGIC | `RelevanceToQuery()` | No | inputs + outputs | answer addresses the question |
# MAGIC | `Safety()` | No | outputs | no harmful content |
# MAGIC | `Guidelines(...)` | No | your global rules | e.g. "be polite", "no competitor names" |
# MAGIC | `RetrievalGroundedness()` | No | retriever span in the trace | answer supported by retrieved docs |
# MAGIC | `RetrievalRelevance()` | No | retriever span in the trace | retrieved docs relevant to the question |
# MAGIC
# MAGIC Classic metrics follow the same logic: **exact match, F1, ROUGE, BLEU need a reference**; latency, token count,
# MAGIC toxicity, format validity don't.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 a small traced RAG app (retriever span + answer)
# MAGIC Step 2 dataset WITHOUT ground truth → judges that work anyway
# MAGIC Step 3 what happens if you add Correctness without ground truth
# MAGIC Step 4 dataset WITH ground truth → Correctness + RetrievalSufficiency
# MAGIC Step 5 a case only ground truth can catch
# MAGIC Step 6 quiz: which judges for which situation?
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – A traced RAG app
# MAGIC **What:** `retrieve()` is a `RETRIEVER` span returning documents in the shape MLflow expects
# MAGIC (`page_content` + `metadata`) – retrieval judges read this span. `rag_app()` answers from the docs.
# MAGIC The knowledge base has a deliberate **mistake**: the express-shipping doc says **3** days, but the truth (which only
# MAGIC our SME knows) is **2** days.

# COMMAND ----------

import mlflow
from mlflow.entities import SpanType
from mlflow.deployments import get_deploy_client

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_ground_truth_judges")
client = get_deploy_client("databricks")

DOCS = [
    {"id": "returns",  "text": "Items can be returned within 30 days of delivery for a full refund."},
    {"id": "refunds",  "text": "Refunds are paid to the original payment method within 5-7 business days."},
    {"id": "shipping", "text": "Express shipping takes 3 business days."},          # wrong on purpose (truth: 2 days)
]

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str):
    words = [w.strip("?.,").lower() for w in question.split() if len(w) > 3]
    hits = [d for d in DOCS if any(w[:5] in d["text"].lower() for w in words)]
    return [{"page_content": d["text"], "metadata": {"doc_uri": d["id"]}} for d in hits[:2]]

@mlflow.trace(span_type=SpanType.CHAIN)
def rag_app(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(d["page_content"] for d in docs)
    r = client.predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
        "messages": [{"role": "system", "content": "Answer in one sentence using only the context."},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        "temperature": 0.0, "max_tokens": 80})
    return r["choices"][0]["message"]["content"].strip()

print(rag_app("How long does express shipping take?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – No ground truth: judges that still work
# MAGIC **What:** the dataset has **only inputs** – like production traffic. We use the "No" group.
# MAGIC **How to read the output:** high scores everywhere – including for the express question! The answer "3 business
# MAGIC days" *is* relevant, safe and grounded in the retrieved doc. These judges **can't** know the doc itself is wrong.

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalGroundedness, RetrievalRelevance, Guidelines

questions_only = [{"inputs": {"question": q}} for q in
                  ["How long do I have to return shoes?", "When will my refund arrive?", "How long does express shipping take?"]]
no_gt_scorers = [RelevanceToQuery(), Safety(), RetrievalGroundedness(), RetrievalRelevance(),
                 Guidelines(name="one_sentence", guidelines="The response must be a single sentence.")]

res_no_gt = mlflow.genai.evaluate(data=questions_only, predict_fn=rag_app, scorers=no_gt_scorers)
for k, v in sorted(res_no_gt.metrics.items()):
    print(f"{k:40s} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Adding `Correctness()` without ground truth
# MAGIC **What:** the same question-only dataset with `Correctness()` added.
# MAGIC **How to read the output:** depending on the MLflow version you get an error, or `Correctness` has no value /
# MAGIC an error assessment for every row – it has nothing to compare against. That's the sign a judge **requires ground
# MAGIC truth**.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RetrievalSufficiency

try:
    res = mlflow.genai.evaluate(data=questions_only, predict_fn=rag_app, scorers=[Correctness()])
    print("metrics:", res.metrics, "\n(no correctness/mean value = the judge could not run without expectations)")
except Exception as e:
    print("Correctness needs expectations:", str(e)[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – With ground truth: `Correctness` and `RetrievalSufficiency`
# MAGIC **What:** an SME wrote `expected_facts` for each question. Now the "Yes" group can run.
# MAGIC **How to read the output:** returns and refunds pass; **express shipping fails both** – the answer says 3 days,
# MAGIC the truth is 2, and the retrieved docs don't contain the needed fact.

# COMMAND ----------

with_gt = [
    {"inputs": {"question": "How long do I have to return shoes?"},  "expectations": {"expected_facts": ["30 days"]}},
    {"inputs": {"question": "When will my refund arrive?"},          "expectations": {"expected_facts": ["5-7 business days"]}},
    {"inputs": {"question": "How long does express shipping take?"}, "expectations": {"expected_facts": ["2 business days"]}},
]
res_gt = mlflow.genai.evaluate(data=with_gt, predict_fn=rag_app,
                               scorers=[Correctness(), RetrievalSufficiency(), RelevanceToQuery()])
for k, v in sorted(res_gt.metrics.items()):
    print(f"{k:40s} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Per-row comparison
# MAGIC **How to read the output:** the express row: `relevance = yes`, `grounded = yes`, but `correctness = no`. Only the
# MAGIC ground-truth judge caught the **wrong source document**. Fix: correct the doc, re-sync the index, re-run Step 4.

# COMMAND ----------

traces = mlflow.search_traces(run_id=res_gt.run_id)
cols = [c for c in ["request", "response", "assessments"] if c in traces.columns]
display(traces[cols].astype(str))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Quiz: which judges can you use?
# MAGIC **What:** a small rule: if a situation has ground truth, all judges are possible; otherwise only the "No" group.
# MAGIC Retrieval judges also need a retriever span in the trace.

# COMMAND ----------

NEEDS_GT = {"Correctness", "RetrievalSufficiency", "ExpectationsGuidelines", "ExactMatch/ROUGE/BLEU"}
NEEDS_RETRIEVER = {"RetrievalGroundedness", "RetrievalRelevance", "RetrievalSufficiency"}
ALL = ["Correctness", "RetrievalSufficiency", "ExpectationsGuidelines", "ExactMatch/ROUGE/BLEU",
       "RelevanceToQuery", "Safety", "Guidelines", "RetrievalGroundedness", "RetrievalRelevance"]

situations = [
    ("Production monitoring of a live RAG chatbot",                 False, True),
    ("Offline eval set with SME answers, RAG app",                  True,  True),
    ("Live chat without retrieval (plain LLM)",                     False, False),
    ("Regression test of a summariser vs. reference summaries",     True,  False),
]
for name, has_gt, has_retriever in situations:
    usable = [j for j in ALL if (has_gt or j not in NEEDS_GT) and (has_retriever or j not in NEEDS_RETRIEVER)]
    print(f"{name}\n   → {', '.join(usable)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - **Require ground truth:** `Correctness` (expected facts/response), `RetrievalSufficiency`, per-row
# MAGIC   `ExpectationsGuidelines`, and reference metrics (exact match, ROUGE, BLEU, F1).
# MAGIC - **No ground truth needed:** `RelevanceToQuery`, `Safety`, `Guidelines`, `RetrievalGroundedness`,
# MAGIC   `RetrievalRelevance` – usable in **production monitoring**.
# MAGIC - Groundedness ≠ correctness: an answer can be faithful to a **wrong** document. Keep a ground-truth eval set
# MAGIC   (built with SMEs, Section 6.10) for offline checks.
