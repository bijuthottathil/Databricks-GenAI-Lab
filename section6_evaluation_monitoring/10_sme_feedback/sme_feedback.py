# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.10 – Incorporate SME feedback to improve agent performance
# MAGIC
# MAGIC **The big idea for juniors:** engineers know the code; **subject-matter experts (SMEs)** – here the returns-team
# MAGIC lead – know what a *correct* answer is. Their feedback is the most valuable signal you have. The goal is to turn it
# MAGIC into **concrete, repeatable improvements**, not a one-off chat:
# MAGIC
# MAGIC ```
# MAGIC agent v1 answers ─► SME reviews traces ─► feedback + corrections stored ON the traces
# MAGIC        ▲                                         │
# MAGIC        │                ┌────────────────────────┼─────────────────────────┐
# MAGIC        │                ▼                        ▼                         ▼
# MAGIC        │      evaluation dataset        new guidelines (rules)     missing knowledge
# MAGIC        │      (expected facts)          → Guidelines scorer         → prompt / documents
# MAGIC        │                └──────────────► agent v2 ◄───────────────────────┘
# MAGIC        └──────────── re-evaluate v1 vs v2 with the SAME dataset + scorers → deploy if better
# MAGIC ```
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **SME** | Subject-matter expert – knows the business rules |
# MAGIC | **Feedback** (assessment) | A judgement on a trace: 👍/👎, score, comment – `mlflow.log_feedback` |
# MAGIC | **Expectation** (assessment) | The correct answer / facts, from the SME – `mlflow.log_expectation` |
# MAGIC | **Assessment source** | Who gave it: `HUMAN` (SME, end user) or `LLM_JUDGE` |
# MAGIC | **Review App / labeling session** | Databricks UI where SMEs review traces and label them – no notebook needed |
# MAGIC | **Evaluation dataset** | Curated questions + expectations, reused for every new version |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 knowledge + agent v1 (incomplete prompt)        Step 2 run v1 on 6 questions → traces
# MAGIC Step 3 SME review (inline data) → log_feedback + log_expectation on each trace
# MAGIC Step 4 how SMEs do this in the Review App (shown)
# MAGIC Step 5 turn feedback into: eval dataset + guidelines + knowledge fixes
# MAGIC Step 6 agent v2                                        Step 7 evaluate v1 vs v2 → compare
# MAGIC Step 8 (optional) save the dataset as a managed MLflow evaluation dataset in Unity Catalog
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Agent v1
# MAGIC **What:** v1's prompt contains only **part** of the policy – it doesn't know the damaged-item rule (photo within 48
# MAGIC hours), that sale items are final, or that refunds take 5-7 business days. That's realistic: engineers wrote the
# MAGIC prompt from an old wiki page.

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.entities import SpanType, AssessmentSource, AssessmentSourceType
from mlflow.deployments import get_deploy_client

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_sme_feedback")
client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

PROMPT_V1 = """You are ShopCo's support assistant. Answer in max 3 sentences.
Policy: items can be returned within 30 days of delivery."""

def make_agent(system_prompt, version):
    @mlflow.trace(name=f"support_agent_{version}", span_type=SpanType.AGENT)
    def agent(question: str) -> str:
        r = client.predict(endpoint=ENDPOINT, inputs={
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            "temperature": 0.0, "max_tokens": 150})
        return r["choices"][0]["message"]["content"].strip()
    return agent

agent_v1 = make_agent(PROMPT_V1, "v1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Run v1 and keep the trace IDs
# MAGIC **What:** each answer creates a trace; `mlflow.get_last_active_trace_id()` gives its ID so the SME's feedback can be
# MAGIC attached to exactly that interaction.

# COMMAND ----------

questions = [
    "Can I return shoes I bought 2 weeks ago?",
    "My blender arrived broken, what should I do?",
    "I bought a jacket on sale, can I return it?",
    "When will I get my refund?",
    "Can I return a lamp after 40 days?",
    "The TV screen was cracked when I opened the box yesterday. Refund?",
]
v1_runs = []
for q in questions:
    answer = agent_v1(q)
    v1_runs.append({"question": q, "answer": answer, "trace_id": mlflow.get_last_active_trace_id()})
display(pd.DataFrame(v1_runs))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – The SME's review, stored on the traces
# MAGIC **What:** the returns-team lead reviewed each answer. Their verdicts are written inline here (in real life they come
# MAGIC from the Review App, Step 4). For every trace we log:
# MAGIC - **feedback** `sme_rating` (`True` good / `False` bad) with the SME's comment as `rationale`,
# MAGIC - **expectation** `expected_facts` – what a correct answer must contain.
# MAGIC
# MAGIC `AssessmentSource(HUMAN, "returns_team_lead")` records **who** said it, so you can tell SME labels apart from LLM
# MAGIC judges and end-user thumbs.
# MAGIC **How to read the output:** open the experiment → **Traces** → any v1 trace → the **Assessments** panel shows the
# MAGIC SME's rating, comment and expected facts.

# COMMAND ----------

sme_review = {
    # question index: (good?, comment, expected facts)
    0: (True,  "Correct.",                                                   ["30 days"]),
    1: (False, "Damaged items need a photo within 48 hours, then we send a free replacement or refund.",
               ["photo", "48 hours", "replacement or refund"]),
    2: (False, "Sale items are final sale – they cannot be returned, only exchanged for defects.",
               ["final sale", "cannot be returned"]),
    3: (False, "Must say 5-7 business days to the original payment method.", ["5-7 business days", "original payment method"]),
    4: (True,  "Correct – outside the 30 days. Should offer to escalate to a human though.", ["30 days", "escalate"]),
    5: (False, "Same damaged-item rule: photo within 48 hours.",              ["photo", "48 hours"]),
}
SME = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="returns_team_lead")

for i, run in enumerate(v1_runs):
    good, comment, facts = sme_review[i]
    mlflow.log_feedback(trace_id=run["trace_id"], name="sme_rating", value=good, source=SME, rationale=comment)
    mlflow.log_expectation(trace_id=run["trace_id"], name="expected_facts", value=facts, source=SME)

print(f"SME approved {sum(v[0] for v in sme_review.values())}/{len(sme_review)} v1 answers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – How SMEs give feedback without notebooks (shown, not run)
# MAGIC | Way | When |
# MAGIC |---|---|
# MAGIC | **Review App** chat | SMEs try the deployed agent and give 👍/👎 + comments |
# MAGIC | **Labeling sessions** | you send specific traces (e.g. low-scoring production traces) to SMEs to label with a schema |
# MAGIC | End-user feedback in your app | the app calls `mlflow.log_feedback(trace_id, ...)` with source `HUMAN` |
# MAGIC ```python
# MAGIC from mlflow.genai import label_schemas
# MAGIC from mlflow.genai.labeling import create_labeling_session
# MAGIC session = create_labeling_session(name="returns_review_june",
# MAGIC                                   assigned_users=["returns.lead@shopco.example"],
# MAGIC                                   label_schemas=[label_schemas.EXPECTED_FACTS])
# MAGIC session.add_traces(mlflow.search_traces(filter_string="..."))   # e.g. traces judges scored low
# MAGIC print(session.url)                                              # share with the SME
# MAGIC ```
# MAGIC SME labels land on the traces as assessments – exactly like Step 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Turn the feedback into three improvements
# MAGIC 1. **Evaluation dataset** – every reviewed question + the SME's `expected_facts`. Every future version is tested on it.
# MAGIC 2. **Guidelines** – SME comments that are *rules*, not facts ("offer to escalate", "be specific about time frames"),
# MAGIC    become a `Guidelines` scorer.
# MAGIC 3. **Knowledge fixes** – missing policy facts go into the prompt (or into the RAG documents + index sync).
# MAGIC
# MAGIC **How to read the output:** 6 dataset rows; the negative comments become knowledge fixes in Step 6.

# COMMAND ----------

eval_dataset = [{"inputs": {"question": run["question"]}, "expectations": {"expected_facts": sme_review[i][2]}}
                for i, run in enumerate(v1_runs)]

from mlflow.genai.scorers import Correctness, Guidelines
sme_guidelines = Guidelines(name="sme_rules", guidelines=[
    "If the customer cannot be helped under the policy, the response must offer to escalate to a human agent.",
    "Time frames must be stated precisely (e.g. '5-7 business days'), never vaguely ('soon', 'a few days').",
])

negative = [(run["question"], sme_review[i][1]) for i, run in enumerate(v1_runs) if not sme_review[i][0]]
display(pd.DataFrame(negative, columns=["question", "sme_comment → knowledge fix"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Agent v2, built from the SME's input
# MAGIC **What:** the prompt now contains the missing facts and the SME's rules. Nothing else changed – same model, same
# MAGIC settings – so any score change comes from the feedback.

# COMMAND ----------

PROMPT_V2 = """You are ShopCo's support assistant. Answer in max 3 sentences.
Policy:
- Items can be returned within 30 days of delivery.
- Sale items are final sale: they cannot be returned, only exchanged if defective.
- Damaged or broken on arrival: the customer sends a photo within 48 hours of delivery; we then send a free
  replacement or refund.
- Refunds go to the original payment method within 5-7 business days.
Rules:
- State time frames precisely.
- If the policy doesn't allow what the customer wants, explain why and offer to escalate to a human agent."""
agent_v2 = make_agent(PROMPT_V2, "v2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Evaluate v1 and v2 on the SME dataset
# MAGIC **Scorers:** `Correctness()` (uses the SME's `expected_facts`), the SME `Guidelines`, and a simple code scorer
# MAGIC `fact_recall` (share of expected facts found word-for-word).
# MAGIC **How to read the output:** v2 should beat v1 clearly on correctness and fact recall (v1 ≈ 2/6 correct per the SME,
# MAGIC v2 close to 6/6). If v2 isn't better, the feedback wasn't applied well – look at the failing rows before deploying.

# COMMAND ----------

from mlflow.genai.scorers import scorer

@scorer
def fact_recall(outputs, expectations) -> float:
    facts = expectations["expected_facts"]
    return sum(f.lower() in outputs.lower() for f in facts) / len(facts)

scorers = [Correctness(), sme_guidelines, fact_recall]
comparison = {}
for name, agent in [("v1", agent_v1), ("v2", agent_v2)]:
    with mlflow.start_run(run_name=f"sme_eval_{name}"):
        res = mlflow.genai.evaluate(data=eval_dataset, predict_fn=agent, scorers=scorers)
    comparison[name] = res.metrics

display(pd.DataFrame(comparison).reset_index().rename(columns={"index": "metric"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – (Optional) Keep the SME dataset in Unity Catalog
# MAGIC **What:** a **managed evaluation dataset** is a UC table that MLflow knows about – versioned, shareable, and easy to
# MAGIC grow with new SME labels. If the API isn't available in your MLflow version, the cell prints a message; the Python
# MAGIC list above works the same for `mlflow.genai.evaluate`.

# COMMAND ----------

try:
    import mlflow.genai.datasets as eval_datasets
    spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.fde_eval_monitor")
    ds = eval_datasets.create_dataset(uc_table_name="workspace.fde_eval_monitor.sme_returns_eval")
    ds.merge_records(eval_dataset)
    print("Saved", len(eval_dataset), "records to workspace.fde_eval_monitor.sme_returns_eval")
except Exception as e:
    print("Managed evaluation datasets not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making it a habit
# MAGIC - **Weekly:** pull low-scoring production traces (monitoring, Section 6.6) and 👎 user feedback → labeling session
# MAGIC   for the SME → new rows in the eval dataset.
# MAGIC - **Every change:** run the eval dataset; deploy only if metrics don't regress (CI/CD gate, Section 4.12).
# MAGIC - **Align judges:** compare SME ratings with LLM-judge verdicts on the same traces; if they disagree often, refine
# MAGIC   the judge's guidelines until they match the SME.
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - Collect SME input as **assessments on traces** (`log_feedback`, `log_expectation`, Review App, labeling sessions).
# MAGIC - Convert it into an **evaluation dataset** (ground truth), **guidelines** (custom judges) and **knowledge / prompt
# MAGIC   fixes**.
# MAGIC - **Re-evaluate old vs new** on the same dataset and scorers to prove improvement before deploying.
