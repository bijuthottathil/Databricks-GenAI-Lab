# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.9 – Use Databricks custom scorers for evaluating agents and LLMs
# MAGIC
# MAGIC **The big idea for juniors:** built-in judges (Correctness, Safety, …) check general quality. Your business has its
# MAGIC **own** rules: "always mention the order number", "never promise a refund", "reply in under 60 words", "use the
# MAGIC refund tool before confirming a refund". A **custom scorer** is a small Python function that checks one such rule
# MAGIC for every row of an evaluation – and the same scorer can later run in **production monitoring**.
# MAGIC
# MAGIC ### The kinds of custom scorers
# MAGIC | Kind | Returns | Example in this notebook |
# MAGIC |---|---|---|
# MAGIC | Code check, yes/no | `bool` | `mentions_order_id` – deterministic, free, fast |
# MAGIC | Code check, number | `float` / `int` | `brevity_score` – 1.0 short … 0.0 too long |
# MAGIC | With explanation | `Feedback(value, rationale)` | `no_refund_promise` – says *why* it failed |
# MAGIC | Uses the answer key | reads `expectations` | `has_expected_facts` |
# MAGIC | Uses the trace | reads `trace` spans | `refund_tool_before_confirm` – checks the agent's steps |
# MAGIC | LLM-as-judge (custom rubric) | `Feedback` from an LLM call | `empathy_judge` |
# MAGIC | Natural-language rules | built-in `Guidelines(...)` | "must not mention competitors" |
# MAGIC | Declarative judge | `make_judge(...)` (newer MLflow) | `tone_judge` |
# MAGIC
# MAGIC ### How a scorer gets its data
# MAGIC ```python
# MAGIC @scorer
# MAGIC def my_check(inputs, outputs, expectations, trace):   # ask only for what you need
# MAGIC     ...
# MAGIC ```
# MAGIC MLflow looks at the argument names and passes the row's `inputs`, the app's `outputs`, the row's `expectations`
# MAGIC and the `trace` of that run.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 a traced support agent (lookup tool → maybe refund tool → reply)
# MAGIC Step 2 evaluation dataset (inputs + expectations)
# MAGIC Step 3 code scorers   Step 4 trace scorer   Step 5 LLM-judge scorer + Guidelines
# MAGIC Step 6 test scorers on one hand-made example (unit test for scorers!)
# MAGIC Step 7 mlflow.genai.evaluate with all scorers    Step 8 make_judge (optional)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – A traced agent to evaluate
# MAGIC **What:** for refund requests the agent should call `issue_refund` (TOOL span) **before** telling the customer the
# MAGIC refund is done. For "maybe refund" requests it should not promise anything. The prompt is intentionally a bit weak
# MAGIC so some scorers find problems.

# COMMAND ----------

import re
import mlflow
from mlflow.entities import SpanType, Feedback
from mlflow.deployments import get_deploy_client

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_custom_scorers")
client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

ORDERS = {"1002": {"status": "delivered", "duplicate_charge": True}, "1003": {"status": "shipped", "duplicate_charge": False}}

def llm(system, user_msg, max_tokens=120):
    r = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        "temperature": 0.0, "max_tokens": max_tokens})
    return r["choices"][0]["message"]["content"].strip()

@mlflow.trace(span_type=SpanType.TOOL)
def lookup_order(order_id: str) -> dict:
    return ORDERS.get(order_id, {})

@mlflow.trace(span_type=SpanType.TOOL)
def issue_refund(order_id: str) -> str:
    return f"refund issued for duplicate charge on {order_id}"

@mlflow.trace(span_type=SpanType.AGENT)
def support_agent(question: str) -> str:
    m = re.search(r"\b(\d{4})\b", question)
    order_id = m.group(1) if m else ""
    order = lookup_order(order_id) if order_id else {}
    action = issue_refund(order_id) if order.get("duplicate_charge") and "charged twice" in question.lower() else "none"
    return llm("You are a friendly support agent for ShopCo. Reply in a few sentences.",
               f"Customer: {question}\nOrder data: {order}\nAction taken: {action}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Evaluation dataset
# MAGIC `expected_facts` are things the reply must contain; `refund_expected` says whether a refund tool call is correct.

# COMMAND ----------

eval_data = [
    {"inputs": {"question": "I was charged twice for order 1002, please fix it!"},
     "expectations": {"expected_facts": ["1002", "refund"], "refund_expected": True}},
    {"inputs": {"question": "Where is order 1003?"},
     "expectations": {"expected_facts": ["1003", "shipped"], "refund_expected": False}},
    {"inputs": {"question": "Order 1003 is late, will I maybe get money back?"},
     "expectations": {"expected_facts": ["1003"], "refund_expected": False}},
    {"inputs": {"question": "Is your shop cheaper than MegaMart? My order 1002 arrived."},
     "expectations": {"expected_facts": ["1002"], "refund_expected": False}},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Code scorers (deterministic, free)
# MAGIC **Code walk-through:**
# MAGIC - `mentions_order_id` → `bool`: finds the order number in the question and checks it's in the reply.
# MAGIC - `brevity_score` → `float`: 1.0 up to 60 words, falling linearly to 0.0 at 150 words.
# MAGIC - `no_refund_promise` → `Feedback`: fails if the reply says "refunded"/"refund has been issued" while **no** refund
# MAGIC   tool ran. The `rationale` explains the verdict in the MLflow UI.
# MAGIC - `has_expected_facts` → share of `expected_facts` found (case-insensitive).

# COMMAND ----------

from mlflow.genai.scorers import scorer, Guidelines

@scorer
def mentions_order_id(inputs, outputs) -> bool:
    m = re.search(r"\b(\d{4})\b", inputs["question"])
    return (m.group(1) in outputs) if m else True

@scorer
def brevity_score(outputs) -> float:
    words = len(outputs.split())
    return 1.0 if words <= 60 else max(0.0, 1 - (words - 60) / 90)

@scorer
def has_expected_facts(outputs, expectations) -> float:
    facts = expectations.get("expected_facts", [])
    return sum(f.lower() in outputs.lower() for f in facts) / len(facts) if facts else 1.0

PROMISE = r"\b(refunded|refund has been (issued|processed)|issued (you )?a refund|money (is|has been) (on its way|returned))\b"

@scorer
def no_refund_promise(outputs, trace) -> Feedback:
    refund_ran = any(s.name == "issue_refund" for s in trace.search_spans(span_type=SpanType.TOOL))
    promised = bool(re.search(PROMISE, outputs, re.IGNORECASE))
    if promised and not refund_ran:
        return Feedback(value=False, rationale="Reply promises a refund but issue_refund was never called.")
    return Feedback(value=True, rationale="Refund promise consistent with tool calls." if promised else "No refund promised.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Trace scorer: did the agent take the right action?
# MAGIC **What:** compare "was `issue_refund` called?" with `expectations["refund_expected"]`. This grades the agent's
# MAGIC **behaviour**, not its words – a reply can sound perfect while the agent did the wrong thing.

# COMMAND ----------

@scorer
def correct_refund_action(trace, expectations) -> Feedback:
    called = any(s.name == "issue_refund" for s in trace.search_spans(span_type=SpanType.TOOL))
    expected = expectations["refund_expected"]
    return Feedback(value=called == expected, rationale=f"issue_refund called={called}, expected={expected}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – LLM-as-judge custom scorer + Guidelines
# MAGIC **`empathy_judge`:** our own rubric sent to an LLM; returns `Feedback` with a 1–5 score and the reason. Use LLM
# MAGIC judges for things code can't check (tone, empathy, helpfulness).
# MAGIC **`Guidelines`:** the built-in way to write rules in plain English – no code. Each guideline is judged pass/fail.

# COMMAND ----------

@scorer
def empathy_judge(inputs, outputs) -> Feedback:
    text = llm("Rate the support reply's empathy from 1 (cold or dismissive) to 5 (warm, acknowledges the problem, "
               "reassuring). Reply exactly as: '<score> | <reason in max 15 words>'.",
               f"Customer: {inputs['question']}\nReply: {outputs}", max_tokens=40)
    m = re.match(r"\s*(\d)", text)
    return Feedback(value=int(m.group(1)) if m else 0, rationale=text)

no_competitors = Guidelines(name="no_competitors",
                            guidelines="The response must not mention or compare with competitor shops by name.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Unit-test your scorers on hand-made examples
# MAGIC **What:** scorers are code – test them before trusting them. Call a scorer directly with fake `outputs` (and a real
# MAGIC trace from one agent run).
# MAGIC **How to read the output:** a good reply passes, a bad one fails – if not, fix the scorer, not the agent.

# COMMAND ----------

print("mentions_order_id good:", mentions_order_id(inputs={"question": "order 1002?"}, outputs="Order 1002 shipped."))
print("mentions_order_id bad :", mentions_order_id(inputs={"question": "order 1002?"}, outputs="It shipped."))
print("brevity 30 words :", brevity_score(outputs="word " * 30))
print("brevity 120 words:", round(brevity_score(outputs="word " * 120), 2))

support_agent("Where is order 1003?")
t = mlflow.get_trace(mlflow.get_last_active_trace_id())
print("no_refund_promise (fake promise):", no_refund_promise(outputs="Your order was refunded.", trace=t).value)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Evaluate with all scorers
# MAGIC **How to read the output:** aggregate metrics such as `mentions_order_id/mean`, `correct_refund_action/mean`,
# MAGIC `empathy_judge/mean` (1–5) and `no_competitors/mean`. The MegaMart row is the likely `no_competitors` failure; the
# MAGIC "maybe money back" row tests `no_refund_promise`. Open the run's **Evaluations** tab to read each rationale.

# COMMAND ----------

all_scorers = [mentions_order_id, brevity_score, has_expected_facts, no_refund_promise,
               correct_refund_action, empathy_judge, no_competitors]
results = mlflow.genai.evaluate(data=eval_data, predict_fn=support_agent, scorers=all_scorers)
for name, value in sorted(results.metrics.items()):
    print(f"{name:35s} {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Declarative judges with `make_judge` (newer MLflow, optional)
# MAGIC **What:** `make_judge` builds an LLM judge from instructions with template variables like `{{ inputs }}` and
# MAGIC `{{ outputs }}` – no Python logic. If your MLflow version doesn't have it, the cell prints a message.
# MAGIC
# MAGIC ### Using custom scorers in production
# MAGIC The same scorer objects can be registered for **production monitoring** (Section 6.6):
# MAGIC `mentions_order_id.register(name="order_id_check").start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))`.
# MAGIC Scorers that need `expectations` can't run in production (no answer key) – use them offline.

# COMMAND ----------

try:
    from mlflow.genai.judges import make_judge
    tone_judge = make_judge(
        name="tone_judge",
        instructions="Decide whether the reply in {{ outputs }} to the customer message {{ inputs }} is professional and "
                     "free of sarcasm. Answer 'yes' or 'no'.")
    res = mlflow.genai.evaluate(data=eval_data[:2], predict_fn=support_agent, scorers=[tone_judge])
    print(res.metrics)
except Exception as e:
    print("make_judge not available in this MLflow version:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Write custom scorers with **`@scorer`**; request `inputs`, `outputs`, `expectations`, `trace` as needed; return
# MAGIC   `bool`, a number, a string or **`Feedback(value, rationale)`**.
# MAGIC - **Code scorers** for deterministic rules; **trace scorers** for agent behaviour (right tool, right order);
# MAGIC   **LLM-judge scorers / `Guidelines` / `make_judge`** for subjective qualities.
# MAGIC - Test scorers first; run them in `mlflow.genai.evaluate`; register expectation-free ones for production monitoring.
