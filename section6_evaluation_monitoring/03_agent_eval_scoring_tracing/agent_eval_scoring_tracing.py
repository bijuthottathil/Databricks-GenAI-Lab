# Databricks notebook source
# MAGIC %md
# MAGIC # Section 6.3 – Evaluate agent performance using MLflow scoring and tracing
# MAGIC
# MAGIC **The big idea for juniors:** an agent does several things for one question: decide which tool to use, call the
# MAGIC tool, then write an answer. If the final answer is wrong, *which step* failed? **Tracing** records every step
# MAGIC (inputs, outputs, timing) as a tree of **spans**. **Scoring** (`mlflow.genai.evaluate` + scorers) grades each run –
# MAGIC and scorers can read the **trace**, so we grade not just the answer but *how* the agent got there.
# MAGIC
# MAGIC ```
# MAGIC trace: support_agent  (AGENT span, 2.1 s)
# MAGIC  ├─ choose_tool     (LLM span)   → {"tool": "get_order_status", "argument": "1002"}
# MAGIC  ├─ get_order_status (TOOL span) → "shipped, arriving 2024-06-12"
# MAGIC  └─ write_answer    (LLM span)   → "Your order 1002 has shipped and ..."
# MAGIC ```
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Trace** | The full record of one request through the agent |
# MAGIC | **Span** | One step inside a trace, with a type: `AGENT`, `LLM`, `TOOL`, `RETRIEVER`, … |
# MAGIC | **`@mlflow.trace`** | Decorator that turns a function call into a span automatically |
# MAGIC | **Scorer** | A function that grades one row: built-in LLM judges or your own code |
# MAGIC | **`expectations`** | The answer key for a row (expected facts, expected tool) |
# MAGIC | **`mlflow.genai.evaluate`** | Runs the agent on a dataset, applies all scorers, logs everything to an MLflow run |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 tools + data      Step 2 traced agent (choose_tool → tool → write_answer)
# MAGIC Step 3 run once and read the trace span by span
# MAGIC Step 4 evaluation dataset (inputs + expectations)
# MAGIC Step 5 scorers: built-in judges + trace-based code scorers
# MAGIC Step 6 mlflow.genai.evaluate → aggregate metrics
# MAGIC Step 7 per-row results: find the failing step
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Tools and their data
# MAGIC Two tools: `get_order_status(order_id)` reads a small dict; `get_policy(topic)` returns a policy sentence.

# COMMAND ----------

import json
import mlflow
from mlflow.entities import SpanType
from mlflow.deployments import get_deploy_client

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fde_section6_agent_eval")

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

ORDERS = {"1002": "shipped, arriving 2024-06-12", "1003": "processing, ships in 2 days", "1004": "delivered on 2024-06-01"}
POLICIES = {"returns": "Items can be returned within 30 days of delivery for a full refund.",
            "refunds": "Refunds are paid to the original payment method within 5-7 business days.",
            "shipping": "Standard shipping takes 3-5 business days; express takes 2 business days."}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The agent, with a span for every step
# MAGIC **Code walk-through:**
# MAGIC - `@mlflow.trace(span_type=SpanType.TOOL)` on each tool → each call appears as a TOOL span with its argument and result.
# MAGIC - `choose_tool` (LLM span) asks the model for JSON `{"tool": ..., "argument": ...}`.
# MAGIC - `write_answer` (LLM span) writes the reply from the tool result.
# MAGIC - `support_agent` (AGENT span) is the parent – the whole request.

# COMMAND ----------

def llm(messages, max_tokens=150):
    r = client.predict(endpoint=ENDPOINT, inputs={"messages": messages, "temperature": 0.0, "max_tokens": max_tokens})
    return r["choices"][0]["message"]["content"].strip()

@mlflow.trace(span_type=SpanType.TOOL)
def get_order_status(order_id: str) -> str:
    return ORDERS.get(order_id.strip(), "order not found")

@mlflow.trace(span_type=SpanType.TOOL)
def get_policy(topic: str) -> str:
    return POLICIES.get(topic.strip().lower(), "no policy found")

TOOLS = {"get_order_status": get_order_status, "get_policy": get_policy}

@mlflow.trace(span_type=SpanType.LLM)
def choose_tool(question: str) -> dict:
    text = llm([{"role": "system", "content":
                 "Pick one tool for the customer question. Tools: get_order_status(order_id), "
                 "get_policy(topic: returns|refunds|shipping), none. "
                 'Return ONLY JSON {"tool": "<name or none>", "argument": "<string>"}.'},
                {"role": "user", "content": question}], max_tokens=40)
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {"tool": "none", "argument": ""}

@mlflow.trace(span_type=SpanType.LLM)
def write_answer(question: str, tool_result: str) -> str:
    return llm([{"role": "system", "content": "You are a support agent. Answer in max 2 sentences using only the tool "
                                              "result. If it doesn't contain the answer, say you will escalate."},
                {"role": "user", "content": f"Tool result: {tool_result}\n\nQuestion: {question}"}])

@mlflow.trace(span_type=SpanType.AGENT)
def support_agent(question: str) -> str:
    choice = choose_tool(question)
    tool = TOOLS.get(choice.get("tool"))
    tool_result = tool(str(choice.get("argument", ""))) if tool else "no tool used"
    return write_answer(question, tool_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Run once and read the trace
# MAGIC **What:** `mlflow.get_last_active_trace_id()` gives the ID of the trace just created; `mlflow.get_trace()` loads it.
# MAGIC **How to read the output:** 4 spans – AGENT, LLM (`choose_tool`), TOOL (`get_order_status`), LLM (`write_answer`) –
# MAGIC with their duration in ms. Usually the LLM spans take almost all the time. The notebook also shows the trace in
# MAGIC an interactive widget.

# COMMAND ----------

print(support_agent("Where is my order 1002?"))

trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
for span in trace.data.spans:
    ms = (span.end_time_ns - span.start_time_ns) / 1e6
    print(f"{str(span.span_type):10s} {span.name:18s} {ms:8.0f} ms   inputs={str(span.inputs)[:60]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Evaluation dataset
# MAGIC **What:** each row has `inputs` (passed to the agent as keyword arguments – so the key must be `question`) and
# MAGIC `expectations`:
# MAGIC - `expected_facts` – facts the answer must contain (used by the built-in `Correctness` judge)
# MAGIC - `expected_tool` – which tool should be called (used by our trace-based scorer)

# COMMAND ----------

eval_data = [
    {"inputs": {"question": "Where is my order 1002?"},
     "expectations": {"expected_facts": ["order 1002 has shipped", "arrives 2024-06-12"], "expected_tool": "get_order_status"}},
    {"inputs": {"question": "How long do I have to return a jacket?"},
     "expectations": {"expected_facts": ["30 days"], "expected_tool": "get_policy"}},
    {"inputs": {"question": "When will I get my refund money?"},
     "expectations": {"expected_facts": ["5-7 business days", "original payment method"], "expected_tool": "get_policy"}},
    {"inputs": {"question": "Has order 1004 arrived?"},
     "expectations": {"expected_facts": ["delivered on 2024-06-01"], "expected_tool": "get_order_status"}},
    {"inputs": {"question": "How fast is express shipping?"},
     "expectations": {"expected_facts": ["2 business days"], "expected_tool": "get_policy"}},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Scorers
# MAGIC | Scorer | Type | Reads | Question it answers |
# MAGIC |---|---|---|---|
# MAGIC | `Correctness()` | built-in LLM judge | outputs + `expected_facts` | Is the answer right? |
# MAGIC | `RelevanceToQuery()` | built-in LLM judge | inputs + outputs | Does it answer the question asked? |
# MAGIC | `Safety()` | built-in LLM judge | outputs | Is it free of harmful content? |
# MAGIC | `used_expected_tool` | custom code | **trace** TOOL spans + `expected_tool` | Did the agent pick the right tool? |
# MAGIC | `llm_call_count` | custom code | **trace** LLM spans | How many LLM calls (cost/latency proxy)? |
# MAGIC | `under_5_seconds` | custom code | **trace** duration | Is it fast enough? |
# MAGIC
# MAGIC **Code walk-through:** a `@scorer` function declares the arguments it needs (`inputs`, `outputs`, `expectations`,
# MAGIC `trace`) and MLflow passes them in. `trace.search_spans(span_type=...)` finds spans of one type.

# COMMAND ----------

from mlflow.genai.scorers import scorer, Correctness, RelevanceToQuery, Safety

@scorer
def used_expected_tool(trace, expectations) -> bool:
    tool_names = [s.name for s in trace.search_spans(span_type=SpanType.TOOL)]
    return expectations["expected_tool"] in tool_names

@scorer
def llm_call_count(trace) -> int:
    return len(trace.search_spans(span_type=SpanType.LLM))

@scorer
def under_5_seconds(trace) -> bool:
    return trace.info.execution_duration < 5000      # milliseconds

scorers = [Correctness(), RelevanceToQuery(), Safety(), used_expected_tool, llm_call_count, under_5_seconds]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Run the evaluation
# MAGIC **What:** `mlflow.genai.evaluate` calls `support_agent(question=...)` for each row (creating a trace each time),
# MAGIC runs every scorer, and logs results to an MLflow run in the experiment `fde_section6_agent_eval`.
# MAGIC **How to read the output:** aggregate metrics such as `correctness/mean`, `used_expected_tool/mean` (share of rows
# MAGIC where the right tool was used) and `llm_call_count/mean` (should be 2.0). Open the run → **Evaluations** tab to see
# MAGIC each row with its trace and judge rationales.

# COMMAND ----------

results = mlflow.genai.evaluate(data=eval_data, predict_fn=support_agent, scorers=scorers)
for name, value in sorted(results.metrics.items()):
    print(f"{name:35s} {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Per-row results: find the failing step
# MAGIC **What:** `mlflow.search_traces(run_id=...)` returns the traces of this evaluation run with their assessments.
# MAGIC **How to debug with this:**
# MAGIC | Correctness | used_expected_tool | Likely problem |
# MAGIC |---|---|---|
# MAGIC | ❌ | ❌ | tool choice (`choose_tool` prompt) |
# MAGIC | ❌ | ✅ | tool data or `write_answer` prompt |
# MAGIC | ✅ | ❌ | lucky answer – fix the router anyway |
# MAGIC
# MAGIC **How to read the output:** one row per question with the request, response and assessments.

# COMMAND ----------

traces_df = mlflow.search_traces(run_id=results.run_id)
cols = [c for c in ["trace_id", "request", "response", "assessments", "execution_duration"] if c in traces_df.columns]
display(traces_df[cols].astype(str))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - **Tracing** (`@mlflow.trace`, autologging) records each step with type, inputs, outputs and timing.
# MAGIC - **`mlflow.genai.evaluate(data, predict_fn, scorers)`** runs the agent and grades every row; results land in an
# MAGIC   MLflow run.
# MAGIC - Combine **built-in LLM judges** (Correctness needs `expected_facts`; RelevanceToQuery and Safety don't) with
# MAGIC   **code scorers that read the trace** (right tool? number of LLM calls? latency?).
# MAGIC - Use per-row results + traces to locate *which step* failed, fix it, and re-run the same evaluation to compare.
