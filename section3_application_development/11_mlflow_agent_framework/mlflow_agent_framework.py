# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.11 – Use MLflow and the Mosaic AI Agent Framework to develop agentic systems
# MAGIC
# MAGIC **The big idea for juniors:** an **agent** is an LLM that can *decide* to use **tools** (functions that look up data
# MAGIC or take actions), look at the results, and keep going until it can answer. Building one is the easy part –
# MAGIC making it **observable, testable and deployable** is where MLflow and the Agent Framework come in.
# MAGIC
# MAGIC | Need | What Databricks gives you |
# MAGIC |---|---|
# MAGIC | A standard agent interface | MLflow **`ResponsesAgent`** – one `predict()` contract that serving, the Playground and the Review App understand |
# MAGIC | See what the agent did, step by step | MLflow **Tracing** – every LLM call and tool call recorded as a "span" |
# MAGIC | Package the agent with its dependencies | `mlflow.pyfunc.log_model` (+ `resources` so auth to endpoints/tables is set up automatically) |
# MAGIC | Test quality | `mlflow.genai.evaluate` with scorers (code checks or LLM judges) |
# MAGIC | Version & govern | Register the model in **Unity Catalog** |
# MAGIC | Deploy with a REST API, Review App and monitoring | `databricks.agents.deploy()` |
# MAGIC
# MAGIC **Lifecycle we follow:** build tools → write the agent → trace it → log it → load & test → evaluate → (register & deploy).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 0 – Install a recent MLflow (ResponsesAgent and `mlflow.genai` need MLflow 3)

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Data the tools will use
# MAGIC **What:** a small orders "database" and policy book, plus the model endpoint name.

# COMMAND ----------

ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

ORDERS = {
    "1001": {"product": "Laptop Pro 15", "status": "Delivered", "days_late": 0},
    "1002": {"product": "SoundMax Headphones", "status": "Shipped", "days_late": 5},
}
POLICIES = {
    "shipping": "Orders delayed more than 3 days qualify for an investigation and a $10 store credit.",
    "returns": "Most products can be returned within 30 days of delivery.",
    "warranty": "Defective products are replaced free within 15 days of delivery.",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Define tools (and trace them)
# MAGIC **What:** two Python functions plus a **tool description** in the JSON format chat models understand
# MAGIC (name, what it does, which arguments it takes).
# MAGIC **Why `@mlflow.trace(span_type="TOOL")`:** every time a tool runs, MLflow records its input, output and duration,
# MAGIC so later you can see *exactly* which tools the agent used and why an answer came out the way it did.
# MAGIC **Production tip:** tools can also be **Unity Catalog functions** (SQL/Python functions governed in UC) so they're
# MAGIC shareable and permission-controlled.

# COMMAND ----------

import json
import mlflow

@mlflow.trace(span_type="TOOL")
def get_order(order_id: str) -> dict:
    """Return status details for an order."""
    return ORDERS.get(str(order_id), {"error": f"order {order_id} not found"})

@mlflow.trace(span_type="TOOL")
def get_policy(topic: str) -> str:
    """Return the company policy for shipping, returns or warranty."""
    return POLICIES.get(topic.lower(), "No policy found for that topic.")

TOOL_FUNCTIONS = {"get_order": get_order, "get_policy": get_policy}
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_order", "description": "Look up an order's product, status and days late.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_policy", "description": "Get the policy text for a topic: shipping, returns or warranty.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": ["shipping", "returns", "warranty"]}},
                       "required": ["topic"]}}},
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Write the agent as an MLflow `ResponsesAgent`
# MAGIC **What the `predict` method does (the "agent loop"):**
# MAGIC 1. Send the conversation + tool descriptions to the LLM.
# MAGIC 2. If the LLM asks for tool calls → run each tool, add the results to the conversation, go back to 1.
# MAGIC 3. If the LLM replies with plain text → that's the final answer. Stop.
# MAGIC 4. Safety limit: at most `MAX_STEPS` rounds, so a confused agent can't loop forever.
# MAGIC
# MAGIC **Why `ResponsesAgent`:** it gives the agent a standard input/output format (OpenAI "Responses" style), so the
# MAGIC same code works in the notebook, on a serving endpoint, in AI Playground and in the Review App.

# COMMAND ----------

from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

SYSTEM_PROMPT = ("You are ShopEase's support agent. Use the tools to look up orders and policies before answering. "
                 "Answer in at most 3 sentences and never invent facts.")
MAX_STEPS = 5

class SupportAgent(ResponsesAgent):

    @mlflow.trace(span_type="LLM")
    def call_llm(self, messages):
        from mlflow.deployments import get_deploy_client           # created inside, so the agent can be saved & reloaded
        response = get_deploy_client("databricks").predict(
            endpoint=ENDPOINT,
            inputs={"messages": messages, "tools": TOOL_SPECS, "temperature": 0.0, "max_tokens": 300})
        return response["choices"][0]["message"]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": item.role, "content": item.content} for item in request.input]

        for _ in range(MAX_STEPS):
            reply = self.call_llm(messages)
            tool_calls = reply.get("tool_calls") or []
            if not tool_calls:                                       # no tools requested -> final answer
                return ResponsesAgentResponse(output=[self.create_text_output_item(text=reply["content"], id=str(uuid4()))])
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": tool_calls})
            for call in tool_calls:                                  # run every tool the model asked for
                args = json.loads(call["function"]["arguments"] or "{}")
                result = TOOL_FUNCTIONS[call["function"]["name"]](**args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})

        return ResponsesAgentResponse(output=[self.create_text_output_item(
            text="Sorry, I couldn't finish that request. Let me connect you with a human agent.", id=str(uuid4()))])

agent = SupportAgent()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Try it and look at the trace
# MAGIC **What:** ask a question that needs **both** tools (order status + shipping policy).
# MAGIC **How to read the trace:** below the cell output, MLflow shows a **Trace** panel. Expand it: you'll see the LLM call,
# MAGIC then `get_order` and `get_policy` spans with their inputs/outputs, then the final LLM call. This is how you debug agents.

# COMMAND ----------

def answer_text(response):
    data = response if isinstance(response, dict) else response.model_dump()
    return data["output"][-1]["content"][0]["text"]

question = {"input": [{"role": "user", "content": "Order 1002 still hasn't arrived. What's going on and do I get anything?"}]}
response = agent.predict(question)
print(answer_text(response))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Log the agent to MLflow
# MAGIC **What:** save the agent as an MLflow model, with:
# MAGIC - `input_example` – a sample request (documents the expected input),
# MAGIC - `resources` – the serving endpoint it calls, so a deployed agent gets **automatic authentication** to it,
# MAGIC - `pip_requirements` – the packages to install when serving.
# MAGIC
# MAGIC **Why:** a logged model is a versioned, reproducible package. Production teams usually log agents
# MAGIC **"from code"** (`python_model="agent.py"`); for learning we log the object directly.

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_agent_framework")

with mlflow.start_run(run_name="support_agent_v1"):
    logged = mlflow.pyfunc.log_model(
        name="support_agent",
        python_model=agent,
        input_example=question,
        resources=[DatabricksServingEndpoint(endpoint_name=ENDPOINT)],
        pip_requirements=["mlflow>=3.1"],
    )
print("Logged model URI:", logged.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Load the logged agent back and test it (what serving will do)
# MAGIC **Why:** proves the package works on its own, not just inside this notebook's memory.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(logged.model_uri)
print(answer_text(loaded.predict({"input": [{"role": "user", "content": "What is your returns policy?"}]})))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Evaluate with `mlflow.genai.evaluate`
# MAGIC **What:** a small evaluation dataset (question + facts the answer must contain) and a **code-based scorer** that
# MAGIC checks those facts. Results (and a trace per question) appear in the experiment's **Evaluations** tab.
# MAGIC **Why:** before deploying, prove the agent answers correctly. You can add built-in LLM judges such as
# MAGIC `Correctness()` or `RelevanceToQuery()` from `mlflow.genai.scorers` for open-ended quality.

# COMMAND ----------

from mlflow.genai.scorers import scorer

eval_data = [
    {"inputs": {"question": "Where is order 1002?"}, "expectations": {"must_contain": ["5"]}},
    {"inputs": {"question": "How long do I have to return something?"}, "expectations": {"must_contain": ["30 days"]}},
    {"inputs": {"question": "Is order 1001 delivered?"}, "expectations": {"must_contain": ["delivered"]}},
]

def predict_fn(question):
    return answer_text(agent.predict({"input": [{"role": "user", "content": question}]}))

@scorer
def contains_expected_facts(outputs, expectations):
    """1.0 if every required fact appears in the answer, else 0.0."""
    return float(all(fact.lower() in str(outputs).lower() for fact in expectations["must_contain"]))

try:
    results = mlflow.genai.evaluate(data=eval_data, predict_fn=predict_fn, scorers=[contains_expected_facts])
    print(results.metrics)
except Exception as e:
    print("mlflow.genai.evaluate is not available in this environment:", str(e)[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Register and deploy (shown, not run: creates a serving endpoint that runs continuously)
# MAGIC ```python
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC registered = mlflow.register_model(logged.model_uri, "workspace.fde_app_dev.support_agent")   # versioned in Unity Catalog
# MAGIC
# MAGIC %pip install databricks-agents
# MAGIC from databricks import agents
# MAGIC agents.deploy("workspace.fde_app_dev.support_agent", registered.version)
# MAGIC # -> REST endpoint + Review App for expert feedback + inference tables / monitoring
# MAGIC ```
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - Author agents with MLflow's **`ResponsesAgent`** interface (works with any library: plain SDK, LangChain, LangGraph…).
# MAGIC - **Trace** LLM and tool calls (`@mlflow.trace`, or `mlflow.langchain.autolog()` / `mlflow.openai.autolog()`).
# MAGIC - **Log** with `resources` for automatic auth, **evaluate** with `mlflow.genai.evaluate`, **register** in Unity Catalog,
# MAGIC   **deploy** with `agents.deploy()` (gives a REST API, Review App and monitoring).
