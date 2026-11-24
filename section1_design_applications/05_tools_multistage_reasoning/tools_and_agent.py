# Databricks notebook source
# MAGIC %md
# MAGIC # 05 – Define and order tools for multi-stage reasoning
# MAGIC Requirement for a delayed-order question:
# MAGIC **identify issue → look up shipping policy → look up order → recommend action → generate response**
# MAGIC
# MAGIC Part A is **Python orchestration** (fixed order, a chain). Part B is a **tool-calling agent**: the model decides
# MAGIC which tool to call based on the request and prior results.

# COMMAND ----------

# MAGIC %pip install -q databricks-langchain langchain-core
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### How a model call works (plain code, reused in every cell below)
# MAGIC Every model call is the same 3 steps: build a chat `messages` list, send it to the model serving endpoint
# MAGIC with `client.predict`, and pull the reply text out of `response["choices"][0]["message"]["content"]`.
# MAGIC
# MAGIC Part A calls `client.predict` directly; Part B uses `ChatDatabricks.bind_tools`, which sends the same request plus a `tools` list.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")          # authenticated client for this workspace

prompt = "Write a polite one-sentence reply: my order ORD1001 is 5 days late. Do not promise a delivery date."

# 1. Build the chat messages (system = role/rules, user = the actual request)
messages = [
    {"role": "system", "content": "You are a customer support assistant."},
    {"role": "user", "content": prompt},
]

# 2. Call the endpoint (OpenAI-compatible chat format)
response = client.predict(
    endpoint=ENDPOINT,                            # e.g. databricks-meta-llama-3-3-70b-instruct
    inputs={"messages": messages, "temperature": 0.0, "max_tokens": 500},
)

# 3. The raw response: the answer is in choices[0].message.content
print(json.dumps(response, indent=2, default=str)[:1500])

# 4. Extract just the text
answer = response["choices"][0]["message"]["content"].strip()
print("\nAnswer:", answer)

# COMMAND ----------

# MAGIC %md ### Tools

# COMMAND ----------

def classify_issue(message: str) -> str:
    """Identify the ticket category for a customer message."""
    prompt = ("Return one category only, no other text: Shipping, Return, Product Defect, Payment, Wrong Item, Positive Feedback, Other.\n\n"
              f"Message: {message}")
    response = client.predict(
        endpoint=ENDPOINT,
        inputs={
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
        },
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer.strip(".")


def policy_lookup(category: str) -> str:
    """Return the company policy text for a ticket category."""
    row = spark.table("customer_support_policies").filter(f"policy_category = '{category}'").first()
    return row.policy_text if row else "No matching policy found."


def order_lookup(ticket_id: int) -> dict:
    """Return order_id, order_status and delay_days for a ticket."""
    result = spark.sql(f"""
        SELECT order_id, order_status, delay_days
        FROM customer_orders WHERE ticket_id = {int(ticket_id)}
    """).first()
    if result is None:
        return {"error": "Order not found"}
    return result.asDict()


def recommend_action(category: str, delay_days: int = 0) -> str:
    """Recommend the next support action from category and delivery delay."""
    if category == "Shipping" and delay_days > 3:
        return "Escalate to shipping investigation"
    elif category == "Payment":
        return "Send to payment investigation team"
    elif category == "Product Defect":
        return "Initiate replacement process"
    return "Send to customer support team"

# COMMAND ----------

# MAGIC %md ### Part A – fixed-order Python orchestration

# COMMAND ----------

ticket_id = 101
message = spark.table("customer_support_tickets").filter(f"ticket_id = {ticket_id}").first().customer_message

category = classify_issue(message)                    # 1. identify issue
policy = policy_lookup(category)                      # 2. policy
order = order_lookup(ticket_id)                       # 3. order
action = recommend_action(category, order.get("delay_days", 0))  # 4. action

print("Category:", category)
print("Order:   ", order)
print("Policy:  ", policy)
print("Action:  ", action)

final_prompt = f"""You are a customer support assistant.
Customer issue: {message}
Order details: {order}
Company policy: {policy}
Recommended action: {action}
Write a polite response to the customer.
Do not promise a delivery date that is not available."""
response = client.predict(
    endpoint=ENDPOINT,
    inputs={
        "messages": [
            {"role": "user", "content": final_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    },
)
answer = response["choices"][0]["message"]["content"].strip()
print("\n--- Response (5. generate) ---\n" + answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part B – tool-calling agent (the model chooses tools and their order)

# COMMAND ----------

import json
from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

tools = {f.__name__: tool(f) for f in [classify_issue, policy_lookup, order_lookup, recommend_action]}
agent_llm = ChatDatabricks(endpoint=ENDPOINT, temperature=0, max_retries=5).bind_tools(list(tools.values()))

def run_agent(question: str, max_steps: int = 8) -> str:
    msgs = [
        SystemMessage("You are a support agent. Use the tools to gather facts before answering. "
                      "Typical order: classify_issue, order_lookup, policy_lookup, recommend_action. "
                      "Then write a short polite reply. Never invent delivery dates."),
        HumanMessage(question),
    ]
    for step in range(max_steps):
        ai = agent_llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            return ai.content
        for call in ai.tool_calls:
            result = tools[call["name"]].invoke(call["args"])
            print(f"step {step + 1}: {call['name']}({call['args']}) -> {result}")
            msgs.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"]))
    return "Stopped: too many steps."

print("\n" + run_agent("Ticket 101: My order has not arrived. It is 5 days late. What will you do?"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Key distinction:** a *chain* follows a predefined sequence (Part A); an *agent* decides which tools to use (Part B);
# MAGIC a *multi-agent system* coordinates specialized agents (see `06_agent_bricks`).
