# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.13 – Enable multi-agent systems to use Genie Spaces / the Conversation API to retrieve data
# MAGIC
# MAGIC **The big idea for juniors:**
# MAGIC - A **Genie Space** is a Databricks feature where business users ask questions about tables in plain English
# MAGIC   ("What were sales last week?") and Genie writes and runs the SQL for them.
# MAGIC - The **Genie Conversation API** lets *code* – including an AI agent – ask Genie the same questions.
# MAGIC - In a **multi-agent system**, a *supervisor* agent sends each part of a user's request to the right *specialist*:
# MAGIC   questions about **numbers in tables** → the **Genie data agent**; questions about **policies/documents** → a
# MAGIC   **knowledge (RAG) agent**. The supervisor then combines the answers.
# MAGIC
# MAGIC ```
# MAGIC                       ┌──────────────► DATA AGENT ──► Genie Space ──► SQL on Delta tables (orders, products)
# MAGIC user ──► SUPERVISOR ──┤
# MAGIC  ▲        (routes &   └──────────────► POLICY AGENT ──► policy documents (RAG)
# MAGIC  │         combines)
# MAGIC  └───────── one final answer ◄────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Why not let one LLM write SQL itself?** Genie already knows your tables, column meanings, example queries and
# MAGIC business definitions (set up by data owners), respects Unity Catalog permissions, and is reusable by people *and*
# MAGIC agents.
# MAGIC
# MAGIC **This notebook:** creates sample tables, shows how to call a Genie Space from code, wraps it as an agent, and builds a
# MAGIC supervisor. If you don't have a Genie Space yet, a clearly-labelled **stand-in** (LLM-generated SQL) lets every cell run.
# MAGIC
# MAGIC **Analogy:** a team lead (supervisor) with two specialists – a data analyst (Genie, knows the database) and a
# MAGIC policy expert (knows the handbook). The lead splits the customer's question, asks each specialist their part, and
# MAGIC writes one reply.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Genie Space** | A Genie "room" set up for specific tables, with instructions written by the data owners |
# MAGIC | **Conversation API** | The REST/SDK way for code to ask Genie questions |
# MAGIC | **Attachment** | A piece of Genie's reply: a text answer and/or the SQL query it wrote |
# MAGIC | **Specialist agent** | An agent that does one kind of job well (data or policy) |
# MAGIC | **Supervisor** | The agent that plans, routes sub-questions to specialists, and combines their answers |
# MAGIC | **Routing** | Deciding which specialist should handle a question |
# MAGIC | **Stand-in** | A simple replacement used here when no Genie Space is set up |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1  create tables products, orders            ← the data Genie queries
# MAGIC Step 2  genie_space_id widget                     ← empty = use the stand-in
# MAGIC Step 3  ask_genie()       ─┐
# MAGIC Step 4  standin_genie()   ─┴─► data_agent()       ← picks one based on the widget
# MAGIC Step 5  policy_agent()                            ← answers from POLICY_DOCS
# MAGIC Step 6  supervisor(): ROUTER plan → data_agent / policy_agent per task → final combined answer
# MAGIC Step 7  supervisor() on 3 more requests
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Create the sample tables Genie will query
# MAGIC **What:** `products` and `orders` Delta tables in `workspace.fde_app_dev`, with column comments.
# MAGIC **Why comments:** Genie (and any LLM writing SQL) uses table and column descriptions to understand the data.
# MAGIC
# MAGIC **Remember these facts to check answers later:**
# MAGIC - Orders **more than 3 days late**: 5003 (5 days) and 5004 (4 days) → **2 orders**.
# MAGIC - Revenue (quantity × price) by category: Laptops 3 × 1200 = **3600**, Audio 9 × 150 = **1350**, Monitors 2 × 400 =
# MAGIC   **800**, Accessories 3 × 75 = **225**.
# MAGIC
# MAGIC `mode("overwrite")` means re-running this cell replaces the tables rather than adding duplicate rows.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_app_dev"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

spark.createDataFrame([
    (1, "Laptop Pro 15", "Laptops", 1200.0), (2, "SoundMax Headphones", "Audio", 150.0),
    (3, "Mechanical Keyboard", "Accessories", 75.0), (4, "UltraWide Monitor", "Monitors", 400.0),
], "product_id int, product_name string, category string, price double") \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.products")

spark.createDataFrame([
    (5001, 1, 2, "2026-11-01", "Delivered", 0), (5002, 2, 5, "2026-11-03", "Delivered", 1),
    (5003, 3, 3, "2026-11-05", "Shipped", 5), (5004, 1, 1, "2026-11-10", "Shipped", 4),
    (5005, 4, 2, "2026-11-12", "Processing", 0), (5006, 2, 4, "2026-11-15", "Delivered", 0),
], "order_id int, product_id int, quantity int, order_date string, status string, days_late int") \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.orders")

spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.orders IS 'One row per customer order; join to products on product_id'")
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.orders ALTER COLUMN days_late COMMENT 'Days past the promised delivery date (0 = on time)'")
display(spark.table(f"{CATALOG}.{SCHEMA}.orders"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Create a Genie Space (one-time, in the UI)
# MAGIC 1. Left menu → **Genie** → **New**.
# MAGIC 2. Add tables `workspace.fde_app_dev.orders` and `workspace.fde_app_dev.products`.
# MAGIC 3. Add an instruction, e.g. *"Revenue = quantity × price. 'Late' means days_late > 3."*
# MAGIC 4. Ask it a test question in the UI, e.g. *"Revenue by category"*.
# MAGIC 5. Copy the **space ID** from the URL (`.../genie/rooms/<SPACE_ID>`) into the widget at the top of this notebook.
# MAGIC
# MAGIC Leave the widget empty to use the stand-in data agent instead.
# MAGIC
# MAGIC **About the widget:** the next cell adds a text box at the top of the notebook. After pasting a space ID, re-run
# MAGIC from this cell down. The printed line tells you which path will be used.

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id").strip()
print("Using Genie Space:", GENIE_SPACE_ID or "(none – stand-in data agent will be used)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Ask Genie from code (Conversation API)
# MAGIC **What:** the Databricks SDK calls the Genie Conversation API:
# MAGIC 1. `start_conversation_and_wait(space_id, question)` – sends the question, waits for Genie to finish.
# MAGIC 2. The reply has **attachments**: a *text* answer and/or a *query* (the SQL Genie wrote).
# MAGIC 3. For a query attachment we fetch the **result rows**.
# MAGIC
# MAGIC Same thing over REST: `POST /api/2.0/genie/spaces/{space_id}/start-conversation`, then poll
# MAGIC `GET .../conversations/{conversation_id}/messages/{message_id}`, then
# MAGIC `GET .../messages/{message_id}/attachments/{attachment_id}/query-result`.
# MAGIC Follow-up questions in the same conversation use `create_message_and_wait(space_id, conversation_id, text)`.
# MAGIC
# MAGIC **Code walk-through of `ask_genie()`:** send the question → loop over the attachments: copy any text answer; for a
# MAGIC query attachment copy the SQL and fetch its result rows. It returns one tidy dict:
# MAGIC `{"question", "sql", "text", "rows"}` – the same shape as the stand-in, so the supervisor doesn't care which is used.
# MAGIC
# MAGIC This cell only defines the function; it's called in Step 4 (and only if a space ID is set).

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def ask_genie(question):
    message = w.genie.start_conversation_and_wait(space_id=GENIE_SPACE_ID, content=question)
    answer = {"question": question, "sql": None, "text": None, "rows": None}
    for att in message.attachments or []:
        if att.text:
            answer["text"] = att.text.content
        if att.query:
            answer["sql"] = att.query.query
            result = w.genie.get_message_attachment_query_result(
                space_id=GENIE_SPACE_ID, conversation_id=message.conversation_id,
                message_id=message.id, attachment_id=att.attachment_id)
            data = result.statement_response.result
            answer["rows"] = data.data_array if data else []
    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Stand-in data agent (only used when no Genie Space ID is set)
# MAGIC **What:** an LLM writes one SQL query from the table schemas, we run it with Spark. This imitates what Genie does so
# MAGIC the rest of the notebook works – but real Genie adds curated instructions, example queries, permissions and a UI.
# MAGIC
# MAGIC **Code walk-through:**
# MAGIC - `SCHEMA_TEXT` describes the tables and the business rules ("late" = days_late > 3, revenue = quantity × price).
# MAGIC   That's the same kind of information you give Genie as instructions.
# MAGIC - `standin_genie()` asks the LLM for SQL only, strips any ``` code fences, and **refuses anything that isn't a
# MAGIC   SELECT/WITH query** – a simple guardrail so an LLM can never run `DROP TABLE`. Results are capped at 50 rows.
# MAGIC - `data_agent()` is the single entry point: Genie if a space ID is set, otherwise the stand-in.
# MAGIC
# MAGIC **How to read the output:** a JSON dict with the SQL that was written and the rows. Check the numbers against
# MAGIC Step 1: Laptops 3600, Audio 1350, Monitors 800, Accessories 225.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

def chat(system, user, max_tokens=300):
    response = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": max_tokens})
    return response["choices"][0]["message"]["content"].strip()

SCHEMA_TEXT = f"""Tables:
{CATALOG}.{SCHEMA}.products(product_id int, product_name string, category string, price double)
{CATALOG}.{SCHEMA}.orders(order_id int, product_id int, quantity int, order_date string, status string,
  days_late int -- days past promised date; 'late' means days_late > 3)
Revenue = quantity * price."""

def standin_genie(question):
    sql = chat("Write ONE Spark SQL SELECT query that answers the question. Return only the SQL, no explanation, "
               "no code fences.\n" + SCHEMA_TEXT, question)
    sql = sql.replace("```sql", "").replace("```", "").strip().rstrip(";")
    if not sql.lower().startswith(("select", "with")):                 # guardrail: read-only queries only
        return {"question": question, "sql": sql, "text": "Refused: not a SELECT query.", "rows": None}
    rows = [list(r) for r in spark.sql(sql).limit(50).collect()]
    return {"question": question, "sql": sql, "text": None, "rows": rows}

def data_agent(question):
    return ask_genie(question) if GENIE_SPACE_ID else standin_genie(question)

print(json.dumps(data_agent("What is total revenue by product category?"), indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – The policy (knowledge) agent
# MAGIC **What:** answers from policy documents only. (In production: a Vector Search retriever or a Knowledge Assistant.)
# MAGIC
# MAGIC **Why "say 'not covered by policy' otherwise":** gives the agent a safe way to say "I don't know" instead of
# MAGIC inventing a rule.
# MAGIC
# MAGIC **How to read the output:** an answer mentioning the investigation and the $10 store credit.

# COMMAND ----------

POLICY_DOCS = """Shipping: orders more than 3 days late qualify for an investigation and a $10 store credit.
Returns: most products can be returned within 30 days of delivery.
Warranty: defective products are replaced free within 15 days of delivery."""

def policy_agent(question):
    return chat(f"Answer ONLY from these policies; say 'not covered by policy' otherwise.\n{POLICY_DOCS}", question)

print(policy_agent("What do customers get when an order is late?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – The supervisor: route, call specialists, combine
# MAGIC **What:**
# MAGIC 1. **Route** – the LLM splits the request into sub-questions and assigns each to `data` or `policy` (returned as JSON).
# MAGIC 2. **Delegate** – call the matching agent for each sub-question.
# MAGIC 3. **Combine** – the LLM writes one answer from all the specialists' results, without inventing anything.
# MAGIC
# MAGIC **Why split the question:** "How many orders are late, and what do we owe those customers?" needs *numbers*
# MAGIC (Genie) **and** *policy* (documents). Neither agent alone can answer it.
# MAGIC
# MAGIC **Example plan the router should return:**
# MAGIC ```json
# MAGIC {"tasks": [{"agent": "data",   "question": "How many orders have days_late > 3?"},
# MAGIC            {"agent": "policy", "question": "What are customers entitled to when an order is late?"}]}
# MAGIC ```
# MAGIC
# MAGIC **Code walk-through of `supervisor()`:**
# MAGIC 1. Ask the LLM (with `ROUTER` instructions) for the plan and parse the JSON.
# MAGIC 2. For each task, call `data_agent` or `policy_agent`; collect everything in `findings` and print each step.
# MAGIC 3. One final LLM call writes the answer from `findings` only.
# MAGIC
# MAGIC **How to read the output:** the `[data]` and `[policy]` lines show the routing and each specialist's raw result.
# MAGIC The final answer should say **2 orders** are more than 3 days late and that they qualify for an investigation and a
# MAGIC **$10 store credit**.

# COMMAND ----------

ROUTER = """You route requests for a shop's support supervisor. Agents:
- data: questions answered from order/product TABLES (counts, totals, revenue, which orders, statuses).
- policy: questions about company POLICIES (returns, shipping compensation, warranty).
Split the user's request into sub-questions. Return ONLY JSON: {"tasks": [{"agent": "data|policy", "question": "..."}]}"""

def supervisor(request):
    text = chat(ROUTER, request, max_tokens=250)
    plan = json.loads(text[text.find("{"): text.rfind("}") + 1])
    findings = []
    for task in plan["tasks"]:
        result = data_agent(task["question"]) if task["agent"] == "data" else policy_agent(task["question"])
        findings.append({"agent": task["agent"], "question": task["question"], "result": result})
        print(f"[{task['agent']}] {task['question']}\n   -> {json.dumps(result, default=str)[:300]}\n")
    final = chat("You are the supervisor. Write one short, clear answer for the user using ONLY the findings. "
                 "Include the numbers from the data agent and the rules from the policy agent.",
                 f"User request: {request}\n\nFindings:\n{json.dumps(findings, default=str)}")
    return final

print("FINAL ANSWER:\n" + supervisor("How many orders are more than 3 days late, and what are those customers entitled to?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Try more requests
# MAGIC **How to read:** check the printed routing – pure data questions should go only to `data`, pure policy questions
# MAGIC only to `policy`, and mixed questions to both.
# MAGIC
# MAGIC | Request | Expected routing | Expected answer |
# MAGIC |---|---|---|
# MAGIC | Which category earned the most revenue? | data only | Laptops (3600) |
# MAGIC | Can a customer return headphones bought 20 days ago? | policy only | yes – within 30 days of delivery |
# MAGIC | List shipped orders + warranty rule | data **and** policy | orders 5003, 5004 + free replacement within 15 days |
# MAGIC
# MAGIC Each request makes several model calls (router, specialists, final answer), so this cell takes a little while.

# COMMAND ----------

for req in ["Which product category earned the most revenue?",
            "Can a customer return headphones bought 20 days ago?",
            "List the shipped orders and tell me the warranty rule for defective items."]:
    print("=" * 100, "\nREQUEST:", req)
    print("FINAL:", supervisor(req), "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production options on Databricks
# MAGIC - **Agent Bricks – Multi-Agent Supervisor:** low-code; add a Genie Space and a Knowledge Assistant as sub-agents.
# MAGIC - **`databricks-langchain` `GenieAgent`:** a ready-made LangChain/LangGraph agent that wraps a Genie Space, to use as a
# MAGIC   node in a LangGraph multi-agent supervisor.
# MAGIC - **Agent Framework (3.11):** wrap the supervisor in a `ResponsesAgent`, log it with resources
# MAGIC   (`DatabricksGenieSpace`, `DatabricksServingEndpoint`, `DatabricksTable`) so the deployed agent can authenticate.
# MAGIC
# MAGIC **Exam takeaways:** use **Genie** (via the **Conversation API** or `GenieAgent`) as the *structured-data* specialist in a
# MAGIC multi-agent system; a **supervisor** routes sub-questions to specialists (Genie for tables, RAG for documents) and
# MAGIC combines the results. Genie respects Unity Catalog permissions and uses curated instructions for accurate SQL.
