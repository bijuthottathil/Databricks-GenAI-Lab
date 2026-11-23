# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.11 – Configure a persistent datastore to store and retrieve intermediate memory or structured information
# MAGIC
# MAGIC **The big idea for juniors:** an LLM **forgets everything** between calls. A deployed endpoint may also restart or
# MAGIC run on several machines, so Python variables disappear too. If the assistant must remember the conversation, the
# MAGIC customer's preferences, or the progress of a multi-step task, that information must be **saved in a datastore**
# MAGIC and **loaded back** on the next request.
# MAGIC
# MAGIC ```
# MAGIC request (session_id, customer_id, message)
# MAGIC    │ 1. LOAD  recent turns (short-term memory) + customer facts (long-term memory)
# MAGIC    ▼
# MAGIC  prompt = facts + history + new message ──► LLM ──► reply
# MAGIC    │ 2. SAVE  the new turns;  extract new facts → MERGE into the facts table
# MAGIC    ▼
# MAGIC Delta tables (or Lakebase Postgres)  ← survives restarts, shared by all serving replicas
# MAGIC ```
# MAGIC
# MAGIC ### Types of memory
# MAGIC | Memory | Example | Stored as |
# MAGIC |---|---|---|
# MAGIC | **Short-term** (conversation) | the last 10 messages of this chat | rows per turn, keyed by `session_id` |
# MAGIC | **Long-term** (structured facts) | "prefers email", "owns a Laptop Pro 15" | key/value rows per `customer_id` |
# MAGIC | **Intermediate / task state** | "step 2 of 4 done, refund approved" | a row per task with status columns |
# MAGIC
# MAGIC ### Where to store it on Databricks
# MAGIC | Store | Good for |
# MAGIC |---|---|
# MAGIC | **Delta table** (this notebook) | simple, governed by Unity Catalog, easy to analyse; writes take ~seconds |
# MAGIC | **Lakebase (managed Postgres)** | low-latency reads/writes per request – production chat memory, LangGraph checkpoints |
# MAGIC | **Vector Search index** | "semantic" memory – find *relevant* past notes rather than the latest ones |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Stateless** | Remembers nothing between requests (LLMs and serving replicas are) |
# MAGIC | **Session / thread id** | An id that groups the messages of one conversation |
# MAGIC | **MERGE (upsert)** | Update the row if it exists, insert it if not |
# MAGIC | **Checkpointer** | LangGraph's component that saves agent state after each step |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 create tables: chat_memory, customer_facts, task_state
# MAGIC Step 2 save_turn() / load_history()       Step 3 upsert_fact() / load_facts()
# MAGIC Step 4 extract_facts() with the LLM       Step 5 chat() = load → prompt → LLM → save
# MAGIC Step 6 session 1 conversation             Step 7 NEW session 2 still knows the customer
# MAGIC Step 8 task state for a multi-step process   Step 9 production notes (Lakebase, retention, PII)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Create the memory tables
# MAGIC | Table | Primary key | Holds |
# MAGIC |---|---|---|
# MAGIC | `chat_memory` | (session_id, turn) | every message: role + content + time |
# MAGIC | `customer_facts` | (customer_id, fact_key) | long-term facts as key/value |
# MAGIC | `task_state` | task_id | progress of a multi-step task |
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` keeps memory between runs – that's the point of *persistent*. The `DELETE` lines reset
# MAGIC only our demo customer so the demo starts clean each time.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_deploy"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"""CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.chat_memory
              (session_id STRING, customer_id STRING, turn INT, role STRING, content STRING, created_at TIMESTAMP)""")
spark.sql(f"""CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.customer_facts
              (customer_id STRING, fact_key STRING, fact_value STRING, updated_at TIMESTAMP)""")
spark.sql(f"""CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.task_state
              (task_id STRING, customer_id STRING, step STRING, status STRING, details STRING, updated_at TIMESTAMP)""")

CUSTOMER = "cust-42"
for t in ["chat_memory", "customer_facts", "task_state"]:
    spark.sql(f"DELETE FROM {CATALOG}.{SCHEMA}.{t} WHERE customer_id = '{CUSTOMER}'")
print("Memory tables ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Short-term memory: save and load conversation turns
# MAGIC **Code walk-through:**
# MAGIC - `save_turn` – find the next turn number for the session, insert one row.
# MAGIC - `load_history` – the last `n` turns in order, as chat messages `{"role", "content"}` ready for the LLM.
# MAGIC
# MAGIC **Why only the last `n`:** the prompt has a size limit and every token costs money – old turns are dropped (or
# MAGIC summarised into a fact).
# MAGIC **Why parameters (`:sid`):** values are passed separately from the SQL text, so user text can't break the query
# MAGIC (SQL injection).

# COMMAND ----------

def save_turn(session_id, customer_id, role, content):
    next_turn = spark.sql(f"SELECT coalesce(max(turn), 0) + 1 AS t FROM {CATALOG}.{SCHEMA}.chat_memory WHERE session_id = :sid",
                          args={"sid": session_id}).first()["t"]
    spark.sql(f"""INSERT INTO {CATALOG}.{SCHEMA}.chat_memory
                  VALUES (:sid, :cid, :turn, :role, :content, current_timestamp())""",
              args={"sid": session_id, "cid": customer_id, "turn": next_turn, "role": role, "content": content})

def load_history(session_id, n=10):
    rows = spark.sql(f"""SELECT role, content FROM (
                           SELECT * FROM {CATALOG}.{SCHEMA}.chat_memory WHERE session_id = :sid ORDER BY turn DESC LIMIT {int(n)})
                         ORDER BY turn""", args={"sid": session_id}).collect()
    return [{"role": r.role, "content": r.content} for r in rows]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Long-term memory: structured facts with MERGE
# MAGIC **What:** `upsert_fact` uses `MERGE` – if the customer already has that fact key, update it; otherwise insert.
# MAGIC **Why key/value:** we can't predict every fact in advance; key/value rows handle new kinds of facts without changing
# MAGIC the table. (If you know the fields, a normal column per field is even better.)

# COMMAND ----------

def upsert_fact(customer_id, key, value):
    spark.sql(f"""
    MERGE INTO {CATALOG}.{SCHEMA}.customer_facts t
    USING (SELECT :cid AS customer_id, :k AS fact_key, :v AS fact_value) s
    ON t.customer_id = s.customer_id AND t.fact_key = s.fact_key
    WHEN MATCHED THEN UPDATE SET fact_value = s.fact_value, updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (customer_id, fact_key, fact_value, updated_at)
                          VALUES (s.customer_id, s.fact_key, s.fact_value, current_timestamp())
    """, args={"cid": customer_id, "k": key, "v": value})

def load_facts(customer_id):
    rows = spark.sql(f"SELECT fact_key, fact_value FROM {CATALOG}.{SCHEMA}.customer_facts WHERE customer_id = :cid",
                     args={"cid": customer_id}).collect()
    return {r.fact_key: r.fact_value for r in rows}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Let the LLM extract facts worth remembering
# MAGIC **What:** after each user message, ask the LLM for durable facts as JSON (e.g. `{"product_owned": "Laptop Pro 15"}`).
# MAGIC Only a fixed list of keys is allowed, so the facts table stays tidy.
# MAGIC **Why not save everything:** memory should hold **useful, long-lived** information – not every sentence.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
FACT_KEYS = ["name", "product_owned", "preferred_contact", "order_id", "open_issue"]

def llm(messages, max_tokens=200):
    r = client.predict(endpoint=ENDPOINT, inputs={"messages": messages, "temperature": 0.0, "max_tokens": max_tokens})
    return r["choices"][0]["message"]["content"].strip()

def extract_facts(message):
    text = llm([{"role": "system", "content": f"Extract durable customer facts from the message. Allowed keys: {FACT_KEYS}. "
                                               "Return ONLY a JSON object with the keys you found (or {} if none)."},
                {"role": "user", "content": message}], max_tokens=120)
    try:
        facts = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {}
    return {k: str(v) for k, v in facts.items() if k in FACT_KEYS and v}

print(extract_facts("Hi, I'm Priya. My Laptop Pro 15 from order 1001 has a flickering screen. Please email me."))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – The chat function: load → prompt → LLM → save
# MAGIC **Code walk-through:**
# MAGIC 1. **Load** long-term facts for the customer and short-term history for the session.
# MAGIC 2. **Build** the prompt: system message (with the facts) + history + the new message.
# MAGIC 3. **Call** the LLM.
# MAGIC 4. **Save** the user message and the reply; extract and upsert new facts.
# MAGIC
# MAGIC Because everything is read from and written to tables, this function holds **no state of its own** – any serving
# MAGIC replica could run it.

# COMMAND ----------

def chat(session_id, customer_id, message):
    facts = load_facts(customer_id)
    history = load_history(session_id)
    system = ("You are ShopEase support. Be brief (max 2 sentences). Known facts about this customer: "
              + (json.dumps(facts) if facts else "none yet"))
    reply = llm([{"role": "system", "content": system}] + history + [{"role": "user", "content": message}])
    save_turn(session_id, customer_id, "user", message)
    save_turn(session_id, customer_id, "assistant", reply)
    for k, v in extract_facts(message).items():
        upsert_fact(customer_id, k, v)
    return reply

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Session 1: a short conversation
# MAGIC **How to read the output:** the second answer should refer back to the laptop without the customer repeating it –
# MAGIC it came from `chat_memory` (short-term). The facts table now holds name, product, contact preference, order id.

# COMMAND ----------

print("BOT:", chat("session-1", CUSTOMER, "Hi, I'm Priya. My Laptop Pro 15 from order 1001 has a flickering screen."))
print("BOT:", chat("session-1", CUSTOMER, "Is it still under warranty? Please contact me by email."))
display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.customer_facts WHERE customer_id = '{CUSTOMER}'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Session 2 (a new chat, days later): long-term memory at work
# MAGIC **What:** a brand-new `session_id` – so **no** conversation history – but the same customer.
# MAGIC **How to read the output:** the bot should still know it's Priya with a Laptop Pro 15 and that she prefers email,
# MAGIC because those facts were loaded from `customer_facts`. Below: the full stored conversation of session 1.

# COMMAND ----------

print("BOT:", chat("session-2", CUSTOMER, "Hello again, any update on my issue?"))
display(spark.sql(f"SELECT session_id, turn, role, content FROM {CATALOG}.{SCHEMA}.chat_memory "
                  f"WHERE customer_id = '{CUSTOMER}' ORDER BY session_id, turn"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Intermediate task state (multi-step processes)
# MAGIC **What:** a replacement process has steps (verify → approve → ship). Each step writes its status, so if the agent
# MAGIC or endpoint restarts, it **resumes** from the last completed step instead of starting over.
# MAGIC **How to read the output:** three rows; `next step` = the first one not `done`.

# COMMAND ----------

def set_step(task_id, customer_id, step, status, details=""):
    spark.sql(f"""
    MERGE INTO {CATALOG}.{SCHEMA}.task_state t
    USING (SELECT :tid AS task_id, :cid AS customer_id, :step AS step, :status AS status, :d AS details) s
    ON t.task_id = s.task_id AND t.step = s.step
    WHEN MATCHED THEN UPDATE SET status = s.status, details = s.details, updated_at = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (task_id, customer_id, step, status, details, updated_at)
                          VALUES (s.task_id, s.customer_id, s.step, s.status, s.details, current_timestamp())
    """, args={"tid": task_id, "cid": customer_id, "step": step, "status": status, "d": details})

set_step("replace-1001", CUSTOMER, "1_verify_warranty", "done", "delivered 2026-11-02, within 15 days")
set_step("replace-1001", CUSTOMER, "2_approve_replacement", "done", "auto-approved")
set_step("replace-1001", CUSTOMER, "3_ship_replacement", "pending")

state = spark.sql(f"SELECT step, status, details FROM {CATALOG}.{SCHEMA}.task_state WHERE task_id = 'replace-1001' ORDER BY step")
display(state)
print("next step:", next((r.step for r in state.collect() if r.status != "done"), "all done"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9 – Production notes
# MAGIC - **Latency:** Delta is great for analytics and modest traffic. For per-request reads/writes at chat speed, use
# MAGIC   **Lakebase** (managed Postgres) – e.g. `databricks-langchain`'s LangGraph `CheckpointSaver` stores agent state there.
# MAGIC - **Declare it as a resource** when logging the agent (e.g. `DatabricksTable(...)` or the Lakebase instance) so the
# MAGIC   serving endpoint is allowed to read/write it (4.2).
# MAGIC - **Privacy:** memory can contain PII – mask it, restrict access with UC grants, and set a **retention** policy
# MAGIC   (e.g. delete chat turns older than 90 days).
# MAGIC - **Size:** load only the last N turns; summarise older ones into facts.
# MAGIC
# MAGIC **Exam takeaways:** LLMs and serving replicas are **stateless** – persist **short-term** (conversation by
# MAGIC session id), **long-term** (structured facts by user, upserted with MERGE) and **intermediate task state** in a
# MAGIC datastore (Delta tables, Lakebase Postgres, or Vector Search for semantic memory), load it at the start of each
# MAGIC request and save at the end.
