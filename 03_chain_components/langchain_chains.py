# Databricks notebook source
# MAGIC %md
# MAGIC # 03 – Select chain components for a desired model input and output
# MAGIC Flow: **Input (complaint) → PromptTemplate → LLM → OutputParser → Structured ticket**
# MAGIC
# MAGIC | Component | Responsibility |
# MAGIC |---|---|
# MAGIC | Input | Receives customer message |
# MAGIC | PromptTemplate | Constructs model instructions |
# MAGIC | LLM | Processes the prompt |
# MAGIC | OutputParser | Converts output to desired format |
# MAGIC | RunnableSequence | Executes components in order (`a | b | c`) |
# MAGIC | Retriever | Retrieves relevant documents |
# MAGIC | Memory | Maintains conversation context |
# MAGIC
# MAGIC > Exam tip: dictionary output → JSON parser; plain string → string parser; needs company knowledge → add a retriever before the LLM.

# COMMAND ----------

# MAGIC %pip install -q databricks-langchain langchain langchain-core
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plain-code equivalent of `ask_llm` (for understanding)
# MAGIC `ask_llm(prompt)` from `00_setup/config` is just this: build a chat `messages` list, send it to the
# MAGIC model serving endpoint, and pull the reply text out of the response. (`ask_llm` also retries on HTTP 429 rate limits.)
# MAGIC
# MAGIC LangChain's `ChatDatabricks` (used below) does the same job: `prompt` builds the messages, `llm` sends the same chat request to the serving endpoint, and `StrOutputParser` is the `["choices"][0]["message"]["content"]` step.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")          # authenticated client for this workspace

prompt = "Classify the following support ticket. Return only:\nCategory: <category>\nPriority: <priority>\n\nCustomer message:\nMy order is delayed by 5 days."

# 1. Build the chat messages (system = role/rules, user = the actual request)
messages = [
    # optional: {"role": "system", "content": "You are ..."},
    {"role": "user", "content": prompt},
]

# 2. Call the endpoint (OpenAI-compatible chat format)
response = client.predict(
    endpoint=ENDPOINT,                            # e.g. databricks-meta-llama-3-3-70b-instruct
    inputs={"messages": messages, "temperature": 0.0, "max_tokens": 500},
)

# 3. The raw response: the answer is in choices[0].message.content
print(json.dumps(response, indent=2, default=str)[:1500])

# 4. Extract just the text - this is what ask_llm returns
answer = response["choices"][0]["message"]["content"].strip()
print("\nAnswer:", answer)

# COMMAND ----------

# Same thing in one line with the helper
print(ask_llm(prompt))

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

llm = ChatDatabricks(endpoint=ENDPOINT, temperature=0, max_retries=5)

# COMMAND ----------

# MAGIC %md ### 1. Prompt → LLM → StrOutputParser (plain text out)

# COMMAND ----------

prompt = ChatPromptTemplate.from_template("""
Classify the following support ticket.
Return only:
Category: <category>
Priority: <priority>

Customer message:
{message}
""")

text_chain = prompt | llm | StrOutputParser()
print(text_chain.invoke({"message": "My order is delayed by 5 days."}))

# COMMAND ----------

# MAGIC %md ### 2. Prompt → LLM → JsonOutputParser (Python dict out)

# COMMAND ----------

json_prompt = ChatPromptTemplate.from_template("""
Classify the support ticket. Return ONLY JSON with keys category, sentiment, priority.
category: Shipping, Return, Product Defect, Payment, Wrong Item, Positive Feedback, Other.

Customer message:
{message}
""")
json_chain = json_prompt | llm | JsonOutputParser()
ticket = json_chain.invoke({"message": "I was charged twice for the same order."})
print(type(ticket), ticket)

# COMMAND ----------

# MAGIC %md ### 3. Batch the chain over the ticket table

# COMMAND ----------

tickets = spark.table("customer_support_tickets").collect()
outs = json_chain.batch([{"message": t.customer_message} for t in tickets])
display(spark.createDataFrame(
    [(t.ticket_id, t.customer_message, o.get("category"), o.get("sentiment"), o.get("priority")) for t, o in zip(tickets, outs)],
    "ticket_id int, customer_message string, category string, sentiment string, priority string"))

# COMMAND ----------

# MAGIC %md ### 4. Add a Retriever before the LLM (company knowledge)
# MAGIC A simple retriever over the `customer_support_policies` Delta table. In production this would be a Vector Search retriever.

# COMMAND ----------

POLICIES = {r.policy_category: r.policy_text for r in spark.table("customer_support_policies").collect()}

def retrieve_policy(inputs: dict) -> str:
    cat = json_chain.invoke({"message": inputs["message"]}).get("category", "Other")
    return f"[{cat}] {POLICIES.get(cat, 'No matching policy found.')}"

rag_prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant. Answer using ONLY the policy below.

Policy: {policy}
Customer message: {message}

Reply in 2-3 sentences and cite the policy category in brackets.
""")

rag_chain = (
    RunnablePassthrough.assign(policy=RunnableLambda(retrieve_policy))
    | rag_prompt | llm | StrOutputParser()
)
print(rag_chain.invoke({"message": "The headphones stopped working after 2 days. Can I get a new pair?"}))

# COMMAND ----------

# MAGIC %md ### 5. Memory – keep conversation context

# COMMAND ----------

from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise e-commerce support assistant."),
    MessagesPlaceholder("history"),
    ("human", "{message}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()

history = []
for msg in ["Hi, my laptop order ORD1001 is 5 days late.", "What order number did I mention?"]:
    reply = chat_chain.invoke({"history": history, "message": msg})
    history += [HumanMessage(msg), AIMessage(reply)]
    print(f"USER: {msg}\nBOT: {reply}\n")
