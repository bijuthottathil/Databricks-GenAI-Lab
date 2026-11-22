# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.1 – Select LangChain (or similar tools) for a Generative AI application
# MAGIC
# MAGIC **The big idea for juniors:** you *can* build a GenAI app by calling the model directly (send text → get text).
# MAGIC Frameworks like **LangChain** give you ready-made building blocks so you don't re-write the same plumbing:
# MAGIC prompt templates, output parsers, retrievers, tool calling, memory, and agents.
# MAGIC
# MAGIC **Analogy:** cooking from raw ingredients (plain SDK) vs using a meal kit (framework). The meal kit is faster and
# MAGIC tidier for common recipes; raw ingredients give full control for unusual ones.
# MAGIC
# MAGIC | Tool | What it is best at | Pick it when… |
# MAGIC |---|---|---|
# MAGIC | **Plain SDK** (`mlflow.deployments`, OpenAI client) | One model call, full control, fewest dependencies | Simple tasks: classify, summarise, extract |
# MAGIC | **LangChain** (`databricks-langchain`) | Chains, retrievers, tools, agents, many integrations | Multi-step apps, RAG, tool calling |
# MAGIC | **LangGraph** | Agents as a graph of steps with loops and state | Complex agents, multi-agent, human-in-the-loop |
# MAGIC | **LlamaIndex** | Document loading, indexing, query engines | Document-heavy RAG |
# MAGIC | **DSPy** | Optimising prompts automatically from examples | You have labelled data and want better prompts |
# MAGIC | **Databricks SQL AI Functions** (`ai_query`, `ai_classify`…) | LLM calls on whole tables in SQL | Batch jobs over Delta tables |
# MAGIC | **Mosaic AI Agent Framework + MLflow** | Log, evaluate, trace and deploy agents (any framework) | Taking an agent to production on Databricks |
# MAGIC
# MAGIC **In this notebook** we solve the *same* support-ticket task three ways and compare how much code each needs:
# MAGIC 1. Plain SDK  2. LangChain chain  3. LangChain with a retriever + tool (where a framework really pays off).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 0 – Install LangChain's Databricks integration
# MAGIC **What:** `databricks-langchain` lets LangChain talk to Databricks model endpoints (and Vector Search, Genie…).
# MAGIC **Why restart:** Python must reload so the newly installed package can be imported.

# COMMAND ----------

# MAGIC %pip install -q databricks-langchain langchain-core
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample data: support tickets and a tiny policy book
# MAGIC **What:** 4 customer messages and 4 policy sentences.
# MAGIC **Why:** a realistic, small dataset so you can read every output yourself.

# COMMAND ----------

ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"     # the chat model we will use everywhere in this notebook

tickets = [
    "My order has not arrived. It is 5 days late.",
    "I was charged twice for the same order.",
    "The headphones stopped working after 2 days.",
    "Can I return a laptop I bought 20 days ago?",
]
policies = {
    "Shipping": "Orders delayed more than 3 days qualify for a shipping investigation and a $10 credit.",
    "Payment": "Duplicate payments are refunded within 3 business days once confirmed.",
    "Product Defect": "Defective products are replaced free within 15 days of delivery.",
    "Return": "Most products can be returned within 30 days of delivery.",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1 – Plain SDK (no framework)
# MAGIC **What:** build the prompt with an f-string, call the endpoint, and pull the text out of the JSON response by hand.
# MAGIC **Why look at this first:** it shows exactly what every framework does underneath. For a *single* call, this is
# MAGIC often the best choice – nothing extra to install or learn.
# MAGIC **Notice:** we must parse the JSON ourselves, and if we wanted retries, tools or memory we'd write those too.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def classify_plain(message):
    prompt = (f"Classify this support ticket. Return ONLY JSON like "
              f'{{"category": "Shipping|Payment|Product Defect|Return|Other", "priority": "High|Medium|Low"}}\n\n'
              f"Ticket: {message}")
    response = client.predict(
        endpoint=ENDPOINT,
        inputs={"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 100},
    )
    text = response["choices"][0]["message"]["content"]
    return json.loads(text[text.find("{"): text.rfind("}") + 1])      # cut out the {...} part and parse it

for t in tickets:
    print(t, "->", classify_plain(t))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2 – LangChain chain: `prompt | llm | parser`
# MAGIC **What:** three reusable building blocks joined with `|` ("pipe" = send the output of one into the next):
# MAGIC - `ChatPromptTemplate` – a prompt with `{placeholders}` filled in at run time.
# MAGIC - `ChatDatabricks` – the Databricks model, wrapped so LangChain can use it.
# MAGIC - `JsonOutputParser` – turns the model's text into a Python dictionary (handles ```json fences for you).
# MAGIC
# MAGIC **Why:** the same chain can be `.invoke()`d on one input, `.batch()`ed over many (in parallel), or `.stream()`ed,
# MAGIC with no extra code. Swapping the model = changing one line.

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

llm = ChatDatabricks(endpoint=ENDPOINT, temperature=0)

classify_prompt = ChatPromptTemplate.from_template(
    "Classify this support ticket. Return ONLY JSON with keys category "
    "(Shipping, Payment, Product Defect, Return, Other) and priority (High, Medium, Low).\n\nTicket: {ticket}")

classify_chain = classify_prompt | llm | JsonOutputParser()

results = classify_chain.batch([{"ticket": t} for t in tickets])     # all 4 tickets, run in parallel
for t, r in zip(tickets, results):
    print(t, "->", r)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 3 – Where a framework really helps: retriever + tool in one chain
# MAGIC **What:** a 3-step chain: (1) classify the ticket, (2) **retrieve** the matching policy, (3) write a reply that
# MAGIC uses the policy. `RunnablePassthrough.assign` adds new fields to the data flowing through the chain.
# MAGIC
# MAGIC **Why:** this is where hand-written code starts to get messy (passing data between steps, error handling,
# MAGIC tracing). LangChain keeps each step small and swappable – e.g. replace `lookup_policy` with a real
# MAGIC Vector Search retriever later without touching the rest.

# COMMAND ----------

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def lookup_policy(data):
    """A stand-in 'retriever': find the policy for the predicted category."""
    return policies.get(data["classification"]["category"], "No specific policy found.")

reply_prompt = ChatPromptTemplate.from_template(
    "You are a polite support agent. Using ONLY this policy, reply to the customer in 2 sentences.\n"
    "Policy: {policy}\nCustomer: {ticket}")

support_chain = (
    RunnablePassthrough.assign(classification=classify_chain)          # step 1: add "classification"
    | RunnablePassthrough.assign(policy=RunnableLambda(lookup_policy))  # step 2: add "policy"
    | RunnablePassthrough.assign(reply=reply_prompt | llm | StrOutputParser())  # step 3: add "reply"
)

out = support_chain.invoke({"ticket": tickets[0]})
print(json.dumps(out, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 4 – No Python at all: SQL AI Functions for batch work
# MAGIC **What:** `ai_query` calls the same model for every row of a table, in parallel, from SQL.
# MAGIC **Why:** when the job is "run one prompt over a million rows", SQL is simpler and scales automatically.

# COMMAND ----------

spark.createDataFrame([(t,) for t in tickets], "ticket string").createOrReplaceTempView("tickets")
display(spark.sql(f"""
SELECT ticket,
       ai_query('{ENDPOINT}', concat('Classify into one word (Shipping, Payment, Product Defect, Return, Other): ', ticket)) AS category
FROM tickets
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to choose – a decision guide
# MAGIC | Your situation | Choose |
# MAGIC |---|---|
# MAGIC | One prompt, one answer, inside a Python app | Plain SDK |
# MAGIC | Same prompt over a big table | SQL AI Functions (`ai_query`) |
# MAGIC | Several steps, retrieval (RAG), parsing, tools | LangChain |
# MAGIC | Agent that loops, branches, keeps state, or several agents | LangGraph (or Agent Framework with any library) |
# MAGIC | Mostly about loading and indexing many documents | LlamaIndex |
# MAGIC | You have examples and want the prompt tuned automatically | DSPy |
# MAGIC | Production: tracing, evaluation, deployment, monitoring | MLflow + Mosaic AI Agent Framework (works with all of the above) |
# MAGIC
# MAGIC **Exam takeaways:** frameworks add value when you need **composition** (chains), **retrieval**, **tools/agents**,
# MAGIC or **integrations**; for a single call, plain SDK is fine. On Databricks, `databricks-langchain` provides
# MAGIC `ChatDatabricks`, `DatabricksEmbeddings`, `DatabricksVectorSearch` and Genie integrations.
