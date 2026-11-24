# Databricks notebook source
# MAGIC %md
# MAGIC # 02 – Select model tasks to accomplish a business requirement
# MAGIC
# MAGIC | Business requirement | Model task | Example |
# MAGIC |---|---|---|
# MAGIC | Identify complaint type | Classification | Shipping or Payment |
# MAGIC | Identify customer emotion | Sentiment analysis | Positive / Negative |
# MAGIC | Extract order info | Information extraction | Order ID, product, amount |
# MAGIC | Shorten complaint | Summarization | One-line summary |
# MAGIC | Translate complaint | Translation | Malayalam → English |
# MAGIC | Generate response | Text generation | Write an apology |
# MAGIC | Answer policy questions | RAG / QA | What is the return policy? |
# MAGIC | Find similar tickets | Embeddings / semantic search | Similar complaints |
# MAGIC
# MAGIC **Distinction:** classification assigns a label; extraction identifies fields; summarization reduces text;
# MAGIC generation creates text; embeddings represent meaning for similarity/retrieval.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### How a model call works (plain code, reused in every cell below)
# MAGIC Every model call is the same 3 steps: build a chat `messages` list, send it to the model serving endpoint
# MAGIC with `client.predict`, and pull the reply text out of `response["choices"][0]["message"]["content"]`.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")          # authenticated client for this workspace

prompt = "Summarize in a maximum of 15 words: I purchased a laptop for $1200. The delivery is 5 days late. I am very disappointed."

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

# 4. Extract just the text
answer = response["choices"][0]["message"]["content"].strip()
print("\nAnswer:", answer)

# COMMAND ----------

message = ("I purchased a laptop for $1200. The delivery is 5 days late. I am very disappointed. "
           "Please tell me when I will receive it.")

# COMMAND ----------

# MAGIC %md ### A. Classification → expected `Shipping`

# COMMAND ----------

prompt = f"""Classify the customer complaint. Return one category only, no other text:
Shipping, Return, Payment, Product Defect, Other.

Complaint: {message}"""

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
print(answer)

# COMMAND ----------

# MAGIC %md ### B. Sentiment analysis

# COMMAND ----------

prompt = f"What is the sentiment of this message? Answer with one word: Positive, Neutral, or Negative.\n\n{message}"

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
print(answer)

# COMMAND ----------

# MAGIC %md ### C. Summarization (max 15 words)

# COMMAND ----------

prompt = f"Summarize the complaint in a maximum of 15 words.\n\n{message}"

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
print(answer)

# COMMAND ----------

# MAGIC %md ### D. Information extraction (JSON)

# COMMAND ----------

prompt = f"""Extract these fields from the complaint and return JSON only:
product (string), amount (number), delivery_delay_days (number), customer_sentiment (Positive/Neutral/Negative).

Complaint: {message}"""

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
# the model may wrap JSON in ``` fences, so keep only the {...} part before json.loads
extracted = json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
extracted

# COMMAND ----------

# MAGIC %md ### E. Translation and F. Text generation

# COMMAND ----------

# Translation
response = client.predict(
    endpoint=ENDPOINT,
    inputs={
        "messages": [
            {"role": "user", "content": f"Translate this customer message into Malayalam:\n\n{message}"},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    },
)
answer = response["choices"][0]["message"]["content"].strip()
print(answer)
print("---")

# Text generation (temperature 0.3 = a little more varied wording)
response = client.predict(
    endpoint=ENDPOINT,
    inputs={
        "messages": [
            {"role": "user", "content": f"Write a short, polite apology to this customer. Do not promise a delivery date.\n\n{message}"},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    },
)
answer = response["choices"][0]["message"]["content"].strip()
print(answer)

# COMMAND ----------

# MAGIC %md ### G. Embeddings → find similar tickets (semantic search)

# COMMAND ----------

import numpy as np

EMBED_ENDPOINT = "databricks-gte-large-en"
tickets = spark.table("customer_support_tickets").select("ticket_id", "customer_message").collect()
texts = [message] + [t.customer_message for t in tickets]
emb = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": texts})
vecs = np.array([d["embedding"] for d in emb["data"]])
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
scores = vecs[1:] @ vecs[0]

display(spark.createDataFrame(
    [(t.ticket_id, t.customer_message, float(s)) for t, s in sorted(zip(tickets, scores), key=lambda x: -x[1])],
    "ticket_id int, customer_message string, similarity double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Same tasks as SQL AI Functions (batch over the whole table)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticket_id,
# MAGIC        customer_message,
# MAGIC        ai_classify(customer_message, ARRAY('Shipping','Return','Product Defect','Payment','Wrong Item','Positive Feedback','Other')) AS category,
# MAGIC        ai_analyze_sentiment(customer_message) AS sentiment,
# MAGIC        ai_summarize(customer_message, 10)   AS summary
# MAGIC FROM customer_support_tickets
