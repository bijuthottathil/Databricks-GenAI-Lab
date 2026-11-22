# Databricks notebook source
# MAGIC %md
# MAGIC # 01 – Design a prompt that elicits a specifically formatted response
# MAGIC **Requirement:** classify every ticket into a predefined category and return the result as JSON.
# MAGIC
# MAGIC | Technique | Purpose |
# MAGIC |---|---|
# MAGIC | Role prompting | Defines the AI's role |
# MAGIC | Instruction prompting | Tells the model what to do |
# MAGIC | Constrained output | Restricts possible categories |
# MAGIC | JSON formatting | Ensures structured output |
# MAGIC | Few-shot prompting | Examples improve consistency |
# MAGIC
# MAGIC You can also paste the prompt below into **AI/ML → Playground** with an instruction-following model.
# MAGIC
# MAGIC > Exam tip: for JSON / predefined labels / a schema, use explicit output instructions, not a vague "Analyze this ticket."

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

prompt = "Classify this message into one category (Shipping, Return, Payment, Product Defect, Other): My order has not arrived. It is 5 days late."

# 1. Build the chat messages (system = role/rules, user = the actual request)
messages = [
    {"role": "system", "content": "You are an e-commerce customer support classifier."},
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

# MAGIC %md
# MAGIC ### Vague prompt (anti-pattern) – free text, unpredictable format

# COMMAND ----------

response = client.predict(
    endpoint=ENDPOINT,
    inputs={
        "messages": [
            {"role": "user", "content": "Analyze this ticket: My order has not arrived. It is 5 days late."},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    },
)
answer = response["choices"][0]["message"]["content"].strip()
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Structured prompt – role + instructions + constrained labels + JSON schema

# COMMAND ----------

PROMPT_TEMPLATE = """You are an e-commerce customer support classifier.
Analyze the customer message.
Return ONLY valid JSON.
Use the following rules:
1. category must be one of: Shipping, Return, Product Defect, Payment, Wrong Item, Positive Feedback, Other.
2. sentiment must be: Positive, Neutral, or Negative.
3. priority must be: High, Medium, or Low.
4. Do not add explanations outside JSON.

Required output format:
{{"category": "", "sentiment": "", "priority": "", "reason": "", "recommended_action": ""}}

Customer message:
{message}"""

response = client.predict(
    endpoint=ENDPOINT,
    inputs={
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(message="My order has not arrived. It is 5 days late.")},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    },
)
answer = response["choices"][0]["message"]["content"].strip()
print(answer)

# the model may wrap JSON in ``` fences, so keep only the {...} part before json.loads
result = json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
result

# COMMAND ----------

# MAGIC %md
# MAGIC ### Few-shot variant – add examples to improve consistency

# COMMAND ----------

FEW_SHOT = """Examples:
Customer message: I was charged twice for the same order.
{{"category": "Payment", "sentiment": "Negative", "priority": "High", "reason": "Duplicate charge.", "recommended_action": "Investigate and refund duplicate payment."}}

Customer message: Excellent service. My order arrived early.
{{"category": "Positive Feedback", "sentiment": "Positive", "priority": "Low", "reason": "Customer is happy with early delivery.", "recommended_action": "Thank the customer."}}

"""
FEW_SHOT_TEMPLATE = PROMPT_TEMPLATE.replace("Customer message:\n{message}", FEW_SHOT + "Customer message:\n{message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run against every ticket and validate the format

# COMMAND ----------

ALLOWED = {
    "category": {"Shipping", "Return", "Product Defect", "Payment", "Wrong Item", "Positive Feedback", "Other"},
    "sentiment": {"Positive", "Neutral", "Negative"},
    "priority": {"High", "Medium", "Low"},
}

rows = []
for t in spark.table("customer_support_tickets").collect():
    response = client.predict(
        endpoint=ENDPOINT,
        inputs={
            "messages": [
                {"role": "user", "content": FEW_SHOT_TEMPLATE.format(message=t.customer_message)},
            ],
            "temperature": 0.0,
            "max_tokens": 500,
        },
    )
    answer = response["choices"][0]["message"]["content"].strip()
    # the model may wrap JSON in ``` fences, so keep only the {...} part before json.loads
    out = json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
    valid = all(out.get(k) in v for k, v in ALLOWED.items())
    rows.append((t.ticket_id, t.customer_message, t.expected_category, out.get("category"),
                 out.get("sentiment"), out.get("priority"), out.get("recommended_action"), valid))

results = spark.createDataFrame(rows, "ticket_id int, customer_message string, expected_category string, category string, sentiment string, priority string, recommended_action string, format_valid boolean")
display(results)

# COMMAND ----------

accuracy = results.filter("category = expected_category").count() / results.count()
print(f"Category accuracy: {accuracy:.0%}  |  all outputs valid JSON with allowed labels: {results.filter('NOT format_valid').count() == 0}")
