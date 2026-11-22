# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.4 – Augment a prompt with context from the user's input (key fields, terms, intents)
# MAGIC
# MAGIC **The big idea for juniors:** users write messy, short messages like *"where's my stuff?? order 1002"*.
# MAGIC If we send that straight to the model, it knows nothing about order 1002 and will guess. Instead we:
# MAGIC
# MAGIC 1. **Extract** key information from the message – *fields* (order ID, product), *terms* (keywords like "refund"),
# MAGIC    and the *intent* (what the user wants: track order, return, complain…).
# MAGIC 2. **Look up** facts using those fields (the order's real status from a table, the right policy for the intent).
# MAGIC 3. **Augment** (enrich) the prompt with those facts, so the model answers from real data.
# MAGIC
# MAGIC ```
# MAGIC "where's my stuff?? order 1002"
# MAGIC         │  extract
# MAGIC         ▼
# MAGIC {intent: track_order, order_id: 1002, sentiment: negative}
# MAGIC         │  look up
# MAGIC         ▼
# MAGIC order 1002 = Shipped, 5 days late, Headphones   +   shipping-delay policy
# MAGIC         │  augment
# MAGIC         ▼
# MAGIC Prompt = instructions + order facts + policy + original message   ──►  accurate, specific answer
# MAGIC ```
# MAGIC
# MAGIC **Analogy:** a call-centre agent who looks up your account on screen before answering, instead of guessing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample data: orders table, policies by intent, and incoming messages
# MAGIC **Why a table:** in real apps, facts live in databases. Here we create a small Delta-style DataFrame.

# COMMAND ----------

orders = spark.createDataFrame([
    (1001, "Laptop Pro 15", 1200, "Delivered", "2026-11-02", 0),
    (1002, "SoundMax Headphones", 150, "Shipped", "2026-11-10", 5),
    (1003, "Mechanical Keyboard", 75, "Processing", "2026-11-18", 0),
], "order_id int, product string, amount int, status string, order_date string, days_late int")
orders.createOrReplaceTempView("orders")
display(orders)

policies_by_intent = {
    "track_order": "Orders delayed more than 3 days qualify for an investigation and a $10 store credit.",
    "return_item": "Most products can be returned within 30 days of delivery. Opened software cannot be returned.",
    "report_defect": "Defective products are replaced free within 15 days of delivery; after that the manufacturer warranty applies.",
    "billing_issue": "Duplicate payments are refunded within 3 business days once confirmed.",
    "other": "Answer politely and offer to connect the customer with a human agent.",
}

messages = [
    "where's my stuff?? order 1002",
    "the laptop from order 1001 screen is flickering, what now",
    "i want my money back for #1003",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Baseline: send the raw message with no context
# MAGIC **What:** a plain call with just the message.
# MAGIC **Why:** to see the problem – the model doesn't know the order, so it gives generic or invented answers.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

def chat(system, user, max_tokens=200):
    response = client.predict(
        endpoint=ENDPOINT,
        inputs={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": max_tokens},
    )
    return response["choices"][0]["message"]["content"].strip()

baseline_answer = chat("You are a customer support assistant.", messages[0])
print("BASELINE (no context):\n", baseline_answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Extract key fields, terms and intent from the message
# MAGIC **What:** ask the model to return a small JSON object with fixed keys and allowed values.
# MAGIC **Why JSON with fixed choices:** our code needs predictable values (`track_order`, not "the user wants tracking")
# MAGIC to look things up. A regular expression also grabs the order number as a safety net.

# COMMAND ----------

import re

EXTRACT_INSTRUCTIONS = """Extract information from a customer message. Return ONLY JSON with these keys:
- intent: one of track_order, return_item, report_defect, billing_issue, other
- order_id: the order number as an integer, or null
- product_terms: list of product words mentioned (may be empty)
- key_terms: list of important words such as refund, late, broken, charged twice
- sentiment: positive, neutral or negative"""

def extract(message):
    text = chat(EXTRACT_INSTRUCTIONS, message, max_tokens=150)
    info = json.loads(text[text.find("{"): text.rfind("}") + 1])
    if not info.get("order_id"):                                # safety net: find a 4-digit number like 1003 or #1003
        found = re.search(r"\b(\d{4})\b", message)
        info["order_id"] = int(found.group(1)) if found else None
    return info

for m in messages:
    print(m, "->", extract(m))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Look up facts using the extracted fields
# MAGIC **What:** use `order_id` to fetch the real order row, and `intent` to pick the right policy.
# MAGIC **Why:** this is the "augmentation" data. It is *facts from our systems*, not the model's guesses.

# COMMAND ----------

def lookup_context(info):
    order = None
    if info.get("order_id"):
        row = spark.sql(f"SELECT * FROM orders WHERE order_id = {int(info['order_id'])}").first()
        order = row.asDict() if row else {"error": f"order {info['order_id']} not found"}
    policy = policies_by_intent.get(info.get("intent"), policies_by_intent["other"])
    return order, policy

info = extract(messages[0])
print(lookup_context(info))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Build the augmented prompt
# MAGIC **What:** a template with clearly labelled sections: instructions, extracted info, order facts, policy, message.
# MAGIC **Why labelled sections:** the model can see what each piece is, and the instruction "use only the facts below"
# MAGIC keeps it from inventing things. Matching tone to *sentiment* (apologise when negative) personalises the reply.

# COMMAND ----------

AUGMENTED_TEMPLATE = """You are a ShopEase support assistant.
Use ONLY the facts below. If something is not in the facts, say you will check with a colleague.
{tone_rule}
Reply in at most 3 sentences.

### Detected request
Intent: {intent}
Key terms: {key_terms}

### Order facts (from our order system)
{order}

### Relevant policy
{policy}

### Customer message
{message}"""

def answer_with_context(message):
    info = extract(message)
    order, policy = lookup_context(info)
    tone_rule = ("The customer is upset: start with a short, sincere apology."
                 if info.get("sentiment") == "negative" else "Use a friendly tone.")
    prompt = AUGMENTED_TEMPLATE.format(tone_rule=tone_rule, intent=info.get("intent"),
                                       key_terms=", ".join(info.get("key_terms", [])),
                                       order=json.dumps(order, default=str) if order else "No order number given.",
                                       policy=policy, message=message)
    return prompt, chat("You follow the instructions in the user message exactly.", prompt)

prompt, answer = answer_with_context(messages[0])
print("AUGMENTED PROMPT SENT TO THE MODEL:\n" + prompt)
print("\nANSWER:\n" + answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Compare baseline vs augmented for every message
# MAGIC **How to read:** the augmented answers should mention the *real* product, status and delay, and the right policy
# MAGIC (e.g. "$10 store credit" for order 1002). Baseline answers stay generic ("please provide your order number").

# COMMAND ----------

rows = []
for m in messages:
    base = chat("You are a customer support assistant.", m)
    _, aug = answer_with_context(m)
    rows.append((m, base, aug))
display(spark.createDataFrame(rows, "message string, baseline_answer string, augmented_answer string"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Augmentation = **extract** (fields, terms, intent) → **retrieve/look up** (tables, documents) → **insert** into a
# MAGIC   structured prompt template.
# MAGIC - Use constrained JSON for extraction so code can rely on the values; add simple rules (regex) as a safety net.
# MAGIC - Intent decides *which* knowledge to fetch (route to the right policy, tool or index).
# MAGIC - Tell the model to use **only** the provided facts to reduce hallucinations.
