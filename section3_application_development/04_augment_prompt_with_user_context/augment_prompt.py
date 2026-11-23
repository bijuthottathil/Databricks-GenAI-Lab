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
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning | Example from this notebook |
# MAGIC |---|---|---|
# MAGIC | **Prompt** | Everything we send to the model in one call | instructions + facts + the customer's message |
# MAGIC | **System message** | The "job description" for the model | `"You are a customer support assistant."` |
# MAGIC | **User message** | What the customer (or our code) says | `"where's my stuff?? order 1002"` |
# MAGIC | **Field** | A specific value we can use in code | `order_id = 1002` |
# MAGIC | **Term** | An important keyword | `late`, `refund`, `broken` |
# MAGIC | **Intent** | What the customer *wants to do* | `track_order`, `return_item` |
# MAGIC | **Context** | Facts we add so the model doesn't guess | the order row + the matching policy |
# MAGIC | **Augment** | Add that context to the prompt | Step 5 |
# MAGIC
# MAGIC ### Important: the model cannot see your tables
# MAGIC The LLM only knows **what is written inside the prompt** (plus general knowledge from training).
# MAGIC It cannot query the `orders` table by itself. If the facts aren't in the prompt, the model either asks for more
# MAGIC details or **invents** them. Augmentation is how we put the facts in front of it.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1  data: orders table, policies_by_intent, messages
# MAGIC   │
# MAGIC   ├─► Step 2  chat(message)                → baseline answer (no facts → generic / guessed)
# MAGIC   │
# MAGIC   └─► Step 3  extract(message)             → {intent, order_id, key_terms, sentiment}
# MAGIC         └─► Step 4  lookup_context(info)   → (order row, policy text)
# MAGIC               └─► Step 5  answer_with_context(message) → prompt with facts → accurate answer
# MAGIC                     └─► Step 6  baseline vs augmented for all 3 messages
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample data: orders table, policies by intent, and incoming messages
# MAGIC **What:** three pieces of sample data:
# MAGIC - `orders` – a small table standing in for the company's order system (3 orders).
# MAGIC - `policies_by_intent` – a Python dictionary: *intent name → the policy text that applies*. In a real app this
# MAGIC   might be a vector search index of policy documents.
# MAGIC - `messages` – three short, messy customer messages, like real chat messages.
# MAGIC
# MAGIC **Why a table:** in real apps, facts live in databases. We register it as a temporary view called `orders` so
# MAGIC Step 4 can query it with SQL.
# MAGIC
# MAGIC **Look at row 1002:** SoundMax Headphones, `Shipped`, `days_late = 5`. Remember it – the good answer later should
# MAGIC mention exactly these facts.

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
# MAGIC **What:** a plain call with just the message. The *only* things the model receives are:
# MAGIC - system message: `"You are a customer support assistant."`
# MAGIC - user message: `"where's my stuff?? order 1002"`
# MAGIC
# MAGIC **Why:** to see the problem first. The model has never seen our `orders` table, so it doesn't know order 1002 is
# MAGIC headphones that are 5 days late.
# MAGIC
# MAGIC **What you will typically see:** a polite but generic reply such as *"Can you confirm your full name, email and
# MAGIC order date?"*. Those questions are **not from our data** – the model makes them up because that's what support
# MAGIC agents usually ask. A weaker model might even invent a status ("your order arrives tomorrow"), which is a
# MAGIC hallucination. Either way, the customer doesn't get a real answer.
# MAGIC
# MAGIC **About `chat()`:** it's just the plain model call (`client.predict`) wrapped so we can reuse it in this notebook.
# MAGIC `temperature = 0.0` makes answers repeatable so baseline and augmented runs are a fair comparison.

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")                # talks to Databricks model serving endpoints
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"     # the chat model we use everywhere in this notebook

def chat(system, user, max_tokens=200):
    # system = instructions for the model, user = the text to respond to
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
# MAGIC
# MAGIC **Here the model is used as a *reader*, not an answerer.** It doesn't reply to the customer; it just turns messy
# MAGIC text into tidy fields. Example:
# MAGIC ```
# MAGIC "where's my stuff?? order 1002"
# MAGIC   → {"intent": "track_order", "order_id": 1002, "product_terms": [], "key_terms": ["late"], "sentiment": "negative"}
# MAGIC ```
# MAGIC | Extracted value | Used later for |
# MAGIC |---|---|
# MAGIC | `order_id` | Step 4 – find the right row in the `orders` table |
# MAGIC | `intent` | Step 4 – pick the right policy from `policies_by_intent` |
# MAGIC | `key_terms` | Step 5 – shown in the prompt so the model sees what matters |
# MAGIC | `sentiment` | Step 5 – choose the tone (apologise if negative) |
# MAGIC
# MAGIC **Code walk-through of `extract()`:**
# MAGIC 1. Send the message with `EXTRACT_INSTRUCTIONS` as the system message.
# MAGIC 2. Cut out the text between the first `{` and last `}` (models sometimes add words around the JSON) and parse it.
# MAGIC 3. If the model missed the order number, the regex `\b(\d{4})\b` looks for any 4-digit number (so `#1003` → `1003`).
# MAGIC
# MAGIC **How to read the output:** one line per message. Check that each intent makes sense: 1002 → `track_order`,
# MAGIC 1001 flickering screen → `report_defect`, 1003 money back → `return_item` or `billing_issue`.

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
# MAGIC
# MAGIC **No AI in this step** – it's ordinary code: a SQL query and a dictionary lookup. That's on purpose: facts should
# MAGIC come from reliable systems, and the model's job is only to *phrase* the answer.
# MAGIC
# MAGIC **Code walk-through of `lookup_context()`:**
# MAGIC - If we have an `order_id`, run `SELECT * FROM orders WHERE order_id = 1002` and turn the row into a dict.
# MAGIC   `int(...)` makes sure only a number goes into the SQL (protects against injected text).
# MAGIC - If no row is found, we return an `"error"` note instead of nothing, so the model can say "I couldn't find it".
# MAGIC - Pick the policy for the intent; unknown intents fall back to the `"other"` policy.
# MAGIC
# MAGIC **How to read the output:** for message 1 you should see the order-1002 row (Headphones, Shipped, 5 days late)
# MAGIC and the shipping-delay policy ($10 store credit).

# COMMAND ----------

def lookup_context(info):
    order = None
    if info.get("order_id"):
        # Fetch the real order row from the table created in Step 1
        row = spark.sql(f"SELECT * FROM orders WHERE order_id = {int(info['order_id'])}").first()
        order = row.asDict() if row else {"error": f"order {info['order_id']} not found"}
    # Choose the policy that matches what the customer wants; fall back to "other"
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
# MAGIC
# MAGIC **What each part of the template does:**
# MAGIC | Template part | Filled with | Purpose |
# MAGIC |---|---|---|
# MAGIC | "Use ONLY the facts below…" | fixed text | Stops the model guessing; gives it a safe fallback ("check with a colleague") |
# MAGIC | `{tone_rule}` | apology rule if sentiment is negative | Personalise the tone |
# MAGIC | "at most 3 sentences" | fixed text | Controls length/format |
# MAGIC | `{intent}`, `{key_terms}` | from Step 3 | Tells the model what the customer wants |
# MAGIC | `{order}` | order row from Step 4 (as JSON) | The real facts |
# MAGIC | `{policy}` | policy text from Step 4 | The rule to apply |
# MAGIC | `{message}` | the original message | So the reply still answers the customer's words |
# MAGIC
# MAGIC **`answer_with_context()` runs the whole pipeline** for one message: extract (Step 3) → look up (Step 4) →
# MAGIC fill the template → call the model. It returns both the prompt (so you can inspect it) and the answer.
# MAGIC
# MAGIC **How to read the output:** first you'll see the full prompt – notice the real order facts are now *inside* it.
# MAGIC Then the answer, which should apologise, say the headphones are shipped and 5 days late, and mention the
# MAGIC investigation + $10 store credit. Compare that with the generic baseline from Step 2.

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
# MAGIC
# MAGIC **What to expect per message:**
# MAGIC | Message | Baseline (no context) | Augmented (with context) |
# MAGIC |---|---|---|
# MAGIC | order 1002 "where's my stuff" | asks for name/email/date | apology + shipped, 5 days late + $10 credit |
# MAGIC | order 1001 flickering screen | generic troubleshooting tips | Laptop Pro 15, delivered 2026-11-02 + defect policy (15-day replacement) |
# MAGIC | #1003 money back | asks why / generic refund steps | Keyboard still *Processing* + return or billing policy |
# MAGIC
# MAGIC **Note:** this cell calls the model 3 times per message (baseline, extract, augmented answer) = 9 calls in total.
# MAGIC Answers can differ slightly between runs or models – focus on whether the *facts* are right, not the exact wording.

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
