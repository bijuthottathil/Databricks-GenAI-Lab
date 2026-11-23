# Databricks notebook source
# MAGIC %md
# MAGIC # Section 5.1 – Use masking techniques as guardrails to meet a performance objective
# MAGIC
# MAGIC **The big idea for juniors:** customers put personal data (names, emails, phone numbers, card numbers) into
# MAGIC support messages. A **masking guardrail** replaces that data *before* the text reaches the LLM, logs or a vector
# MAGIC index. But masking has a cost: mask too much and the model can no longer understand the message; mask with a slow
# MAGIC method and the app becomes slow.
# MAGIC
# MAGIC So we pick the masking technique that meets a **performance objective** – a clear, measurable target:
# MAGIC
# MAGIC | Objective | Target |
# MAGIC |---|---|
# MAGIC | **Privacy** | 0 tickets with raw PII sent to the LLM |
# MAGIC | **Quality** | classification accuracy ≥ 87.5 % (7 of 8 tickets) – same as without masking |
# MAGIC | **Speed** | masking adds < 50 ms per request |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **PII** | Personally Identifiable Information – data that points to a real person |
# MAGIC | **Masking** | Replacing sensitive values, e.g. `maria@x.com` → `<EMAIL>` |
# MAGIC | **Typed placeholder** | A mask that keeps the *kind* of value (`<PHONE>`), so the model still understands the sentence |
# MAGIC | **Redaction / sentence drop** | Removing text completely – safe, but destroys meaning |
# MAGIC | **Pseudonymisation** | Reversible masking: `Maria Lopez` → `<PERSON_1>`, with a private map to put the name back later |
# MAGIC | **`ai_mask()`** | Databricks SQL AI function that masks chosen entity types with an LLM |
# MAGIC | **Column mask** | A Unity Catalog rule that hides a column's value unless the reader is allowed to see it |
# MAGIC | **Guardrail** | An automatic check or transformation around the LLM that keeps the app safe |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 8 tickets (text + true category + the PII hidden in each)
# MAGIC Step 2 three masking functions: none / sentence drop / typed placeholders
# MAGIC Step 3 classify every masked ticket with the LLM   → accuracy
# MAGIC Step 4 measure leaks + masking time                → table "does it meet the objective?"
# MAGIC Step 5 reversible pseudonymisation (mask → LLM reply → unmask)
# MAGIC Step 6 ai_mask() in SQL (LLM-based, higher recall, slower)
# MAGIC Step 7 Unity Catalog column mask (masking at the data layer)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample tickets
# MAGIC **What:** each row has the ticket text, the **true category** (our answer key) and a list of the PII values hidden
# MAGIC in it. The PII list is only used to **check** for leaks – it is never given to the masking functions.
# MAGIC
# MAGIC Notice that most tickets are **one sentence** where the PII and the problem are mixed together. That's realistic,
# MAGIC and it's what makes "just delete the sensitive sentence" a bad idea.

# COMMAND ----------

import re, time, json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
CATEGORIES = ["Shipping", "Payment", "Product Defect", "Return", "Account"]
CATALOG, SCHEMA = "workspace", "fde_governance"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

tickets = [
    # id, text, true category, PII values hidden in the text (for leak checking only)
    (1, "Hi, this is Maria Lopez, my parcel for order 5521 never arrived, call me on +1 415 555 0134.",
        "Shipping", ["Maria Lopez", "+1 415 555 0134"]),
    (2, "Card 4111 1111 1111 1111 was charged twice for one order, email me at maria.lopez@example.com.",
        "Payment", ["4111 1111 1111 1111", "maria.lopez@example.com"]),
    (3, "My login stopped working, my email is tom.baker@example.org and the password reset never arrives.",
        "Account", ["tom.baker@example.org"]),
    (4, "I'm Tom Baker and the blender I received from you is broken, the motor smokes.",
        "Product Defect", ["Tom Baker"]),
    (5, "I want to return the shoes and get my money back, my phone is 020 7946 0958.",
        "Return", ["020 7946 0958"]),
    (6, "You charged 49.99 again after I cancelled the subscription, contact anna.k@example.net.",
        "Payment", ["anna.k@example.net"]),
    (7, "Anna Kowalski here, tracking shows delivered but nothing came to 12 Oak Street.",
        "Shipping", ["Anna Kowalski", "12 Oak Street"]),
    (8, "The headphones from order 7781 stopped charging after a week, reach me at 555-201-7788.",
        "Product Defect", ["555-201-7788"]),
]
for t in tickets:
    print(t[0], t[2].ljust(15), t[1])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Three masking techniques
# MAGIC | Technique | What it does | Expected result |
# MAGIC |---|---|---|
# MAGIC | `no_mask` | sends the text unchanged | best accuracy, but **PII leaks** – fails privacy |
# MAGIC | `sentence_drop` | deletes every sentence that contains PII | no leaks, but the **problem description disappears too** |
# MAGIC | `typed_placeholders` | replaces only the PII with `<EMAIL>`, `<PHONE>`, `<CARD>`, `<ADDRESS>`, `<PERSON>` | no leaks **and** the sentence still makes sense |
# MAGIC
# MAGIC **Code walk-through – `DETECTORS`:** a list of `(label, regex)` pairs, applied in order.
# MAGIC - Email first, then **card before phone** (a 16-digit card number would otherwise look like a phone number).
# MAGIC - The person-name rules only catch self-introductions ("this is X Y", "I'm X Y", "X Y here"). Names are the hardest
# MAGIC   PII for regex – that's where an LLM-based masker like `ai_mask` (Step 6) helps.
# MAGIC - Order numbers like `5521` are **not** PII here and are kept, because the model and the agent need them.

# COMMAND ----------

DETECTORS = [
    ("EMAIL",   r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("CARD",    r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    ("PHONE",   r"\+?\d[\d\s\-]{7,}\d"),
    ("ADDRESS", r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane)\b"),
    ("PERSON",  r"(?<=this is )[A-Z][a-z]+ [A-Z][a-z]+|(?<=I'm )[A-Z][a-z]+ [A-Z][a-z]+|^[A-Z][a-z]+ [A-Z][a-z]+(?= here)"),
]

def no_mask(text):
    return text

def typed_placeholders(text):
    for label, pattern in DETECTORS:
        text = re.sub(pattern, f"<{label}>", text)
    return text

def sentence_drop(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not any(re.search(p, s) for _, p in DETECTORS)]
    return " ".join(kept)

TECHNIQUES = {"no_mask": no_mask, "sentence_drop": sentence_drop, "typed_placeholders": typed_placeholders}

for name, fn in TECHNIQUES.items():
    print(f"--- {name}\n   {fn(tickets[0][1])!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Classify every masked ticket with the LLM
# MAGIC **What:** for each technique, mask all 8 tickets, send each one to the LLM, and compare the answer with the true
# MAGIC category. We also time **only the masking function** (not the LLM) – that's the overhead the guardrail adds.
# MAGIC
# MAGIC **Code walk-through:**
# MAGIC - `classify()` asks for the category only, with `temperature 0.0` so repeat runs give the same answer.
# MAGIC - An empty input (everything was dropped) is sent as `"(empty message)"` – the model can only guess.
# MAGIC - `time.perf_counter()` measures how long masking took, in milliseconds.

# COMMAND ----------

def classify(text):
    r = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [
            {"role": "system", "content": f"Classify the support ticket into exactly one of {CATEGORIES}. "
                                          "Placeholders like <EMAIL> are masked personal data. Reply with the category only."},
            {"role": "user", "content": text or "(empty message)"},
        ],
        "temperature": 0.0, "max_tokens": 10})
    answer = r["choices"][0]["message"]["content"].strip().strip(".")
    return answer if answer in CATEGORIES else "Other"

results = []
for name, fn in TECHNIQUES.items():
    for tid, text, label, pii in tickets:
        start = time.perf_counter()
        masked = fn(text)
        mask_ms = (time.perf_counter() - start) * 1000
        predicted = classify(masked)
        leaked = any(value in masked for value in pii)
        results.append((name, tid, masked, label, predicted, predicted == label, leaked, mask_ms))

display(spark.createDataFrame(results, "technique string, ticket_id int, text_sent_to_llm string, true_label string, "
                                       "predicted string, correct boolean, pii_leaked boolean, mask_ms double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Does each technique meet the objective?
# MAGIC **How to read the output (typical result):**
# MAGIC | technique | accuracy | leaked tickets | meets objective? |
# MAGIC |---|---|---|---|
# MAGIC | `no_mask` | ~100 % | 8 | ❌ privacy |
# MAGIC | `sentence_drop` | low (most tickets become empty) | 0 | ❌ quality |
# MAGIC | `typed_placeholders` | ~100 % | 0 | ✅ |
# MAGIC
# MAGIC Masking time for regex is well under 1 ms – far below the 50 ms target. **Lesson:** the right guardrail removes the
# MAGIC sensitive *values* but keeps the *meaning*. Measuring all three targets together is how you prove it.

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(results, columns=["technique", "ticket_id", "text", "label", "pred", "correct", "leaked", "mask_ms"])
summary = df.groupby("technique").agg(accuracy=("correct", "mean"), leaked_tickets=("leaked", "sum"),
                                      avg_mask_ms=("mask_ms", "mean")).reset_index()
summary["meets_privacy"] = summary.leaked_tickets == 0
summary["meets_quality"] = summary.accuracy >= 0.875
summary["meets_speed"] = summary.avg_mask_ms < 50
summary["meets_objective"] = summary.meets_privacy & summary.meets_quality & summary.meets_speed
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Reversible masking (pseudonymisation)
# MAGIC **Problem:** sometimes the *reply* needs the real value, e.g. "Dear Maria Lopez, ...". But the LLM must not see it.
# MAGIC
# MAGIC **Solution:** replace each value with a **numbered token** (`<PERSON_1>`, `<PHONE_1>`) and keep a private map
# MAGIC `token → real value` in your app (never sent to the model). The LLM writes a reply using the tokens, and the app puts
# MAGIC the real values back afterwards.
# MAGIC ```
# MAGIC "this is Maria Lopez"  ─mask─►  "this is <PERSON_1>"  ─LLM─►  "Dear <PERSON_1>, ..."  ─unmask─►  "Dear Maria Lopez, ..."
# MAGIC                                  map: {<PERSON_1>: Maria Lopez}  (stays in the app)
# MAGIC ```
# MAGIC **How to read the output:** the text sent to the LLM has tokens only; the final reply has the real name again.

# COMMAND ----------

def pseudonymise(text):
    mapping, counters = {}, {}
    for label, pattern in DETECTORS:
        def replace(match):
            value = match.group(0)
            for token, original in mapping.items():          # same value → same token
                if original == value:
                    return token
            counters[label] = counters.get(label, 0) + 1
            token = f"<{label}_{counters[label]}>"
            mapping[token] = value
            return token
        text = re.sub(pattern, replace, text)
    return text, mapping

def unmask(text, mapping):
    for token, value in mapping.items():
        text = text.replace(token, value)
    return text

masked, mapping = pseudonymise(tickets[0][1])
print("Sent to LLM :", masked)
print("Private map :", mapping)

r = client.predict(endpoint=ENDPOINT, inputs={
    "messages": [
        {"role": "system", "content": "You are a support agent. Write a 2-sentence reply. Tokens like <PERSON_1> are "
                                      "placeholders: copy them exactly as written, never invent real values."},
        {"role": "user", "content": masked},
    ],
    "temperature": 0.0, "max_tokens": 120})
llm_reply = r["choices"][0]["message"]["content"]
print("\nLLM reply   :", llm_reply)
print("\nFinal reply :", unmask(llm_reply, mapping))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – `ai_mask()`: LLM-based masking in SQL
# MAGIC **What:** `ai_mask(text, array('person', 'email', ...))` masks the entity types you list, using a model.
# MAGIC **Trade-off vs. regex:** it finds PII that regex misses (names anywhere in a sentence, unusual formats), but each row
# MAGIC is a model call – much slower than regex. Good for **batch** cleaning of tables (before indexing, before sharing);
# MAGIC usually too slow as a per-request guardrail with a tight latency target.
# MAGIC
# MAGIC **How to read the output:** masked text per ticket and the total time. Compare with Step 4's sub-millisecond regex.
# MAGIC If AI Functions aren't enabled in your workspace, the cell prints a message instead.

# COMMAND ----------

spark.createDataFrame([(t[0], t[1]) for t in tickets], "ticket_id int, text string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.pii_tickets")

try:
    start = time.perf_counter()
    masked_df = spark.sql(f"""
        SELECT ticket_id, ai_mask(text, array('person', 'email', 'phone', 'address', 'credit card')) AS masked_text
        FROM {CATALOG}.{SCHEMA}.pii_tickets""").toPandas()
    elapsed = time.perf_counter() - start
    display(masked_df)
    print(f"ai_mask on {len(masked_df)} rows took {elapsed:.1f} s  (≈ {elapsed / len(masked_df) * 1000:.0f} ms per row)")
except Exception as e:
    print("ai_mask not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Masking at the data layer: Unity Catalog column masks
# MAGIC **What:** a **column mask** is a SQL function attached to a column. Every query – notebook, dashboard, RAG
# MAGIC pipeline – sees the masked value unless the reader passes the check (here: membership of group `support_admins`).
# MAGIC **Why:** guardrails in your app protect the LLM; column masks protect the **source table**, so a data pipeline that
# MAGIC feeds a vector index can't accidentally copy raw emails into it.
# MAGIC
# MAGIC **How to read the output:** you are (probably) not in `support_admins`, so you see `***@example.com`.

# COMMAND ----------

spark.createDataFrame([(1, "Maria Lopez", "maria.lopez@example.com"), (2, "Tom Baker", "tom.baker@example.org")],
                      "customer_id int, name string, email string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.customer_contacts")

try:
    spark.sql(f"""
        CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.mask_email(email STRING)
        RETURN CASE WHEN is_account_group_member('support_admins') THEN email
                    ELSE regexp_replace(email, '^[^@]+', '***') END""")
    spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.customer_contacts ALTER COLUMN email SET MASK {CATALOG}.{SCHEMA}.mask_email")
    display(spark.table(f"{CATALOG}.{SCHEMA}.customer_contacts"))
except Exception as e:
    print("Column masks not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Where each masking guardrail fits
# MAGIC | Layer | Technique | Speed | Use it for |
# MAGIC |---|---|---|---|
# MAGIC | Request path (real time) | regex / typed placeholders, pseudonymisation | < 1 ms | every user message before the LLM |
# MAGIC | Endpoint | **AI Gateway PII guardrail** (block or mask input/output) | low | central policy for all apps using an endpoint |
# MAGIC | Batch / ingestion | `ai_mask()`, NER models | seconds per batch | cleaning documents before chunking + indexing |
# MAGIC | Data layer | UC column masks, row filters | built into queries | source tables, logs, inference tables |
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - State the **performance objective** (privacy + quality + latency) and **measure** each technique against it.
# MAGIC - **Typed placeholders** keep meaning, so accuracy stays high; dropping text is safe but hurts quality.
# MAGIC - Use **reversible pseudonymisation** when the reply must contain the real value.
# MAGIC - Regex is fast for structured PII; LLM-based masking (`ai_mask`) has better recall but costs latency → batch.
# MAGIC - Combine with UC column masks and AI Gateway guardrails for defence in depth.
