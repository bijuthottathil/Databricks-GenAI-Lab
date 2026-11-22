# Databricks notebook source
# MAGIC %md
# MAGIC # 06 – How and when to use Agent Bricks
# MAGIC
# MAGIC | Feature | Knowledge Assistant | Multiagent Supervisor | Information Extraction |
# MAGIC |---|---|---|---|
# MAGIC | Purpose | Answer questions | Coordinate agents | Extract structured data |
# MAGIC | Input | Docs + questions | Complex request | Documents |
# MAGIC | Output | Answer with citations | Consolidated response | Structured fields |
# MAGIC | Typical data | Policies, manuals, PDFs | Multiple business systems | Invoices, forms |
# MAGIC | Capability | RAG | Agent orchestration | Document extraction |
# MAGIC
# MAGIC **Keywords:** "answer questions from company documents" → Knowledge Assistant;
# MAGIC "multiple agents with specialized responsibilities" → Multiagent Supervisor;
# MAGIC "extract invoice number, date and amount" → Information Extraction.
# MAGIC
# MAGIC Each section shows (1) how to build it in the UI with Agent Bricks, and (2) a code version of the same pattern
# MAGIC you can run anywhere. Note: Databricks docs now mark Knowledge Assistant and Supervisor Agent as legacy and
# MAGIC recommend Genie Agents for new low-code agents; not every builder is available in every workspace.

# COMMAND ----------

# MAGIC %run ../00_setup/config

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plain-code equivalent of `ask_llm` (for understanding)
# MAGIC `ask_llm(prompt)` from `00_setup/config` is just this: build a chat `messages` list, send it to the
# MAGIC model serving endpoint, and pull the reply text out of the response. (`ask_llm` also retries on HTTP 429 rate limits.)

# COMMAND ----------

import json
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")          # authenticated client for this workspace

prompt = "Policy: Customers can return products within 30 days.\n\nQuestion: Can a customer return a laptop after 20 days?"

# 1. Build the chat messages (system = role/rules, user = the actual request)
messages = [
    {"role": "system", "content": "You are an e-commerce policy assistant. Answer only using the policy provided."},
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
print(ask_llm(prompt, system="You are an e-commerce policy assistant. Answer only using the policy provided."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Knowledge Assistant – document Q&A (RAG)
# MAGIC **UI:** Agents → Knowledge Assistant → create `Customer_Policy_Assistant` → source: the Volume
# MAGIC `/Volumes/<catalog>/fde_customer_support/policy_docs` (created by `00_setup/create_sample_data`) → instructions:
# MAGIC ```
# MAGIC You are an e-commerce policy assistant.
# MAGIC Answer questions only using the provided company policies.
# MAGIC Always provide supporting citations.
# MAGIC If the answer is unavailable, say that the policy does not specify it.
# MAGIC ```
# MAGIC Test: "Can a customer return a laptop after 20 days?" → Yes (30-day policy).

# COMMAND ----------

vol = f"/Volumes/{CATALOG}/{SCHEMA}/policy_docs"
docs = {f.name: dbutils.fs.head(f.path) for f in dbutils.fs.ls(vol)}

KA_INSTRUCTIONS = """You are an e-commerce policy assistant.
Answer questions only using the provided company policies.
Always provide supporting citations as [file name].
If the answer is unavailable, say that the policy does not specify it."""

def knowledge_assistant(question: str) -> str:
    context = "\n\n".join(f"[{name}]\n{text}" for name, text in docs.items())
    return ask_llm(f"Company policies:\n{context}\n\nQuestion: {question}", system=KA_INSTRUCTIONS)

for q in ["Can a customer return a laptop after 20 days?",
          "Can I get a replacement for defective headphones?",
          "Do you offer gift wrapping?"]:
    print(f"Q: {q}\nA: {knowledge_assistant(q)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Multiagent Supervisor – specialized agents + coordinator
# MAGIC Request: *"I purchased a laptop for $1200. It arrived five days late, and now it is not working. I want a refund."*
# MAGIC
# MAGIC | Agent | Responsibility |
# MAGIC |---|---|
# MAGIC | Shipping Agent | Check shipping delay and determine whether an investigation is required. |
# MAGIC | Product Agent | Determine whether the product issue qualifies for replacement. |
# MAGIC | Refund Agent | Check the return and refund policies. Do not issue a refund without authorization. |
# MAGIC | Supervisor | Delegate relevant tasks, collect results, resolve conflicting recommendations, one final response. |
# MAGIC
# MAGIC **UI:** Agents → Multi-Agent Supervisor → add each sub-agent (Knowledge Assistant, Genie space, UC function, or endpoint).
# MAGIC Don't use it for a task one LLM call or one agent can handle.

# COMMAND ----------

policy_text = "\n".join(f"- {r.policy_category}: {r.policy_text}" for r in spark.table("customer_support_policies").collect())

AGENTS = {
    "shipping_agent": "You are the Shipping Agent. Check shipping delay and determine whether an investigation is required.",
    "product_agent": "You are the Product Agent. Determine whether the product issue qualifies for replacement.",
    "refund_agent": "You are the Refund Agent. Check the return and refund policies. Do not issue a refund without authorization.",
}

def call_agent(name: str, request: str) -> str:
    return ask_llm(f"Policies:\n{policy_text}\n\nCustomer request: {request}\n\nGive your finding in 2 sentences.",
                   system=AGENTS[name])

def supervisor(request: str) -> str:
    # 1. Route: which specialists are relevant?
    routing = parse_json(ask_llm(
        f"Agents available: {list(AGENTS)}. Return JSON {{\"agents\": [...]}} listing every agent relevant to this request.\n\nRequest: {request}"))
    findings = {}
    for name in routing.get("agents", []):
        if name in AGENTS:
            findings[name] = call_agent(name, request)
            print(f"[{name}] {findings[name]}\n")
    # 2. Consolidate
    return ask_llm(
        f"Customer request: {request}\n\nSpecialist findings:\n{json.dumps(findings, indent=2)}\n\n"
        "As the Supervisor, resolve any conflicting recommendations and write ONE final response to the customer.",
        system="You are the Supervisor agent coordinating customer support specialists.")

import json
print("FINAL:\n" + supervisor("I purchased a laptop for $1200. It arrived five days late, and now it is not working. I want a refund."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Information Extraction – structured fields from documents
# MAGIC **UI:** Agents → Information Extraction → point at a Volume of invoices → define the schema below.

# COMMAND ----------

invoice = """
INVOICE
Invoice Number: INV-2026-101
Vendor: ABC Technologies
Date: 2026-11-20
Laptop Purchase: $1200
Tax: $96
Total Amount: $1296
"""

extraction_prompt = f"""Extract the following fields from the invoice. Return valid JSON only.
Fields: invoice_number, vendor_name, invoice_date, subtotal (number), tax (number), total_amount (number)
Invoice:
{invoice}"""

fields = parse_json(ask_llm(extraction_prompt))
fields

# COMMAND ----------

expected = {"invoice_number": "INV-2026-101", "vendor_name": "ABC Technologies", "invoice_date": "2026-11-20",
            "subtotal": 1200, "tax": 96, "total_amount": 1296}
print({k: (fields.get(k) == v) for k, v in expected.items()})

# COMMAND ----------

# MAGIC %md ### Same extraction in SQL with `ai_extract` (batch over many documents)

# COMMAND ----------

spark.createDataFrame([(invoice,)], "invoice_text string").createOrReplaceTempView("invoices")
display(spark.sql("""
SELECT ai_extract(invoice_text, ARRAY('invoice_number', 'vendor_name', 'invoice_date', 'total_amount')) AS extracted
FROM invoices
"""))
