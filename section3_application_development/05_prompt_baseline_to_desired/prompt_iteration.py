# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.5 – Create a prompt that moves an LLM's response from a baseline to a desired output
# MAGIC
# MAGIC **The big idea for juniors:** the first prompt you write rarely gives the output you want. **Prompt engineering**
# MAGIC is changing the prompt *one step at a time* and checking whether the output gets closer to a written target.
# MAGIC
# MAGIC **Our task:** reply to an angry customer whose headphones broke after 10 days.
# MAGIC
# MAGIC **The desired output (write this down first – it's your target):**
# MAGIC 1. Starts with an apology.
# MAGIC 2. States the correct policy: free replacement within 15 days.
# MAGIC 3. Gives one clear next step (reply with a photo of the defect).
# MAGIC 4. At most 60 words.
# MAGIC 5. Does **not** promise refunds or compensation the policy doesn't offer.
# MAGIC 6. Returned as JSON: `{"reply": "...", "next_step": "..."}` so our app can display it.
# MAGIC
# MAGIC We'll go through 5 prompt versions, each adding **one technique**, and score every output against the 6 checks.
# MAGIC
# MAGIC | Version | Technique added |
# MAGIC |---|---|
# MAGIC | v0 | Baseline – just the request |
# MAGIC | v1 | **Role** (system prompt: who the model is) |
# MAGIC | v2 | **Context** (the policy) + **instructions** (what to include) |
# MAGIC | v3 | **Constraints** (length, what NOT to do) + **output format** (JSON) |
# MAGIC | v4 | **Few-shot example** (show one perfect answer) |

# COMMAND ----------

# MAGIC %md ### Step 1 – The input and a function to call the model

# COMMAND ----------

import json, re
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

customer_message = "These headphones died after 10 days. This is ridiculous, I want my money back NOW!"
policy = "Defective products are replaced free within 15 days of delivery. Refunds are only for returns within 30 days of unopened items."

def run(system, user):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    response = client.predict(endpoint=ENDPOINT, inputs={"messages": messages, "temperature": 0.0, "max_tokens": 300})
    return response["choices"][0]["message"]["content"].strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Automatic checks for the 6 target properties
# MAGIC **What:** small Python tests, one per requirement. Each returns True/False.
# MAGIC **Why:** "it looks better" is opinion; a score out of 6 lets you *prove* each prompt change helped.

# COMMAND ----------

def checks(output):
    try:
        data = json.loads(output[output.find("{"): output.rfind("}") + 1])
        reply = data.get("reply", "")
        is_json = "reply" in data and "next_step" in data
    except Exception:
        reply, is_json = output, False
    text = reply.lower()
    return {
        "apology": any(w in text for w in ["sorry", "apologi"]),
        "correct_policy": "15 days" in text and "replace" in text,
        "next_step": any(w in output.lower() for w in ["photo", "picture", "video"]),
        "max_60_words": len(reply.split()) <= 60,
        "no_false_promise": not re.search(r"refund|compensat|money back", text) or "not" in text,
        "json_format": is_json,
    }

def show(version, output):
    result = checks(output)
    print(f"=== {version}: score {sum(result.values())}/6 ===")
    print(output)
    print({k: ("✓" if v else "✗") for k, v in result.items()}, "\n")
    return result

scores = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ### v0 – Baseline
# MAGIC **What:** just the customer's message.
# MAGIC **Expect:** a generic, possibly long reply; it may promise a refund (the customer asked for one) and won't know the
# MAGIC 15-day policy.

# COMMAND ----------

v0 = run(None, f"Reply to this customer: {customer_message}")
scores["v0 baseline"] = show("v0 baseline", v0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### v1 – Add a role (system prompt)
# MAGIC **What:** tell the model *who it is* and *how to behave*.
# MAGIC **Why:** roles set tone and professionalism. **Expect:** more polite, but still no policy knowledge.

# COMMAND ----------

v1_system = "You are a calm, professional customer support agent for the electronics shop ShopEase."
v1 = run(v1_system, f"Reply to this customer: {customer_message}")
scores["v1 + role"] = show("v1 + role", v1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### v2 – Add context (policy) and explicit instructions
# MAGIC **What:** paste the policy and list what the reply must contain.
# MAGIC **Why:** the model can't follow a policy it hasn't seen. Explicit instructions ("apologise", "give one next step")
# MAGIC turn hopes into requirements. **Expect:** correct policy and a next step; length/format still uncontrolled.

# COMMAND ----------

v2_user = f"""Policy: {policy}

Customer message: {customer_message}

Write a reply that:
- apologises for the problem,
- explains the policy that applies,
- gives one clear next step (ask for a photo of the defect)."""
v2 = run(v1_system, v2_user)
scores["v2 + context & instructions"] = show("v2 + context & instructions", v2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### v3 – Add constraints and an output format
# MAGIC **What:** a word limit, a "do NOT" rule, and an exact JSON shape.
# MAGIC **Why:** negative instructions prevent known failure modes (promising refunds); a fixed format lets code use the
# MAGIC output. **Expect:** short, safe, machine-readable output.

# COMMAND ----------

v3_user = v2_user + """
Rules:
- At most 60 words in "reply".
- Do NOT promise a refund, money back or compensation. Offer only what the policy allows.
- Return ONLY valid JSON, no other text: {"reply": "<message to the customer>", "next_step": "<one action for the customer>"}"""
v3 = run(v1_system, v3_user)
scores["v3 + constraints & format"] = show("v3 + constraints & format", v3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### v4 – Add a few-shot example
# MAGIC **What:** show the model one example of a *perfect* answer for a different case.
# MAGIC **Why:** examples communicate style, length and structure better than rules alone, and make outputs more
# MAGIC consistent across many customers. **Expect:** the most consistent, on-target output.

# COMMAND ----------

few_shot = """Example
Customer message: My keyboard arrived with a broken key, so annoying.
Output: {"reply": "I'm sorry your keyboard arrived damaged. Because it's within 15 days of delivery, we'll replace it free of charge.", "next_step": "Reply with a photo of the broken key so we can send the replacement."}

Now do the same for the real customer.
"""
v4 = run(v1_system, few_shot + v3_user)
scores["v4 + few-shot"] = show("v4 + few-shot", v4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Compare all versions
# MAGIC **How to read:** each row is a prompt version; ✓ columns should fill in as we go down. If a change doesn't add a ✓,
# MAGIC it wasn't needed – or the instruction needs rewording. The final prompt is the one that passes all 6.

# COMMAND ----------

table = [(v, sum(r.values()), *["✓" if r[k] else "✗" for k in r]) for v, r in scores.items()]
cols = "version string, score int, " + ", ".join(f"{k} string" for k in next(iter(scores.values())))
display(spark.createDataFrame(table, cols))

# COMMAND ----------

# MAGIC %md
# MAGIC **Tips for juniors**
# MAGIC - Define the **target output** and checks *before* writing prompts.
# MAGIC - Change **one thing at a time**, so you know which change helped.
# MAGIC - Test on **several inputs**, not just one – a prompt can be perfect for one customer and fail for another.
# MAGIC - Keep `temperature = 0` while iterating so differences come from the prompt, not randomness.
# MAGIC - Save prompt versions (e.g. in MLflow's **Prompt Registry**) so you can roll back.
# MAGIC
# MAGIC **Exam takeaways:** techniques that move output toward a target – **role/system prompt**, **context**, **explicit
# MAGIC instructions**, **constraints (incl. negative ones)**, **output format/schema**, **few-shot examples**, and
# MAGIC (for reasoning tasks) **step-by-step instructions**.
