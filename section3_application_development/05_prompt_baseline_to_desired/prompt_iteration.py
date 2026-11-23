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
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning | Example here |
# MAGIC |---|---|---|
# MAGIC | **Baseline** | The first, simplest prompt – our "before" picture | v0 |
# MAGIC | **Desired output** | A written description of the perfect answer | the 6 points above |
# MAGIC | **System prompt / role** | Instructions about *who* the model is | "You are a calm, professional support agent…" |
# MAGIC | **Context** | Facts the model needs but doesn't know | the 15-day replacement policy |
# MAGIC | **Constraint** | A limit or a "do NOT" rule | "at most 60 words", "do NOT promise a refund" |
# MAGIC | **Output format** | The exact shape of the answer | `{"reply": ..., "next_step": ...}` |
# MAGIC | **Few-shot** | Showing example answers inside the prompt | the keyboard example in v4 |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1  customer_message, policy, run()          ← the same input is used for every version
# MAGIC Step 2  checks(), show(), scores = {}            ← the "marking scheme" (6 True/False tests)
# MAGIC v0 → v1 → v2 → v3 → v4                           ← each calls run(), then show() adds a row to scores
# MAGIC Step 3  table from scores                        ← side-by-side comparison of all versions
# MAGIC ```
# MAGIC Every version is graded by the **same** `checks()` function, so the scores are comparable.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The input and a function to call the model
# MAGIC **What:**
# MAGIC - `customer_message` – the angry message we must answer (headphones died after **10 days**, wants money back).
# MAGIC - `policy` – the rule that applies: free **replacement** within 15 days; refunds only for unopened returns.
# MAGIC   So the correct answer is "we'll replace them", **not** "here's your money back".
# MAGIC - `run(system, user)` – the plain model call. `system` is optional: v0 sends no system prompt at all
# MAGIC   (`None`), later versions send one.
# MAGIC
# MAGIC **Why `temperature = 0.0`:** keeps the output stable, so any change you see comes from *your prompt change*, not
# MAGIC randomness.

# COMMAND ----------

import json, re
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

customer_message = "These headphones died after 10 days. This is ridiculous, I want my money back NOW!"
policy = "Defective products are replaced free within 15 days of delivery. Refunds are only for returns within 30 days of unopened items."

def run(system, user):
    # Build the message list: add a system message only if one was given
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    response = client.predict(endpoint=ENDPOINT, inputs={"messages": messages, "temperature": 0.0, "max_tokens": 300})
    return response["choices"][0]["message"]["content"].strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Automatic checks for the 6 target properties
# MAGIC **What:** small Python tests, one per requirement. Each returns True/False.
# MAGIC **Why:** "it looks better" is opinion; a score out of 6 lets you *prove* each prompt change helped.
# MAGIC
# MAGIC **What each check looks for:**
# MAGIC | Check | Passes when… |
# MAGIC |---|---|
# MAGIC | `apology` | the reply contains "sorry" or "apologi…" |
# MAGIC | `correct_policy` | the reply mentions "15 days" **and** "replace" |
# MAGIC | `next_step` | the output asks for a photo / picture / video |
# MAGIC | `max_60_words` | the reply has 60 words or fewer |
# MAGIC | `no_false_promise` | no "refund / compensation / money back", unless the reply says "not" (e.g. "we can't offer a refund") |
# MAGIC | `json_format` | the output is valid JSON with both `reply` and `next_step` keys |
# MAGIC
# MAGIC **How it handles non-JSON output:** early versions return plain text, so `json.loads` fails. The `except` branch
# MAGIC then grades the whole text as the reply and marks `json_format` as False.
# MAGIC
# MAGIC **Honest limitation:** these are simple keyword checks, so they can be fooled (e.g. "not" anywhere in the text
# MAGIC passes `no_false_promise`). They're good enough to see the trend; in real projects you'd add an LLM judge
# MAGIC (see notebook 3.2) or human review.
# MAGIC
# MAGIC `show()` prints the output, the score out of 6 and a ✓/✗ per check; `scores` collects the results for Step 3.

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
# MAGIC
# MAGIC **How to read the output:** look at the ✓/✗ line. A low score (often 0–2 out of 6) is normal here – this is the
# MAGIC "before" picture we're going to improve. If it mentions a refund, that's the model trying to please the customer
# MAGIC without knowing our policy.

# COMMAND ----------

v0 = run(None, f"Reply to this customer: {customer_message}")
scores["v0 baseline"] = show("v0 baseline", v0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### v1 – Add a role (system prompt)
# MAGIC **What:** tell the model *who it is* and *how to behave*.
# MAGIC **Why:** roles set tone and professionalism. **Expect:** more polite, but still no policy knowledge.
# MAGIC
# MAGIC **What changed vs v0:** only the system prompt. The user message is identical. Typically `apology` turns ✓, but
# MAGIC `correct_policy` stays ✗ – a role can change *how* the model speaks, not *what it knows*.

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
# MAGIC
# MAGIC **What changed vs v1:** the user message now has three labelled parts – `Policy`, `Customer message`, and a
# MAGIC bullet list of what to write. Same system prompt as v1. Look for `correct_policy` and `next_step` turning ✓.

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
# MAGIC
# MAGIC **What changed vs v2:** `v3_user = v2_user + Rules` – we keep everything from v2 and add a `Rules` block at the
# MAGIC end. Look for `max_60_words`, `no_false_promise` and `json_format` turning ✓. The output should now be pure JSON.

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
# MAGIC
# MAGIC **What changed vs v3:** the example (a *different* product – a keyboard) is placed **before** the v3 prompt.
# MAGIC Using a different case teaches the *pattern* without letting the model copy the answer word-for-word.
# MAGIC If v3 already scored 6/6, v4 may look the same here – its real value shows when you test many different customers.

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
