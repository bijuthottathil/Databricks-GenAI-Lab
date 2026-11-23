# Databricks notebook source
# MAGIC %md
# MAGIC # Section 5.4 – Recommend an alternative for problematic text in a data source feeding a GenAI app
# MAGIC
# MAGIC **The big idea for juniors:** our support assistant uses a **community forum** as a knowledge source. Most posts are
# MAGIC useful, but some contain problems: rude language, personal data, **outdated** policy, dangerous advice, bias, spam.
# MAGIC If they go into the RAG index, the assistant may repeat them.
# MAGIC
# MAGIC The lazy fix is "delete the whole source". That throws away all the good content. The better answer is to pick the
# MAGIC **right mitigation per problem**:
# MAGIC
# MAGIC | Problem in the text | Recommended alternative |
# MAGIC |---|---|
# MAGIC | Personal data (PII) | **mask** it (Section 5.1), keep the post |
# MAGIC | Rude / toxic wording but useful facts | **rewrite** (detoxify) with an LLM, keep the facts |
# MAGIC | Toxic with no useful content, spam / ads | **filter out** the row |
# MAGIC | Outdated information | **replace with the authoritative source** (official policy) or add dates and prefer recent |
# MAGIC | Dangerous advice, biased / discriminatory content | **exclude** and send to a human (SME) **review queue** |
# MAGIC | Unsure | **flag with metadata** and exclude from retrieval until reviewed |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Problematic text** | Content that is harmful, wrong, private or low-quality for the app |
# MAGIC | **Mitigation** | The action we take: mask, rewrite, filter, replace, review |
# MAGIC | **Detoxify** | Rewrite text politely while keeping the facts |
# MAGIC | **Authoritative source** | The official, owned, current document (e.g. the real returns policy) |
# MAGIC | **Review queue** | A table of items waiting for a human expert to decide |
# MAGIC | **Metadata filter** | A column like `quality_flag` used to keep rows out of retrieval |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 forum posts (10) + the official policy
# MAGIC Step 2 detect problems: regex for PII + LLM labeller for the rest   → issues per post
# MAGIC Step 3 choose a mitigation per post (rules)                          → plan table
# MAGIC Step 4 apply: mask / rewrite / drop / replace / review               → clean table + review queue
# MAGIC Step 5 compare: "drop the source" vs "targeted mitigation"
# MAGIC Step 6 the same question answered from the raw vs the clean knowledge base
# MAGIC Step 7 keep it clean: checks at ingestion
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The data source: forum posts
# MAGIC **What:** 10 posts. The comment next to each says what's wrong with it (for you – the code doesn't see it).
# MAGIC The **official** returns policy today is **30 days**; post `p3` still says 14 days (the old policy).

# COMMAND ----------

import re, json
import pandas as pd
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
CATALOG, SCHEMA = "workspace", "fde_governance"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

OFFICIAL_POLICY = {"returns": "Items can be returned within 30 days of delivery for a full refund (policy updated 2024)."}

posts = [
    ("p1", "To pair the X200 headphones, hold the power button for 10 seconds until the light blinks blue."),   # good
    ("p2", "Refunds go back to the original card within 5-7 business days."),                                    # good
    ("p3", "You only have 14 days to return anything, after that they refuse."),                                 # outdated
    ("p4", "Support is useless idiots, but FYI resetting the router fixed my streaming box."),                   # toxic + useful
    ("p5", "Email me at jake.r@example.com and I'll tell you how I got my refund, or call 555-301-2299."),       # PII
    ("p6", "If the battery swells, just pierce it to let the gas out, works every time."),                        # dangerous
    ("p7", "BUY CHEAP WATCHES!!! visit cheap-watch-deals.example now!!!"),                                       # spam
    ("p8", "Older customers never understand the app, don't bother helping them."),                               # biased
    ("p9", "Express shipping usually arrives in 2 business days."),                                               # good
    ("p10", "You people are morons."),                                                                            # toxic, no content
]
for pid, text in posts:
    print(pid.ljust(4), text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Detect the problems
# MAGIC **Two detectors, because each is good at different things:**
# MAGIC - **Regex** for PII (emails, phones) – exact and fast.
# MAGIC - **LLM labeller** for meaning-based problems: toxic, outdated (compared with the official policy we pass in),
# MAGIC   dangerous, biased, spam – plus `has_useful_info` (is there a fact worth keeping?).
# MAGIC
# MAGIC **How to read the output:** one row per post with its issue list, e.g. `p4 → [toxic]`, useful = `true`;
# MAGIC `p10 → [toxic]`, useful = `false`. Good posts have an empty list.

# COMMAND ----------

PII_PATTERNS = [r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", r"\+?\d[\d\s\-]{7,}\d"]
ISSUES = ["toxic", "outdated", "dangerous", "biased", "spam"]

LABEL_PROMPT = f"""You review forum posts before they are used as a knowledge source for a customer-support assistant.
Official policy (the truth): {OFFICIAL_POLICY['returns']}
Possible issues: {ISSUES}
- toxic: insults or abusive wording.  - outdated: contradicts the official policy.
- dangerous: advice that could hurt someone.  - biased: unfair statements about a group of people.
- spam: advertising or unrelated links.
Return ONLY JSON: {{"issues": [list of issues, empty if none], "has_useful_info": true/false}}
has_useful_info = the post contains a correct, helpful product or service fact."""

def llm_label(text):
    r = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": LABEL_PROMPT}, {"role": "user", "content": text}],
        "temperature": 0.0, "max_tokens": 60})
    out = r["choices"][0]["message"]["content"]
    try:
        v = json.loads(out[out.find("{"): out.rfind("}") + 1])
        return [i for i in v.get("issues", []) if i in ISSUES], bool(v.get("has_useful_info", False))
    except Exception:
        return ["unparseable"], False

labels = []
for pid, text in posts:
    issues, useful = llm_label(text)
    if any(re.search(p, text) for p in PII_PATTERNS):
        issues = ["pii"] + issues
    labels.append((pid, text, issues, useful))

display(spark.createDataFrame(labels, "post_id string, text string, issues array<string>, has_useful_info boolean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Choose a mitigation for each post
# MAGIC **Rules, most serious first (first match wins):**
# MAGIC | If the post is… | Mitigation |
# MAGIC |---|---|
# MAGIC | dangerous or biased | `review` – exclude now, a human expert decides |
# MAGIC | spam | `drop` |
# MAGIC | outdated | `replace` – use the official policy text instead |
# MAGIC | toxic **and** useful | `rewrite` – detoxify, keep the fact |
# MAGIC | toxic, not useful | `drop` |
# MAGIC | PII (only) | `mask` |
# MAGIC | nothing wrong | `keep` |
# MAGIC
# MAGIC A post can have several issues – e.g. toxic **and** PII: we rewrite **and** mask (`rewrite+mask`).

# COMMAND ----------

def choose_mitigation(issues, useful):
    if "dangerous" in issues or "biased" in issues or "unparseable" in issues:
        return "review"
    if "spam" in issues:
        return "drop"
    if "outdated" in issues:
        return "replace"
    if "toxic" in issues:
        action = "rewrite" if useful else "drop"
        return action + ("+mask" if "pii" in issues and action == "rewrite" else "")
    if "pii" in issues:
        return "mask"
    return "keep"

plan = [(pid, text, issues, useful, choose_mitigation(issues, useful)) for pid, text, issues, useful in labels]
display(spark.createDataFrame(plan, "post_id string, text string, issues array<string>, has_useful_info boolean, mitigation string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Apply the mitigations
# MAGIC **Code walk-through:**
# MAGIC - `mask()` – typed placeholders for emails/phones.
# MAGIC - `detoxify()` – LLM rewrite: "keep every fact, remove insults, don't add information".
# MAGIC - `replace` – the row's text becomes the official policy and `source` becomes `official_policy`.
# MAGIC - `drop` – row is not written. `review` – row goes to the `forum_review_queue` table instead.
# MAGIC
# MAGIC **How to read the output:** the clean table (`forum_posts_clean`) has the good posts, the rewritten `p4`, the masked
# MAGIC `p5`, and the official policy instead of `p3`. The review queue has `p6` and `p8`.

# COMMAND ----------

def mask(text):
    text = re.sub(PII_PATTERNS[0], "<EMAIL>", text)
    return re.sub(PII_PATTERNS[1], "<PHONE>", text)

def detoxify(text):
    r = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": "Rewrite the text politely. Keep every factual statement, remove insults, "
                                                   "do not add new information. Return only the rewritten text."},
                     {"role": "user", "content": text}],
        "temperature": 0.0, "max_tokens": 80})
    return r["choices"][0]["message"]["content"].strip()

clean_rows, review_rows = [], []
for pid, text, issues, useful, action in plan:
    if action == "keep":
        clean_rows.append((pid, "forum", text, action))
    elif action == "mask":
        clean_rows.append((pid, "forum", mask(text), action))
    elif action.startswith("rewrite"):
        new_text = detoxify(text)
        clean_rows.append((pid, "forum", mask(new_text) if action.endswith("+mask") else new_text, action))
    elif action == "replace":
        clean_rows.append((pid, "official_policy", OFFICIAL_POLICY["returns"], action))
    elif action == "review":
        review_rows.append((pid, text, ", ".join(issues), "pending"))
    # "drop" → nothing written

spark.createDataFrame(clean_rows, "post_id string, source string, text string, mitigation string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.forum_posts_clean")
spark.createDataFrame(review_rows, "post_id string, text string, issues string, status string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.forum_review_queue")

display(spark.table(f"{CATALOG}.{SCHEMA}.forum_posts_clean"))
display(spark.table(f"{CATALOG}.{SCHEMA}.forum_review_queue"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Compare the options
# MAGIC **How to read the output:**
# MAGIC | Option | Useful facts kept | Problematic text left |
# MAGIC |---|---|---|
# MAGIC | keep everything | all | all problems |
# MAGIC | drop the whole source | 0 | none – but the pairing tip, refund timing, shipping time are lost |
# MAGIC | **targeted mitigation** | most (≈ 6–7 rows) | none (problem rows masked, rewritten, replaced or held for review) |
# MAGIC
# MAGIC That is the recommendation: **targeted mitigation** beats both extremes.

# COMMAND ----------

problem_rows = sum(1 for _, _, issues, _, _ in plan if issues)
print(f"Keep everything   : {len(posts)} rows, {problem_rows} with problems")
print(f"Drop the source   : 0 rows, 0 problems, all useful facts lost")
print(f"Targeted mitigation: {len(clean_rows)} rows kept, {len(review_rows)} sent to review, "
      f"{len(posts) - len(clean_rows) - len(review_rows)} dropped")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Why it matters: the same question, raw vs. clean knowledge base
# MAGIC **What:** "How many days do I have to return an item?" answered from posts that mention "return".
# MAGIC **How to read the output:** the raw knowledge base gives the **wrong** answer (14 days, from `p3`); the clean one
# MAGIC gives **30 days**, because `p3` was replaced by the official policy.

# COMMAND ----------

def answer(question, docs):
    r = client.predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": "Answer in one sentence using only the context."},
                     {"role": "user", "content": "Context:\n" + "\n".join(docs) + f"\n\nQuestion: {question}"}],
        "temperature": 0.0, "max_tokens": 60})
    return r["choices"][0]["message"]["content"].strip()

q = "How many days do I have to return an item?"
raw_docs = [t for _, t in posts if "return" in t.lower()]
clean_docs = [r[2] for r in clean_rows if "return" in r[2].lower()]
print("RAW   :", answer(q, raw_docs))
print("CLEAN :", answer(q, clean_docs))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – Keep the source clean over time
# MAGIC Mitigation is not a one-off. New forum posts arrive every day:
# MAGIC - Run Steps 2–4 as a **scheduled job** on new rows only, before they are synced to Vector Search.
# MAGIC - Add **data quality expectations** (Lakeflow Declarative Pipelines `@dlt.expect_or_drop`) e.g. "no email pattern".
# MAGIC - Keep a `quality_flag` / `mitigation` column and **filter on it** at retrieval time.
# MAGIC - When an SME approves or rejects a review-queue item, update the clean table and **sync the index**.
# MAGIC - Monitor answers in production (Section 6) – complaints often reveal new bad content.
# MAGIC
# MAGIC **Exam takeaways:** don't just delete a valuable source – recommend the **targeted alternative**: mask PII,
# MAGIC detoxify useful-but-rude text, filter spam and pure toxicity, **replace outdated content with the authoritative
# MAGIC source**, and route dangerous or biased content to human review; enforce it continuously at ingestion.
