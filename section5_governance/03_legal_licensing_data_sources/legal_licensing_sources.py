# Databricks notebook source
# MAGIC %md
# MAGIC # Section 5.3 – Use legal / licensing requirements for data sources to avoid legal risk
# MAGIC
# MAGIC **The big idea for juniors:** "it's on the internet" does **not** mean you may use it. Every document you put into
# MAGIC a RAG index, a fine-tuning set or a prompt has an **owner** and a **licence** (or terms of service, or a contract),
# MAGIC and personal data has **privacy rules** (consent, purpose, retention). Using the wrong source can lead to copyright
# MAGIC claims, contract breaches or privacy fines.
# MAGIC
# MAGIC So before indexing, we keep a **data source register**: one row per source with its licence and personal-data
# MAGIC facts, and a decision: **include**, **include with conditions**, **needs legal review**, or **exclude**.
# MAGIC
# MAGIC > This notebook teaches the engineering workflow. It is **not legal advice** – real decisions go to your legal /
# MAGIC > privacy team. Licence summaries below are simplified.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Licence** | The permission the owner gives you: what you may do with their content |
# MAGIC | **Commercial use** | Using it in a product or service that makes money – our support assistant counts |
# MAGIC | **Attribution (BY)** | You must credit the author/source |
# MAGIC | **Share-alike (SA)** | Things you make from it must be shared under the same licence |
# MAGIC | **Non-commercial (NC)** | Not allowed in commercial products |
# MAGIC | **Copyleft (e.g. GPL)** | Strong share-alike for code; needs review before use |
# MAGIC | **Terms of service (ToS)** | A website's rules; many forbid scraping or AI use |
# MAGIC | **Consent / purpose limitation** | Personal data may only be used for the purposes people agreed to |
# MAGIC | **Unity Catalog tags** | Key–value labels on tables/columns (e.g. `license = CC BY 4.0`) so governance is searchable |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 licence knowledge table (what each licence allows)
# MAGIC Step 2 candidate data sources (9) with licence + personal-data facts
# MAGIC Step 3 decision rules → include / conditions / review / exclude
# MAGIC Step 4 save the register as a Delta table + tag tables in Unity Catalog
# MAGIC Step 5 chunks carry source + licence metadata → only allowed chunks reach the RAG index
# MAGIC Step 6 attribution added to answers automatically
# MAGIC Step 7 don't forget the model's licence
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – What each licence allows (simplified)
# MAGIC **What:** a small lookup table. In a real company your legal team maintains this list.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_governance"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

LICENSES = {
    # licence:                       (commercial_ok, attribution, share_alike, needs_review, note)
    "Internal (company-owned)":      (True,  False, False, False, "we own it"),
    "CC0 / public domain":           (True,  False, False, False, "no conditions"),
    "CC BY 4.0":                     (True,  True,  False, False, "credit the source"),
    "CC BY-SA 4.0":                  (True,  True,  True,  True,  "credit + share-alike; check how outputs are shared"),
    "CC BY-NC 4.0":                  (False, True,  False, False, "non-commercial only"),
    "MIT / Apache-2.0":              (True,  True,  False, False, "keep copyright/licence notice"),
    "GPL-3.0":                       (True,  True,  True,  True,  "copyleft – legal review before use"),
    "Vendor contract (support use)": (True,  False, False, False, "allowed by our contract for support"),
    "Unknown / no licence":          (False, False, False, True,  "all rights reserved by default"),
}
display(spark.createDataFrame([(k, *v) for k, v in LICENSES.items()],
        "license string, commercial_ok boolean, attribution boolean, share_alike boolean, needs_review boolean, note string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Candidate data sources for the support assistant
# MAGIC **Fields:**
# MAGIC | Field | Meaning |
# MAGIC |---|---|
# MAGIC | `license` | one of the keys from Step 1 |
# MAGIC | `tos_allows_ai_use` | `True`/`False`/`None` (not applicable) – for websites |
# MAGIC | `has_personal_data` | does it contain PII? |
# MAGIC | `consent_for_ai` | did people agree to this use? (`None` if no personal data) |

# COMMAND ----------

sources = [
    # source,                    license,                          tos_allows_ai_use, has_personal_data, consent_for_ai
    ("returns_policy_docs",      "Internal (company-owned)",       None,  False, None),
    ("vendor_product_manuals",   "Vendor contract (support use)",  None,  False, None),
    ("open_faq_dataset",         "CC BY 4.0",                      None,  False, None),
    ("wikipedia_product_pages",  "CC BY-SA 4.0",                   True,  False, None),
    ("academic_review_corpus",   "CC BY-NC 4.0",                   None,  False, None),
    ("competitor_site_scrape",   "Unknown / no licence",           False, False, None),
    ("customer_chat_logs",       "Internal (company-owned)",       None,  True,  True),
    ("old_marketing_email_list", "Internal (company-owned)",       None,  True,  False),
    ("github_troubleshoot_repo", "GPL-3.0",                        None,  False, None),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Decision rules
# MAGIC **The rules, in order (first match wins):**
# MAGIC 1. Website ToS forbids AI use → **exclude**.
# MAGIC 2. Licence doesn't allow commercial use → **exclude** (our assistant is a commercial service).
# MAGIC 3. Personal data **without** consent for this purpose → **exclude**.
# MAGIC 4. Licence flagged `needs_review` (copyleft, share-alike, unknown) → **needs legal review**.
# MAGIC 5. Otherwise **include**, plus conditions: attribution, PII masking (Section 5.1), retention limits.
# MAGIC
# MAGIC **How to read the output:**
# MAGIC | Source | Decision | Why |
# MAGIC |---|---|---|
# MAGIC | returns_policy_docs, vendor manuals | include | owned / contract allows |
# MAGIC | open_faq_dataset | include + attribution | CC BY |
# MAGIC | wikipedia_product_pages, github repo | needs review | share-alike / copyleft |
# MAGIC | academic_review_corpus | exclude | non-commercial |
# MAGIC | competitor_site_scrape | exclude | ToS forbids it |
# MAGIC | customer_chat_logs | include + mask PII | consent given |
# MAGIC | old_marketing_email_list | exclude | personal data, no consent |

# COMMAND ----------

def decide(license, tos_allows_ai_use, has_personal_data, consent_for_ai):
    commercial_ok, attribution, share_alike, needs_review, note = LICENSES[license]
    if tos_allows_ai_use is False:
        return "exclude", "terms of service forbid AI/scraping use"
    if not commercial_ok and license != "Unknown / no licence":
        return "exclude", f"licence: {note}"
    if has_personal_data and not consent_for_ai:
        return "exclude", "personal data without consent for this purpose"
    if needs_review:
        return "needs_legal_review", f"licence: {note}"
    conditions = []
    if attribution:
        conditions.append("show attribution")
    if has_personal_data:
        conditions.append("mask PII before indexing; apply retention policy")
    return ("include_with_conditions" if conditions else "include"), "; ".join(conditions) or note

register = [(s, lic, tos, pii, consent, *decide(lic, tos, pii, consent), LICENSES[lic][1])
            for s, lic, tos, pii, consent in sources]
register_df = spark.createDataFrame(register, "source string, license string, tos_allows_ai_use boolean, "
                                              "has_personal_data boolean, consent_for_ai boolean, decision string, "
                                              "reason string, attribution_required boolean")
display(register_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Save the register and tag tables in Unity Catalog
# MAGIC **What:** the register becomes a Delta table (`data_source_register`) – the audit trail of *why* each source is in or
# MAGIC out. We also put **tags** on a table, so anyone browsing the catalog sees its licence and governance status.
# MAGIC **Why:** auditors and legal teams ask "where did this answer's data come from and were we allowed to use it?" –
# MAGIC Unity Catalog **lineage** + tags + this register answer that.
# MAGIC **How to read the output:** the tags on `kb_chunks` (if tags are supported in your workspace).

# COMMAND ----------

register_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.data_source_register")
spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.data_source_register IS "
          "'Licence and privacy decision for every candidate GenAI data source. Reviewed by legal.'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Licence metadata travels with every chunk
# MAGIC **What:** each chunk keeps `source`, `license` and `attribution` columns. We join to the register and keep only
# MAGIC chunks whose source decision is `include` or `include_with_conditions`. Only this filtered table would be synced to
# MAGIC Vector Search.
# MAGIC **Why:** if a source's decision changes later (licence revoked, consent withdrawn), you update the register, re-run
# MAGIC this filter and **sync the index** – the content disappears from answers.
# MAGIC **How to read the output:** 4 of the 7 chunks are allowed; the NC, scraped and marketing ones are gone.

# COMMAND ----------

chunks = [
    ("c1", "returns_policy_docs",      "Items can be returned within 30 days for a full refund.", "ShopCo Returns Policy"),
    ("c2", "vendor_product_manuals",   "To reset the X200 headphones, hold the power button for 10 seconds.", "Acme X200 Manual"),
    ("c3", "open_faq_dataset",         "Most couriers update tracking within 24 hours of dispatch.", "OpenFAQ project (CC BY 4.0)"),
    ("c4", "academic_review_corpus",   "Users rated the X200 battery life 4.1 out of 5.", "Review Study 2021"),
    ("c5", "competitor_site_scrape",   "Competitor offers 60-day returns.", "competitor.example"),
    ("c6", "customer_chat_logs",       "Customer <PERSON> confirmed the X200 reset fixed pairing.", "Support chat (masked)"),
    ("c7", "old_marketing_email_list", "Dear Maria, our summer sale starts now!", "Marketing archive"),
]
spark.createDataFrame(chunks, "chunk_id string, source string, chunk_text string, attribution_text string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.kb_chunks_raw")

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.kb_chunks AS
SELECT c.chunk_id, c.source, c.chunk_text, c.attribution_text, r.license, r.attribution_required
FROM {CATALOG}.{SCHEMA}.kb_chunks_raw c
JOIN {CATALOG}.{SCHEMA}.data_source_register r ON c.source = r.source
WHERE r.decision IN ('include', 'include_with_conditions')
""")
display(spark.table(f"{CATALOG}.{SCHEMA}.kb_chunks"))

try:
    spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.kb_chunks SET TAGS ('governance' = 'licence-checked', "
              "'contains_pii' = 'masked', 'register' = 'data_source_register')")
    display(spark.sql(f"SELECT tag_name, tag_value FROM {CATALOG}.information_schema.table_tags "
                      f"WHERE schema_name = '{SCHEMA}' AND table_name = 'kb_chunks'"))
except Exception as e:
    print("Table tags not available here:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Attribution added by code, not by the LLM
# MAGIC **What:** a tiny RAG answer. We retrieve chunks with a keyword match (enough for the demo), let the LLM answer from
# MAGIC them, and then the **app** appends a "Sources" line for chunks whose licence requires attribution.
# MAGIC **Why by code?** an LLM may forget or invent citations. A licence condition must be met **every** time.
# MAGIC **How to read the output:** an answer about tracking, followed by `Sources: OpenFAQ project (CC BY 4.0)`.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
client = get_deploy_client("databricks")

question = "How long until my tracking updates after dispatch?"
allowed = spark.table(f"{CATALOG}.{SCHEMA}.kb_chunks").toPandas()
words = [w for w in question.lower().split() if len(w) > 3]
hits = allowed[allowed.chunk_text.str.lower().apply(lambda t: any(w.strip("?") in t for w in words))]

r = client.predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
    "messages": [{"role": "system", "content": "Answer in one sentence using only the context."},
                 {"role": "user", "content": "Context:\n" + "\n".join(hits.chunk_text) + f"\n\nQuestion: {question}"}],
    "temperature": 0.0, "max_tokens": 80})
answer = r["choices"][0]["message"]["content"].strip()
credits = sorted(set(hits[hits.attribution_required].attribution_text))
print(answer + (f"\n\nSources: {', '.join(credits)}" if credits else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – The model has a licence too
# MAGIC Data isn't the only legal input. Open models come with their own licence terms, for example:
# MAGIC | Model family | Licence (summary – always read the current text) |
# MAGIC |---|---|
# MAGIC | Meta Llama 3.x | Llama Community Licence + acceptable-use policy; attribution ("Built with Llama"); extra terms for very large services |
# MAGIC | Google Gemma | Gemma Terms of Use + prohibited-use policy |
# MAGIC | Qwen (many versions), Mistral (some) | Apache-2.0 |
# MAGIC
# MAGIC Check the **model card** (Section 3.9) and Databricks Marketplace listing before choosing a model.
# MAGIC
# MAGIC ### Checklist before a source goes into a GenAI app
# MAGIC 1. Who owns it, and what licence / contract / ToS applies?
# MAGIC 2. Is commercial use allowed? Attribution? Share-alike?
# MAGIC 3. Does it contain personal data? Consent and purpose? Retention?
# MAGIC 4. Record the decision in a register; tag tables in Unity Catalog; keep lineage.
# MAGIC 5. Carry licence metadata into chunks; filter before indexing; attribute in answers.
# MAGIC 6. Re-check when licences or consents change – then update and re-sync the index.
# MAGIC
# MAGIC **Exam takeaways:** prefer **owned, licensed or permissively licensed** sources; **exclude** non-commercial,
# MAGIC ToS-forbidden, unknown-licence and non-consented personal data; send copyleft/share-alike to **legal review**;
# MAGIC keep licence metadata with the data so you can **filter, attribute and remove** content later.
