# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.9 – Select a model from a model hub or marketplace using model metadata / model cards
# MAGIC
# MAGIC **The big idea for juniors:** model hubs (Hugging Face, Databricks Marketplace, the `system.ai` catalog in Unity
# MAGIC Catalog) list thousands of models. Each one comes with a **model card** – its "nutrition label". Reading cards
# MAGIC lets you rule out unsuitable models in minutes, *before* spending time testing them.
# MAGIC
# MAGIC **What a model card tells you**
# MAGIC
# MAGIC | Field | Why you care |
# MAGIC |---|---|
# MAGIC | **Task / pipeline tag** (text-generation, summarization, embeddings, classification…) | Does it do the job at all? |
# MAGIC | **License** (Apache-2.0, MIT, custom, non-commercial) | Are we *allowed* to use it in our product? |
# MAGIC | **Languages** | Does it understand our users' language? |
# MAGIC | **Size (parameters)** | Cost, speed, and hardware needed |
# MAGIC | **Context length** | Will our inputs fit? |
# MAGIC | **Training data** | Domain fit, possible bias, data-usage concerns |
# MAGIC | **Evaluation results** (benchmarks) | How well it did on tests similar to our task |
# MAGIC | **Intended use & limitations** | What the authors say it should *not* be used for |
# MAGIC
# MAGIC **Our requirement (the "job description"):** summarise **German** customer reviews into one English sentence,
# MAGIC for a **commercial** product, at high volume (so ≤ 15B parameters), inputs up to ~6,000 tokens.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Read a model card in its raw form
# MAGIC **What:** Hugging Face model cards are a `README.md` that starts with a **YAML** block of metadata between `---`
# MAGIC lines, followed by human-written text. Below is an example card (a made-up model, for practice).
# MAGIC **Why parse it:** with many candidates you filter on the structured fields automatically.

# COMMAND ----------

import yaml

example_card = """---
license: apache-2.0
language: [en, de, fr]
pipeline_tag: summarization
parameters: 7B
context_length: 8192
datasets: [cnn_dailymail, multilingual_reviews]
model-index:
  - results:
      - task: summarization
        metric: rouge-l
        value: 0.41
---
# acme/multilingual-summarizer-7b
Summarises English, German and French text into short English summaries.
## Limitations
Not suitable for legal or medical documents. May omit numbers from long inputs.
"""

front_matter = example_card.split("---")[1]          # the YAML part between the first two '---'
metadata = yaml.safe_load(front_matter)
print(metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – A small "hub" of candidate model cards
# MAGIC **What:** 7 example cards (made-up names, realistic fields) as a table.
# MAGIC **Why made-up:** so the exercise is about *reading metadata*, not memorising real models' current specs.

# COMMAND ----------

cards = [
    # name, task, license, commercial_ok, languages, params_b, context, rouge_l, limitations
    ("acme/multilingual-summarizer-7b", "summarization", "apache-2.0", True, "en,de,fr", 7, 8192, 0.41, "Not for legal/medical"),
    ("bigco/giant-llm-180b", "text-generation", "custom", True, "en,de,fr,es", 180, 32768, 0.45, "Very expensive to run"),
    ("uni-lab/research-summarizer-3b", "summarization", "cc-by-nc-4.0", False, "en,de", 3, 4096, 0.39, "Non-commercial research only"),
    ("tiny/english-summarizer-1b", "summarization", "mit", True, "en", 1, 2048, 0.35, "English only"),
    ("acme/german-chat-12b", "text-generation", "apache-2.0", True, "de,en", 12, 16384, 0.40, "General chat; summarisation via prompting"),
    ("embedco/multilingual-embed-0.5b", "feature-extraction", "apache-2.0", True, "en,de,fr", 0.5, 8192, None, "Embeddings only – no text output"),
    ("oldco/summarizer-v1-6b", "summarization", "apache-2.0", True, "en,de", 6, 1024, 0.37, "Short inputs only"),
]
cards_df = spark.createDataFrame(cards, "name string, task string, license string, commercial_ok boolean, languages string, "
                                        "params_b double, context_length int, rouge_l double, limitations string")
display(cards_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Filter on hard requirements (must-haves) and record *why* each model is rejected
# MAGIC **What:** check each card against our must-haves. A model failing any must-have is out, however good its score.
# MAGIC **Why record reasons:** you'll need to explain the choice to your team (and auditors).

# COMMAND ----------

REQUIRED_TASKS = {"summarization", "text-generation"}     # a chat model can summarise when prompted
MAX_PARAMS_B = 15
MIN_CONTEXT = 6000

decisions = []
for name, task, lic, commercial, langs, params, ctx, rouge, limits in cards:
    reasons = []
    if task not in REQUIRED_TASKS:
        reasons.append(f"wrong task ({task})")
    if not commercial:
        reasons.append(f"license {lic} forbids commercial use")
    if "de" not in langs.split(","):
        reasons.append("no German")
    if params > MAX_PARAMS_B:
        reasons.append(f"too large ({params}B)")
    if ctx < MIN_CONTEXT:
        reasons.append(f"context too short ({ctx})")
    decisions.append((name, "KEEP" if not reasons else "REJECT", "; ".join(reasons), rouge, task))

display(spark.createDataFrame(decisions, "name string, decision string, reasons string, rouge_l double, task string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Rank the survivors on the most relevant evaluation metric
# MAGIC **What:** among models that pass, prefer the one with the best **ROUGE-L** (a summarisation metric: how much the
# MAGIC model's summary overlaps with human-written summaries; higher is better), then prefer a dedicated
# MAGIC summarisation model over a general chat model.
# MAGIC **Why:** benchmark scores only matter *after* the must-haves are met, and only for metrics that match your task.

# COMMAND ----------

survivors = [d for d in decisions if d[1] == "KEEP"]
ranked = sorted(survivors, key=lambda d: (-(d[3] or 0), d[4] != "summarization"))
for i, d in enumerate(ranked, 1):
    print(f"{i}. {d[0]}  (ROUGE-L {d[3]}, task {d[4]})")
print(f"\nShortlist winner: {ranked[0][0]} → next step: test it on 20–50 of OUR reviews before deciding (see 3.10).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Real metadata in *this* workspace
# MAGIC **What:** Databricks lists the models you can use right now:
# MAGIC - **Serving endpoints** – each has a *task* type (`llm/v1/chat`, `llm/v1/embeddings`…), which is metadata too.
# MAGIC - **`system.ai` catalog** in Unity Catalog – curated models with descriptions (their "cards").
# MAGIC **Why:** in practice you'll start from what's already available and governed in your workspace.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
endpoint_rows = []
for ep in w.serving_endpoints.list():
    if ep.name.startswith("databricks-"):
        endpoint_rows.append((ep.name, ep.task))
display(spark.createDataFrame(endpoint_rows, "endpoint string, task string").orderBy("task", "endpoint"))

# COMMAND ----------

try:
    models = list(w.registered_models.list(catalog_name="system", schema_name="ai", max_results=50))
    display(spark.createDataFrame([(m.full_name, (m.comment or "")[:200]) for m in models],
                                  "model string, description string"))
except Exception as e:
    print("Could not list system.ai models in this workspace:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC **Where to find models**
# MAGIC - **Databricks Foundation Model APIs** – ready-to-use endpoints (pay-per-token or provisioned throughput).
# MAGIC - **`system.ai` in Unity Catalog** – curated open models you can deploy to your own endpoint.
# MAGIC - **Databricks Marketplace** – models and data shared by providers.
# MAGIC - **Hugging Face Hub** – the largest open model hub; read the card's license and limitations carefully.
# MAGIC
# MAGIC **Exam takeaways:** use model-card metadata – **task, license, languages, size, context length, training data,
# MAGIC evaluation results, intended use and limitations** – to filter on must-haves first, rank on task-relevant metrics
# MAGIC second, and then confirm with your own evaluation.
