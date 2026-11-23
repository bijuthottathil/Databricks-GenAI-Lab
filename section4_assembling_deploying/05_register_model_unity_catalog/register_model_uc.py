# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.5 – Register the model to Unity Catalog using MLflow
# MAGIC
# MAGIC **The big idea for juniors:** logging a model (4.1, 4.4) saves it inside one *experiment run*. **Registering** it
# MAGIC gives it a permanent, governed name in **Unity Catalog** – `catalog.schema.model_name` – with numbered
# MAGIC **versions**, **aliases** like `champion`, permissions and lineage. Serving endpoints deploy registered models.
# MAGIC
# MAGIC **Analogy:** logging = saving a draft on your laptop. Registering = publishing it in the company library with a
# MAGIC catalogue number, an edition (version) and a "current recommended edition" sticker (alias).
# MAGIC
# MAGIC ```
# MAGIC  run ──log_model──► models:/m-abc (logged model)
# MAGIC                          │ register
# MAGIC                          ▼
# MAGIC  workspace.fde_deploy.ticket_router   version 1  ◄── alias "champion"
# MAGIC                                       version 2  ◄── alias "challenger"
# MAGIC ```
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Registry URI `databricks-uc`** | Tells MLflow to register into Unity Catalog (not the old workspace registry) |
# MAGIC | **Three-level name** | `catalog.schema.model` – same naming as tables |
# MAGIC | **Version** | A new number each time you register (1, 2, 3…) – old versions are kept |
# MAGIC | **Alias** | A movable label pointing to one version (`champion`, `challenger`, `prod`) |
# MAGIC | **`models:/name@alias`** | Load "whatever version the alias points to" – code never needs to change |
# MAGIC | **Signature** | Required by Unity Catalog – registration fails without it |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0 install → Step 1 set registry to UC → Step 2 TicketRouter model (v1 rules, v2 LLM)
# MAGIC Step 3 log + register v1 → Step 4 log + register v2 → Step 5 aliases champion/challenger
# MAGIC Step 6 load by alias and compare → Step 7 list versions → Step 8 promote + permissions
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Point MLflow at Unity Catalog
# MAGIC **What:** `mlflow.set_registry_uri("databricks-uc")` and create the schema that will hold the model.
# MAGIC **Why:** a UC model lives in a schema, like a table – so permissions and lineage work the same way.

# COMMAND ----------

import mlflow
import pandas as pd

CATALOG, SCHEMA = "workspace", "fde_deploy"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.ticket_router"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
mlflow.set_registry_uri("databricks-uc")
me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_section4_register_uc")
print("Registering into:", MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The model: a ticket router with two strategies
# MAGIC **What:** one pyfunc class. `model_config["strategy"]` decides how it works:
# MAGIC - `"keywords"` (v1) – simple rules, no LLM, instant and free.
# MAGIC - `"llm"` (v2) – asks the LLM, handles wording the rules don't know.
# MAGIC
# MAGIC **Why one class with config:** two *versions* of the same model with different settings – a realistic
# MAGIC upgrade path you'll compare in Step 6.

# COMMAND ----------

class TicketRouter(mlflow.pyfunc.PythonModel):
    CATEGORIES = ["Shipping", "Payment", "Product Defect", "Return", "Other"]
    KEYWORDS = {"Shipping": ["late", "deliver", "package", "tracking"], "Payment": ["charge", "billed", "refund"],
                "Product Defect": ["broken", "defect", "stopped working", "flicker"], "Return": ["return", "send back"]}

    def load_context(self, context):
        self.strategy = (context.model_config or {}).get("strategy", "keywords")

    def route(self, text):
        if self.strategy == "keywords":
            low = text.lower()
            return next((c for c, words in self.KEYWORDS.items() if any(w in low for w in words)), "Other")
        from mlflow.deployments import get_deploy_client
        r = get_deploy_client("databricks").predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
            "messages": [{"role": "user", "content": f"Classify into one of {self.CATEGORIES}. Reply with the category only.\n\n{text}"}],
            "temperature": 0.0, "max_tokens": 10})
        answer = r["choices"][0]["message"]["content"]
        return next((c for c in self.CATEGORIES if c.lower() in answer.lower()), "Other")

    def predict(self, context, model_input, params=None):
        return pd.DataFrame({"category": [self.route(t) for t in model_input["ticket"]]})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Log **and register** version 1
# MAGIC **What:** `registered_model_name=MODEL_NAME` in `log_model` logs and registers in one go. The `signature`
# MAGIC (from `infer_signature`) is mandatory for Unity Catalog.
# MAGIC **How to read the output:** `Registered version: 1` (or a higher number if you've run this notebook before –
# MAGIC every registration adds a version).

# COMMAND ----------

from mlflow.models import infer_signature

example = pd.DataFrame({"ticket": ["My package is a week late."]})
signature = infer_signature(example, pd.DataFrame({"category": ["Shipping"]}))

with mlflow.start_run(run_name="router_keywords"):
    v1 = mlflow.pyfunc.log_model(name="ticket_router", python_model=TicketRouter(),
                                 model_config={"strategy": "keywords"}, signature=signature, input_example=example,
                                 pip_requirements=["mlflow>=3.1", "pandas"], registered_model_name=MODEL_NAME)
print("Registered version:", v1.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Register version 2 (LLM strategy)
# MAGIC **What:** same name → Unity Catalog adds **version 2**; version 1 stays untouched.
# MAGIC Alternative way to register an already-logged model: `mlflow.register_model(model_uri, MODEL_NAME)`.

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

with mlflow.start_run(run_name="router_llm"):
    v2 = mlflow.pyfunc.log_model(name="ticket_router", python_model=TicketRouter(),
                                 model_config={"strategy": "llm"}, signature=signature, input_example=example,
                                 pip_requirements=["mlflow>=3.1", "pandas"],
                                 resources=[DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct")],
                                 registered_model_name=MODEL_NAME)
print("Registered version:", v2.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Aliases: `champion` (in production) and `challenger` (being tested)
# MAGIC **Why aliases instead of version numbers:** apps load `models:/...@champion`. To release a new version you just
# MAGIC **move the alias** – no code change, and rolling back is moving it back.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(MODEL_NAME, "champion", v1.registered_model_version)
client.set_registered_model_alias(MODEL_NAME, "challenger", v2.registered_model_version)
client.set_model_version_tag(MODEL_NAME, v2.registered_model_version, "strategy", "llm")
print("champion   ->", client.get_model_version_by_alias(MODEL_NAME, "champion").version)
print("challenger ->", client.get_model_version_by_alias(MODEL_NAME, "challenger").version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Load by alias and compare
# MAGIC **What:** load both aliases and run the same tickets. The last two tickets use wording the keyword rules don't know.
# MAGIC **How to read the output:** the keyword model (champion) puts the tricky tickets in `Other`; the LLM model
# MAGIC (challenger) should get them right (`Shipping`, `Payment`).

# COMMAND ----------

tests = pd.DataFrame({"ticket": ["The screen is broken.", "Please refund the duplicate charge.",
                                 "Where on earth is my parcel? It's been ages.", "You took my money two times!"],
                      "expected": ["Product Defect", "Payment", "Shipping", "Payment"]})

champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
challenger = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@challenger")
tests["champion"] = champion.predict(tests[["ticket"]])["category"]
tests["challenger"] = challenger.predict(tests[["ticket"]])["category"]
display(tests)
print("champion accuracy:  ", (tests.champion == tests.expected).mean())
print("challenger accuracy:", (tests.challenger == tests.expected).mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – See all versions
# MAGIC **How to read the output:** one row per version with its run and status. You can also open **Catalog →
# MAGIC workspace → fde_deploy → Models → ticket_router** to see versions, aliases and lineage in the UI.

# COMMAND ----------

# Unity Catalog's search_model_versions does not reliably fill in v.aliases, so read the aliases from the
# registered model instead: it gives a dict like {"champion": "1", "challenger": "2"}.
alias_map = client.get_registered_model(MODEL_NAME).aliases or {}
if not isinstance(alias_map, dict):                     # some MLflow versions return a list of alias objects
    alias_map = {a.alias: a.version for a in alias_map}
aliases_by_version = {}
for alias, version in alias_map.items():
    aliases_by_version.setdefault(str(version), []).append(alias)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
display(spark.createDataFrame(
    [(int(v.version), v.run_id, str(v.status), ", ".join(sorted(aliases_by_version.get(str(v.version), []))))
     for v in versions],
    "version int, run_id string, status string, aliases string").orderBy("version"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 – Promote the challenger (only if it's better) + permissions
# MAGIC **What:** a simple rule – if the challenger is more accurate, move `champion` to it. Rollback = move it back.
# MAGIC
# MAGIC **Permissions (shown, not run):** UC models use the same GRANT system as tables. `EXECUTE` lets someone load/serve
# MAGIC the model.
# MAGIC ```sql
# MAGIC GRANT EXECUTE ON MODEL workspace.fde_deploy.ticket_router TO `support-app-sp`;
# MAGIC ```

# COMMAND ----------

if (tests.challenger == tests.expected).mean() > (tests.champion == tests.expected).mean():
    client.set_registered_model_alias(MODEL_NAME, "champion", v2.registered_model_version)
    print(f"Promoted version {v2.registered_model_version} to champion")
else:
    print("Challenger is not better – champion stays on version", v1.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - `mlflow.set_registry_uri("databricks-uc")`, then `log_model(..., registered_model_name="catalog.schema.model")`
# MAGIC   or `mlflow.register_model(uri, name)`.
# MAGIC - A **signature is required** for Unity Catalog models.
# MAGIC - Each registration creates a new **version**; use **aliases** (`champion`, `challenger`) and load with
# MAGIC   `models:/catalog.schema.model@alias` – promotion and rollback = moving the alias.
# MAGIC - UC gives models **permissions (GRANT EXECUTE)**, lineage and discoverability, like tables.
