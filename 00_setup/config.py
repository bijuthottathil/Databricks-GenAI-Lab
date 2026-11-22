# Databricks notebook source
# MAGIC %md
# MAGIC # Shared config
# MAGIC Run by every use-case notebook via `%run ../00_setup/config`.
# MAGIC Change the catalog / schema / endpoint here (or with the widgets) to fit your workspace.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fde_customer_support")
dbutils.widgets.text("endpoint", "databricks-meta-llama-3-3-70b-instruct")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using {CATALOG}.{SCHEMA} with model endpoint {ENDPOINT}")

# COMMAND ----------

from mlflow.deployments import get_deploy_client

_client = get_deploy_client("databricks")


def ask_llm(prompt: str, system: str | None = None, temperature: float = 0.0, max_tokens: int = 500) -> str:
    """Single call to a Databricks Foundation Model endpoint (no extra libraries needed)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    import time
    for attempt in range(5):  # pay-per-token endpoints are rate limited (HTTP 429): back off and retry
        try:
            resp = _client.predict(
                endpoint=ENDPOINT,
                inputs={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            )
            break
        except Exception as e:
            if "429" not in str(e) and "REQUEST_LIMIT_EXCEEDED" not in str(e) or attempt == 4:
                raise
            time.sleep(2 ** attempt * 3)
    return resp["choices"][0]["message"]["content"].strip()


def parse_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating ```json fences."""
    import json, re
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)
