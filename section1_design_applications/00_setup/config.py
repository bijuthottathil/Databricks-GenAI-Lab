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
