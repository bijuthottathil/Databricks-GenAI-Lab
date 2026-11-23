# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.2 – Control access to resources from model serving endpoints
# MAGIC
# MAGIC **The big idea for juniors:** when your chain or agent runs on a **model serving endpoint**, it is no longer running
# MAGIC as *you* in a notebook. It runs in a separate container. So two questions must be answered:
# MAGIC
# MAGIC 1. **Who may call the endpoint?** (people / apps → endpoint) → **endpoint permissions**
# MAGIC 2. **What may the endpoint itself access?** (endpoint → LLM endpoints, Vector Search indexes, tables, functions,
# MAGIC    Genie spaces, external APIs) → **authentication for dependent resources**
# MAGIC
# MAGIC ```
# MAGIC  users / apps ──(1) CAN_QUERY?──► [ your serving endpoint ] ──(2) credentials?──► LLM endpoint
# MAGIC                                                                               ├─► Vector Search index
# MAGIC                                                                               ├─► UC table / function
# MAGIC                                                                               └─► external API (secret key)
# MAGIC ```
# MAGIC
# MAGIC ### The three ways an endpoint gets access to resources
# MAGIC | Method | How it works | Use when |
# MAGIC |---|---|---|
# MAGIC | **Automatic authentication passthrough** | You list the resources when **logging** the model (`resources=[...]`). At deploy time Databricks creates a short-lived **service principal** credential with access to *only* those resources | Default choice for Databricks resources |
# MAGIC | **On-behalf-of-user (OBO) auth** | The endpoint acts **as the user calling it**, so Unity Catalog permissions of *that user* apply | Each user must only see data they're allowed to see |
# MAGIC | **Manual authentication** | You store a token/API key in **Databricks Secrets** and pass it to the endpoint as an **environment variable** | External APIs, or resources not supported by passthrough |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Service principal** | A "robot user" for applications – not a person |
# MAGIC | **Resource** | Something the endpoint needs: another endpoint, an index, a table, a function… |
# MAGIC | **Least privilege** | Give access to *only* what is needed, nothing more |
# MAGIC | **Secret scope** | A safe place in Databricks to store passwords/keys; values are never shown in plain text |
# MAGIC | **ACL** | Access Control List – who has which permission on an object |
# MAGIC | **GRANT** | SQL command that gives a Unity Catalog permission (e.g. `SELECT` on a table) |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0  install MLflow 3
# MAGIC Step 1  create a small table the agent would read  → a UC resource
# MAGIC Step 2  declare resources (automatic passthrough) → log a tiny model with them → read them back
# MAGIC Step 3  manual auth with secrets (list scopes, show env-var pattern)
# MAGIC Step 4  on-behalf-of-user auth (shown)
# MAGIC Step 5  who can call the endpoint: endpoint permission levels (read-only check)
# MAGIC Step 6  Unity Catalog grants on the data (read-only check)
# MAGIC ```
# MAGIC Nothing in this notebook changes permissions or creates endpoints – changes are shown as code to copy.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – A table the agent would read
# MAGIC **What:** a small `order_status` table in `workspace.fde_deploy`. In Step 2 we'll declare it as a resource.

# COMMAND ----------

CATALOG, SCHEMA = "workspace", "fde_deploy"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.createDataFrame([(1001, "Delivered"), (1002, "Shipped"), (1003, "Processing")], "order_id int, status string") \
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.order_status")
display(spark.table(f"{CATALOG}.{SCHEMA}.order_status"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Automatic authentication passthrough: declare `resources` when logging
# MAGIC **What:** MLflow has one class per resource type. You list every Databricks resource the model uses:
# MAGIC
# MAGIC | Class | Resource |
# MAGIC |---|---|
# MAGIC | `DatabricksServingEndpoint` | an LLM or embedding endpoint (e.g. Foundation Model APIs) |
# MAGIC | `DatabricksVectorSearchIndex` | a Vector Search index |
# MAGIC | `DatabricksTable` | a Unity Catalog table |
# MAGIC | `DatabricksFunction` | a Unity Catalog function used as a tool |
# MAGIC | `DatabricksSQLWarehouse` | a SQL warehouse |
# MAGIC | `DatabricksGenieSpace` | a Genie space |
# MAGIC
# MAGIC **Why:** when the model is deployed, Databricks checks that **you** (the deployer) have access to each resource and
# MAGIC then gives the endpoint a credential limited to exactly those resources (least privilege). **Forget one → the
# MAGIC deployed endpoint gets "permission denied"**, even though it worked in your notebook.
# MAGIC
# MAGIC **How to read the output:** the resources stored inside the logged model – this is what deployment reads.

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksTable

class OrderStatusBot(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        from mlflow.deployments import get_deploy_client
        answers = []
        for q in model_input["question"]:
            r = get_deploy_client("databricks").predict(
                endpoint="databricks-meta-llama-3-3-70b-instruct",
                inputs={"messages": [{"role": "user", "content": q}], "max_tokens": 60})
            answers.append(r["choices"][0]["message"]["content"])
        return answers

resources = [
    DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.order_status"),
    # Other examples (uncomment when your model uses them):
    # DatabricksVectorSearchIndex(index_name="workspace.fde_deploy.policy_index"),
    # DatabricksFunction(function_name="workspace.fde_deploy.get_order"),
    # DatabricksGenieSpace(genie_space_id="<space-id>"),
]

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_section4_access_control")
with mlflow.start_run(run_name="order_bot_with_resources"):
    logged = mlflow.pyfunc.log_model(name="order_bot", python_model=OrderStatusBot(),
                                     input_example=pd.DataFrame({"question": ["Where is order 1002?"]}),
                                     resources=resources, pip_requirements=["mlflow>=3.1"])

from mlflow.models import Model

saved = Model.load(logged.model_uri)            # reads the MLmodel file of the logged model
print("Resources recorded in the model:")
print(saved.resources)                          # e.g. {'api_version': '1', 'databricks': {'serving_endpoint': [...], 'table': [...]}}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Manual authentication with Databricks Secrets
# MAGIC **What:** for an **external** API (e.g. a shipping-tracking service), store its key in a secret scope and reference
# MAGIC it in the endpoint's **environment variables** with `{{secrets/<scope>/<key>}}`. The real key is injected when the
# MAGIC container starts – it never appears in code, notebooks or logs.
# MAGIC
# MAGIC **Create a secret (CLI, run once by an admin):**
# MAGIC ```bash
# MAGIC databricks secrets create-scope fde_scope
# MAGIC databricks secrets put-secret fde_scope tracking_api_key
# MAGIC ```
# MAGIC **Reference it in the endpoint config (shown, not run):**
# MAGIC ```python
# MAGIC from databricks.sdk.service.serving import ServedEntityInput
# MAGIC ServedEntityInput(
# MAGIC     entity_name="workspace.fde_deploy.support_chain", entity_version="1",
# MAGIC     workload_size="Small", scale_to_zero_enabled=True,
# MAGIC     environment_vars={"TRACKING_API_KEY": "{{secrets/fde_scope/tracking_api_key}}"},
# MAGIC )
# MAGIC # inside the model:  api_key = os.environ["TRACKING_API_KEY"]
# MAGIC ```
# MAGIC **The cell below** only lists the secret scopes you can see (read-only) – it never prints secret values.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
try:
    scopes = [s.name for s in w.secrets.list_scopes()]
    print("Secret scopes visible to you:", scopes or "(none yet)")
except Exception as e:
    print("Could not list secret scopes:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – On-behalf-of-user (OBO) authentication (shown, not run)
# MAGIC **What:** instead of one service-principal identity for everyone, the endpoint uses the **calling user's**
# MAGIC identity for selected resources. User A sees only rows/tables A may see; user B only what B may see.
# MAGIC
# MAGIC **How (MLflow 3 – API names can vary by version, check the docs):**
# MAGIC ```python
# MAGIC from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
# MAGIC auth_policy = AuthPolicy(
# MAGIC     system_auth_policy=SystemAuthPolicy(resources=[DatabricksServingEndpoint("databricks-meta-llama-3-3-70b-instruct")]),
# MAGIC     user_auth_policy=UserAuthPolicy(api_scopes=["sql.statement-execution", "vectorsearch.vector-search-indexes"]),
# MAGIC )
# MAGIC mlflow.pyfunc.log_model(..., auth_policy=auth_policy)
# MAGIC # inside predict(): WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
# MAGIC ```
# MAGIC **Rule of thumb:** shared reference data (policies, product catalog) → automatic passthrough.
# MAGIC Per-user data (a customer's own orders, HR records) → OBO.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Who may call the endpoint? Endpoint permission levels
# MAGIC | Level | Can do |
# MAGIC |---|---|
# MAGIC | `CAN_VIEW` | see the endpoint and its config |
# MAGIC | `CAN_QUERY` | send requests (what an app's service principal needs) |
# MAGIC | `CAN_MANAGE` | edit, delete, change permissions |
# MAGIC
# MAGIC **Grant query access (shown, not run – it changes permissions):**
# MAGIC ```python
# MAGIC from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel
# MAGIC ep = w.serving_endpoints.get("my-support-endpoint")
# MAGIC w.serving_endpoints.update_permissions(ep.id, access_control_list=[
# MAGIC     ServingEndpointAccessControlRequest(group_name="support-app-users",
# MAGIC                                         permission_level=ServingEndpointPermissionLevel.CAN_QUERY)])
# MAGIC ```
# MAGIC **The cell below** reads the permissions of a serving endpoint you can see (read-only). System Foundation Model
# MAGIC endpoints may not expose permissions – that's fine, the message tells you.

# COMMAND ----------

ep = w.serving_endpoints.get("databricks-meta-llama-3-3-70b-instruct")
print("Endpoint:", ep.name, "| id:", ep.id)
try:
    perms = w.serving_endpoints.get_permissions(ep.id)
    for acl in perms.access_control_list or []:
        who = acl.user_name or acl.group_name or acl.service_principal_name
        print(f"  {who}: {[p.permission_level.value for p in acl.all_permissions or []]}")
except Exception as e:
    print("Permissions not readable for this endpoint:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Unity Catalog permissions on the data
# MAGIC **What:** even with endpoint access, the identity used (service principal or user) still needs UC privileges on
# MAGIC the data: `USE CATALOG`, `USE SCHEMA`, `SELECT` on tables, `EXECUTE` on functions and models.
# MAGIC
# MAGIC **Grant example (shown, not run):**
# MAGIC ```sql
# MAGIC GRANT USE CATALOG ON CATALOG workspace TO `support-app-sp`;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  workspace.fde_deploy TO `support-app-sp`;
# MAGIC GRANT SELECT      ON TABLE   workspace.fde_deploy.order_status TO `support-app-sp`;
# MAGIC ```
# MAGIC **How to read the output:** the current grants on our table – probably just you as the owner.

# COMMAND ----------

display(spark.sql(f"SHOW GRANTS ON TABLE {CATALOG}.{SCHEMA}.order_status"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Two layers: **who can query the endpoint** (CAN_QUERY / CAN_MANAGE) and **what the endpoint can access**.
# MAGIC - Declare Databricks dependencies as **`resources`** at `log_model` time → **automatic authentication passthrough**
# MAGIC   with a least-privilege service principal.
# MAGIC - Use **on-behalf-of-user** auth when results must respect each caller's own permissions.
# MAGIC - Use **Databricks Secrets** + `{{secrets/scope/key}}` environment variables for external credentials – never hard-code keys.
# MAGIC - Unity Catalog **GRANTs** still apply to the identity used. AI Gateway adds rate limits and usage tracking on top.
