# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.7 – Identify how to serve an LLM application that leverages Foundation Model APIs
# MAGIC
# MAGIC **The big idea for juniors:** you usually **don't host the LLM yourself**. Databricks already serves popular models
# MAGIC (Llama, Gemma, GTE embeddings…) through the **Foundation Model APIs (FMAPI)**. *Your* application – the chain or
# MAGIC agent with your prompts, retrieval and logic – is served on **its own** endpoint and **calls** the FMAPI endpoint.
# MAGIC
# MAGIC ```
# MAGIC  client app ──► [ YOUR endpoint: workspace.fde_deploy.support_chain ] ──► [ FMAPI endpoint: databricks-meta-llama-3-3-70b-instruct ]
# MAGIC                 custom model (pyfunc / agent), your code, CPU               Databricks-hosted LLM on GPUs
# MAGIC ```
# MAGIC
# MAGIC ### Ways to use Foundation Models
# MAGIC | Option | What it is | When to choose |
# MAGIC |---|---|---|
# MAGIC | **Pay-per-token** | Shared, ready-made endpoints named `databricks-...`; pay for tokens used | Prototyping, low/medium or spiky traffic |
# MAGIC | **Provisioned throughput** | A dedicated endpoint with guaranteed capacity (tokens/second); pay per hour | Production with steady high volume, latency SLAs, fine-tuned models |
# MAGIC | **External models** | A Databricks endpoint that forwards to OpenAI, Anthropic, etc. | You need a third-party model but want one governed gateway |
# MAGIC | **AI Gateway** (on top of any) | Rate limits, usage tracking, guardrails, inference tables | Governance and monitoring |
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Serving endpoint** | A REST URL that runs a model and answers requests |
# MAGIC | **Task** | The endpoint's API type: `llm/v1/chat`, `llm/v1/embeddings`, `llm/v1/completions` |
# MAGIC | **OpenAI-compatible** | FMAPI accepts the same request format as OpenAI, so OpenAI client code works |
# MAGIC | **Custom model endpoint** | An endpoint serving *your* registered MLflow model |
# MAGIC | **Scale to zero** | The endpoint shuts down when idle (cheaper; first request is slower) |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 1 list FMAPI endpoints in this workspace (read-only)
# MAGIC Step 2 call one 3 ways: MLflow deployments client / OpenAI client / REST
# MAGIC Step 3 the app pattern: your chain calls FMAPI (local demo)
# MAGIC Step 4 deploy your app to its own endpoint (shown; DEPLOY flag off)
# MAGIC Step 5 query your deployed endpoint (shown)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U openai "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Which Foundation Model endpoints can I use?
# MAGIC **How to read the output:** endpoint names with their task. Chat endpoints (`llm/v1/chat`) generate text;
# MAGIC embeddings endpoints (`llm/v1/embeddings`) produce vectors.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
rows = [(ep.name, ep.task, str(ep.state.ready) if ep.state else None)
        for ep in w.serving_endpoints.list() if ep.name.startswith("databricks-")]
display(spark.createDataFrame(rows, "endpoint string, task string, ready string").orderBy("task", "endpoint"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Three ways to call an FMAPI endpoint
# MAGIC **Why know all three:** the request is the same chat format each time – only the client differs.
# MAGIC | Way | Typical use |
# MAGIC |---|---|
# MAGIC | (a) `mlflow.deployments` client | notebooks, pyfunc models (used throughout these samples) |
# MAGIC | (b) OpenAI client | existing OpenAI code – just change the base URL and token |
# MAGIC | (c) REST `POST /serving-endpoints/<name>/invocations` | any language / tool (curl, Java, JavaScript) |
# MAGIC
# MAGIC **How to read the output:** three short answers to the same question.

# COMMAND ----------

ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
messages = [{"role": "user", "content": "In one sentence: what is a Delta table?"}]

# (a) MLflow deployments client
from mlflow.deployments import get_deploy_client
r = get_deploy_client("databricks").predict(endpoint=ENDPOINT, inputs={"messages": messages, "max_tokens": 60})
print("(a) mlflow :", r["choices"][0]["message"]["content"])

# (b) OpenAI-compatible client (the SDK fills in the workspace URL and your token)
try:
    openai_client = w.serving_endpoints.get_open_ai_client()
    r = openai_client.chat.completions.create(model=ENDPOINT, messages=messages, max_tokens=60)
    print("(b) openai :", r.choices[0].message.content)
except Exception as e:
    print("(b) openai client not available:", str(e)[:150])

# (c) Plain REST call through the SDK's authenticated HTTP client
r = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations", body={"messages": messages, "max_tokens": 60})
print("(c) REST   :", r["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – The application pattern: your logic + FMAPI
# MAGIC **What:** a tiny "app" function – prompt template + business rule + FMAPI call. In 4.1 we packaged exactly this
# MAGIC kind of logic as a pyfunc model; that package is what you deploy.
# MAGIC **Why not deploy the LLM itself:** the LLM is already served. You deploy **your code** (lightweight, CPU) and
# MAGIC declare the FMAPI endpoint as a `resource` so the deployed app is allowed to call it (see 4.2).

# COMMAND ----------

def support_app(question):
    system = ("You are ShopEase support. Policy: returns within 30 days; late orders (>3 days) get a $10 credit. "
              "Answer in at most 2 sentences.")
    r = get_deploy_client("databricks").predict(endpoint=ENDPOINT, inputs={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "temperature": 0.0, "max_tokens": 100})
    return r["choices"][0]["message"]["content"]

print(support_app("My order is 5 days late, what do I get?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Deploy your app to its own serving endpoint (shown – `DEPLOY = False`)
# MAGIC **Why off by default:** a serving endpoint is billed while it runs. Flip `DEPLOY = True` only when you want to try
# MAGIC it (and delete it afterwards). It needs a registered UC model – e.g. `workspace.fde_deploy.ticket_router` from 4.5.
# MAGIC
# MAGIC **Two ways:**
# MAGIC - **Agents** (`ResponsesAgent`, 3.11): `from databricks import agents; agents.deploy(model_name, version)` – also
# MAGIC   creates the Review App, inference tables and monitoring.
# MAGIC - **Any pyfunc model**: `w.serving_endpoints.create(...)` with a `ServedEntityInput` (below).
# MAGIC
# MAGIC **Settings explained:** `workload_size="Small"` (CPU size – your code is light; the GPU work happens in FMAPI),
# MAGIC `scale_to_zero_enabled=True` (stop when idle to save money).

# COMMAND ----------

DEPLOY = False
APP_ENDPOINT = "fde-ticket-router"
MODEL_NAME = "workspace.fde_deploy.ticket_router"

if DEPLOY:
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    from mlflow import MlflowClient
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    w.serving_endpoints.create_and_wait(
        name=APP_ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(
            entity_name=MODEL_NAME, entity_version=version, workload_size="Small", scale_to_zero_enabled=True)]),
    )
    print("Deployed", MODEL_NAME, "version", version, "to", APP_ENDPOINT)
else:
    print("DEPLOY is False – nothing created. Code above shows how.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Query your own endpoint (works after Step 4)
# MAGIC **What:** custom pyfunc endpoints take `dataframe_records` (rows matching the model signature).
# MAGIC ```python
# MAGIC w.api_client.do("POST", f"/serving-endpoints/{APP_ENDPOINT}/invocations",
# MAGIC                 body={"dataframe_records": [{"ticket": "My parcel never arrived"}]})
# MAGIC # → {"predictions": [{"category": "Shipping"}]}
# MAGIC ```
# MAGIC **Cleanup:** `w.serving_endpoints.delete(APP_ENDPOINT)`.
# MAGIC
# MAGIC **Provisioned throughput (shown):** for production volume you create a dedicated FMAPI endpoint for a
# MAGIC `system.ai` model with a throughput range, then point your app at that endpoint name instead of the pay-per-token one.
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - Use **Foundation Model APIs** for the LLM: **pay-per-token** (dev, variable load) or **provisioned throughput**
# MAGIC   (production, guaranteed capacity); **external models** for third-party LLMs.
# MAGIC - Serve **your application** as a **custom model endpoint** (registered UC model) that calls FMAPI; declare the FMAPI
# MAGIC   endpoint in `resources`. For agents use `agents.deploy()`.
# MAGIC - FMAPI is **OpenAI-compatible** – call it with MLflow, the OpenAI client or plain REST.
