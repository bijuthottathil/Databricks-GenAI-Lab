# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.12 – Apply CI/CD best practices
# MAGIC ### updating a Vector Search index · promoting prompts across environments · testing individual agent components
# MAGIC
# MAGIC **The big idea for juniors:** **CI/CD** (Continuous Integration / Continuous Delivery) means every change goes
# MAGIC through the **same automatic steps** – test → deploy to dev → check → promote to prod – instead of someone
# MAGIC editing production by hand. For GenAI apps, the things that change are not only code, but also **prompts**,
# MAGIC **knowledge (index contents)** and **models**.
# MAGIC
# MAGIC ```
# MAGIC  git push ─► CI: unit tests (tools, parsers, pre/post-processing)  ─► fail? stop.
# MAGIC              │
# MAGIC              ▼
# MAGIC   deploy to DEV (bundle)  ─► evaluation on eval set ─► pass? ─► promote to PROD
# MAGIC   (code, prompt @dev, index sync)                               (move prompt alias, model alias, bundle -t prod)
# MAGIC ```
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Environment** | A separate copy of the app: `dev` (try things), `staging`, `prod` (real users) |
# MAGIC | **Unit test** | A small automatic check of **one** component, e.g. "does the parser handle bad JSON?" |
# MAGIC | **Mock** | A fake replacement (e.g. a fake LLM) so tests are fast, free and repeatable |
# MAGIC | **Prompt Registry** | MLflow's versioned store for prompts (in Unity Catalog), with aliases like `dev` / `prod` |
# MAGIC | **Promotion** | Moving an alias (`prod`) to a newer version **after** it passes tests |
# MAGIC | **Databricks Asset Bundle** | `databricks.yml` – code + jobs + endpoints defined as files, deployed per target (`-t dev`, `-t prod`) |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Part A  test individual components: tools, parser, preprocessing, agent routing with a MOCK LLM
# MAGIC Part B  prompts: register v1 → alias dev+prod; register v2 → dev; test v2 → promote prod; load by environment
# MAGIC Part C  Vector Search: change the SOURCE TABLE → sync the index → verify with a test query
# MAGIC Part D  wire it into a pipeline: bundle targets + CI workflow (shown)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1" databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A – Test individual components of an agent
# MAGIC ### Step A1 – The components under test
# MAGIC **What:** the pieces of a small support agent:
# MAGIC - `get_order(order_id)` – a **tool**,
# MAGIC - `clean_input(text)` – **pre-processing**,
# MAGIC - `parse_decision(raw)` – **output parser**,
# MAGIC - `agent_step(message, llm)` – the **agent logic**; it takes the LLM *as a parameter*, so tests can pass a fake one.
# MAGIC
# MAGIC **Why pass the LLM in:** this is called **dependency injection**. Real runs pass the real LLM; tests pass a mock that
# MAGIC returns a fixed answer – so tests run in milliseconds, cost nothing and never change.

# COMMAND ----------

import json, re

ORDERS = {"1001": {"status": "Delivered"}, "1002": {"status": "Shipped", "days_late": 5}}

def get_order(order_id):
    return ORDERS.get(str(order_id), {"error": "not found"})

def clean_input(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)[:1000]

def parse_decision(raw):
    """Expect {"tool": "get_order"|"none", "order_id": "..."}; anything else → safe default."""
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {"tool": d.get("tool", "none"), "order_id": str(d.get("order_id", ""))}
    except Exception:
        return {"tool": "none", "order_id": ""}

def agent_step(message, llm):
    decision = parse_decision(llm(clean_input(message)))
    if decision["tool"] == "get_order":
        return {"used_tool": True, "result": get_order(decision["order_id"])}
    return {"used_tool": False, "result": None}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A2 – Unit tests (each test checks one behaviour)
# MAGIC **What:** plain `assert` statements grouped in small functions, and a tiny runner that prints PASS/FAIL.
# MAGIC In a repo you would put these in `tests/test_agent.py` and run `pytest` in CI – same idea.
# MAGIC
# MAGIC **How to read the output:** every line should say `PASS`. If you break a component (try changing `parse_decision`),
# MAGIC the matching test turns `FAIL` – and CI would stop the deployment.

# COMMAND ----------

def test_tool_known_order():
    assert get_order("1002")["status"] == "Shipped"

def test_tool_unknown_order():
    assert "error" in get_order("9999")

def test_clean_input_masks_email_and_spaces():
    assert clean_input("  mail   me at a.b@x.com ") == "mail me at [EMAIL]"

def test_parser_valid_json():
    assert parse_decision('Sure! {"tool": "get_order", "order_id": 1002}') == {"tool": "get_order", "order_id": "1002"}

def test_parser_bad_json_is_safe():
    assert parse_decision("I think you should check the order") == {"tool": "none", "order_id": ""}

def test_agent_calls_tool_with_mock_llm():
    mock_llm = lambda prompt: '{"tool": "get_order", "order_id": "1002"}'      # fake LLM: always asks for the tool
    out = agent_step("where is order 1002?", mock_llm)
    assert out["used_tool"] and out["result"]["days_late"] == 5

def test_agent_no_tool_for_greeting():
    mock_llm = lambda prompt: '{"tool": "none"}'
    assert agent_step("hello!", mock_llm)["used_tool"] is False

tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
failures = 0
for t in tests:
    try:
        t(); print("PASS", t.__name__)
    except AssertionError:
        failures += 1; print("FAIL", t.__name__)
print(f"\n{len(tests) - failures}/{len(tests)} tests passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A3 – One integration test with the real LLM
# MAGIC **What:** unit tests prove each part works alone; an **integration test** checks the parts work together with the
# MAGIC real model. It's slower and costs tokens, so run a *few* of these, not hundreds (quality at scale is the job of
# MAGIC evaluation – 3.10, 3.11).
# MAGIC **How to read the output:** `used_tool: True` with order 1002's status.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

def real_llm(prompt):
    r = get_deploy_client("databricks").predict(endpoint="databricks-meta-llama-3-3-70b-instruct", inputs={
        "messages": [{"role": "system", "content": 'Decide if the order tool is needed. Return ONLY JSON '
                                                   '{"tool": "get_order" or "none", "order_id": "<id or empty>"}'},
                     {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 40})
    return r["choices"][0]["message"]["content"]

result = agent_step("Hi, where is my order 1002?", real_llm)
print(result)
assert result["used_tool"], "integration test failed: agent did not call the tool"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B – Promote prompts across environments
# MAGIC ### Step B1 – Register prompt versions in the MLflow Prompt Registry
# MAGIC **What:** prompts are stored in Unity Catalog as `catalog.schema.prompt_name`, with **versions** (1, 2, …) and
# MAGIC **aliases** (`dev`, `prod`). `{{question}}` is a template variable filled at run time.
# MAGIC **Why:** changing a prompt can change behaviour as much as changing code – so it gets versions, review and
# MAGIC promotion too. Apps load `@prod`; you never edit the prod prompt in place.
# MAGIC
# MAGIC **How to read the output:** two version numbers and which version each alias points to: `dev → v2`, `prod → v1`.

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
PROMPT_NAME = "workspace.fde_deploy.support_prompt"
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.fde_deploy")

v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template="You are a support agent. Answer the question: {{question}}",
    commit_message="v1: basic prompt")
v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=("You are ShopEase support. Policy: returns within 30 days; late orders (>3 days) get a $10 credit. "
              "Answer in at most 2 sentences, only from the policy. Question: {{question}}"),
    commit_message="v2: add policy grounding and length limit")

mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="prod", version=v1.version)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="dev", version=v2.version)
print(f"registered v{v1.version} and v{v2.version}; dev → v{v2.version}, prod → v{v1.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B2 – The app loads the prompt for *its* environment
# MAGIC **What:** the widget `env` (dev or prod) decides which alias is loaded – the **same code** runs in both
# MAGIC environments. In a bundle, the job passes `env` as a parameter per target.
# MAGIC **How to read the output:** the template and answer for the chosen environment.

# COMMAND ----------

dbutils.widgets.dropdown("env", "dev", ["dev", "prod"])
ENV = dbutils.widgets.get("env")

def answer(question, env):
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{env}")
    text = prompt.format(question=question)
    r = get_deploy_client("databricks").predict(endpoint="databricks-meta-llama-3-3-70b-instruct",
            inputs={"messages": [{"role": "user", "content": text}], "temperature": 0.0, "max_tokens": 120})
    return prompt.version, r["choices"][0]["message"]["content"]

version, reply = answer("My order is 5 days late. What do I get?", ENV)
print(f"[{ENV}] prompt v{version}: {reply}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B3 – Gate: test the dev prompt, then promote it to prod
# MAGIC **What:** run a few test questions against `@dev`; each answer must contain a required fact. Only if **all** pass,
# MAGIC move the `prod` alias to the dev version. Rollback = point `prod` back to the old version.
# MAGIC **How to read the output:** PASS/FAIL per question and a promotion decision.

# COMMAND ----------

prompt_tests = [("My order is 5 days late. What do I get?", "$10"),
                ("How long do I have to return an item?", "30 days")]
results = []
for q, must_contain in prompt_tests:
    _, reply = answer(q, "dev")
    ok = must_contain.lower() in reply.lower()
    results.append(ok)
    print("PASS" if ok else "FAIL", "|", q, "→", reply[:100])

dev_version = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@dev").version
if all(results):
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="prod", version=dev_version)
    print(f"\nPromoted prompt v{dev_version} to prod")
else:
    print("\nNot promoted – prod stays on the previous version")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C – Update a Vector Search index the CI/CD way
# MAGIC **Rule:** never edit an index by hand. Change the **source Delta table** (from a reviewed pipeline/job), then
# MAGIC **sync** the index, then run a **retrieval test** that proves the new content is found.
# MAGIC
# MAGIC **This part reuses the index from 4.6** (`workspace.fde_deploy.policy_chunks_index`). If it doesn't exist, the cell
# MAGIC prints a message and skips – nothing is created here.
# MAGIC **How to read the output:** the merged row, then the test query's top result should be `pay-3`.

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

VS_ENDPOINT, INDEX_NAME = "fde_vs_endpoint", "workspace.fde_deploy.policy_chunks_index"
SOURCE_TABLE = "workspace.fde_deploy.policy_chunks"
vsc = VectorSearchClient(disable_notice=True)

try:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
    index.describe()
except Exception:
    index = None
    print("Index from 4.6 not found – run 4.6 first to try this part. Skipping.")

if index:
    # 1. change the SOURCE table (in real CI/CD this runs as a reviewed job)
    spark.sql(f"""
    MERGE INTO {SOURCE_TABLE} t
    USING (SELECT 'pay-3' AS chunk_id, 'payments' AS category, 'payments.md' AS source,
                  'Buy-now-pay-later is available for orders over $200.' AS chunk_text) s
    ON t.chunk_id = s.chunk_id
    WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")
    # 2. sync the index (TRIGGERED pipeline)
    index.sync()
    time.sleep(60)
    # 3. retrieval test – fail the pipeline if the new content isn't found
    res = index.similarity_search(query_text="Can I pay later in instalments?", columns=["chunk_id", "chunk_text"], num_results=1)
    top = res.get("result", {}).get("data_array", [[None]])[0][0]
    print("top result:", top)
    print("retrieval test:", "PASS" if top == "pay-3" else "FAIL (index may still be syncing – re-run in a minute)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D – Wire it into a pipeline (shown)
# MAGIC ### Bundle targets: one definition, several environments
# MAGIC ```yaml
# MAGIC # databricks.yml
# MAGIC targets:
# MAGIC   dev:
# MAGIC     mode: development
# MAGIC     default: true
# MAGIC     variables: { env: dev, catalog: dev_catalog }
# MAGIC   prod:
# MAGIC     mode: production
# MAGIC     variables: { env: prod, catalog: prod_catalog }
# MAGIC     run_as: { service_principal_name: "<prod-sp-application-id>" }
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     refresh_knowledge:
# MAGIC       tasks:
# MAGIC         - task_key: update_source_table   # MERGE new docs
# MAGIC         - task_key: sync_index            # index.sync()
# MAGIC         - task_key: retrieval_tests       # fail if expected chunks aren't found
# MAGIC ```
# MAGIC ### CI workflow (e.g. GitHub Actions)
# MAGIC ```yaml
# MAGIC on: [pull_request, push]
# MAGIC jobs:
# MAGIC   test-and-deploy:
# MAGIC     steps:
# MAGIC       - run: pip install -r requirements.txt && pytest tests/          # Part A unit tests
# MAGIC       - run: databricks bundle validate -t dev
# MAGIC       - run: databricks bundle deploy -t dev                           # on every merge
# MAGIC       - run: databricks bundle run evaluate_agent -t dev               # eval gate (3.10 / 3.11)
# MAGIC       - run: databricks bundle deploy -t prod                          # only on main, after the gate passes
# MAGIC ```
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - **Test components individually** (tools, parsers, pre/post-processing, routing) with **mocked LLMs**; add a few
# MAGIC   integration tests and an **evaluation gate** before promotion.
# MAGIC - **Prompts are versioned artifacts**: MLflow Prompt Registry in UC, aliases per environment (`@dev`, `@prod`),
# MAGIC   promote by **moving the alias** after tests pass; roll back the same way.
# MAGIC - **Update Vector Search** by changing the **source Delta table** in a pipeline, then `sync()` (TRIGGERED) or let
# MAGIC   CONTINUOUS sync, then verify with retrieval tests.
# MAGIC - Use **Databricks Asset Bundles** with `dev`/`prod` targets (+ service principals) so the same definition is deployed
# MAGIC   everywhere; same idea for models: UC aliases `champion`/`challenger` (4.5).
