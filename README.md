# Databricks GenAI Lab

Hands-on Databricks notebooks for building Generative AI applications, organised by the six sections of the
**Databricks Certified Generative AI Engineer Associate** exam guide, with one folder per exam objective.

Every example uses the same scenario: a **Customer Support AI assistant for an e-commerce company** that classifies
complaints, analyses sentiment, retrieves policies, recommends actions and extracts structured information.
All notebooks run on **serverless compute** and work in **Databricks Free Edition**.

---

## Contents

| Section | Folder | Notebooks | Unity Catalog schema |
|---|---|---|---|
| 1 – Design Applications | `section1_design_applications/` | 7 | `workspace.fde_customer_support` |
| 2 – Data Preparation | `section2_data_preparation/` | 8 | `workspace.fde_data_prep` |
| 3 – Application Development | `section3_application_development/` | 13 | `workspace.fde_app_dev` |
| 4 – Assembling and Deploying Applications | `section4_assembling_deploying/` | 12 | `workspace.fde_deploy` |
| 5 – Governance | `section5_governance/` | 4 | `workspace.fde_governance` |
| 6 – Evaluation and Monitoring | `section6_evaluation_monitoring/` | 10 | `workspace.fde_eval_monitor` |

## How the notebooks are written

- **Self-contained:** Sections 2–6 create their own sample data and need no shared helpers. Section 1 shares one
  config notebook (`00_setup/config`) and one sample-data notebook.
- **Beginner-friendly:** cells include plain-language comments, key-word tables, cell-flow notes and
  "how to read the output" explanations.
- **Direct model calls:** LLMs are called with plain `mlflow.deployments` code (`client.predict(...)`) against
  Foundation Model API endpoints, with no wrapper functions hiding what happens.
- **Safe by default:** cells that create billable or long-running resources, or change endpoints, sit behind flags
  that default to `False` (see [Cost and safety notes](#cost-and-safety-notes)).

---

## Section 1 – Design Applications (`section1_design_applications/`)

| Folder | Objective |
|---|---|
| `00_setup/` | `config` (catalog / schema / model endpoint) and `create_sample_data` (tickets, policies, orders tables + policy docs Volume) |
| `01_prompt_formatted_response/` | Design a prompt that elicits a specifically formatted (JSON) response |
| `02_model_task_selection/` | Select model tasks: classification, sentiment, summarization, extraction, translation, generation, embeddings |
| `03_chain_components/` | LangChain chains: PromptTemplate → LLM → OutputParser, retriever, memory |
| `04_pipeline_inputs_outputs/` | Business goals → pipeline stages, output contract, AI Functions batch pipeline → Delta |
| `05_tools_multistage_reasoning/` | Define and order tools: fixed orchestration vs tool-calling agent |
| `06_agent_bricks/` | Knowledge Assistant, Multiagent Supervisor, Information Extraction – when to use each |

Run `00_setup/create_sample_data` first. It creates the `customer_support_tickets`, `customer_support_policies` and
`customer_orders` tables and the `policy_docs` Volume. Notebooks 01–06 load the shared settings with
`%run ../00_setup/config`.

## Section 2 – Data Preparation (`section2_data_preparation/`)

| Folder | Objective |
|---|---|
| `01_chunking_strategy/` | Chunking for a document structure and model limits (fixed, overlap, recursive, header-based, FAQ) |
| `02_filter_extraneous_content/` | Remove menus, footers, page numbers, duplicates; shows how junk hurts search |
| `03_document_extraction_packages/` | Pick the right package per format: pypdf, python-docx, python-pptx, BeautifulSoup, pandas, json |
| `04_write_chunks_to_delta/` | Volume → parse → chunk → IDs/metadata → Delta table (PK + Change Data Feed) → MERGE |
| `05_identify_source_documents/` | Choose relevant, official, current, unique, PII-free sources; find knowledge gaps |
| `06_evaluate_retrieval/` | Hit rate, recall, precision, MRR, nDCG, LLM judge; logged to MLflow |
| `07_advanced_chunking/` | Contextual headers, semantic chunking, parent–child, sentence window, metadata filters |
| `08_reranking/` | Two-stage retrieval: vector search then LLM re-ranker, with before/after MRR |

## Section 3 – Application Development (`section3_application_development/`)

| Folder | Objective |
|---|---|
| `01_select_langchain_tools/` | Plain SDK vs LangChain chain vs chain with retriever vs SQL `ai_query`; decision guide |
| `02_assess_response_quality_safety/` | Spot hallucination, incompleteness, toxicity, PII, injection with rules + rubric LLM judge |
| `03_select_chunking_by_evaluation/` | Grid of chunking strategies × embedding models, Recall@3/MRR in MLflow, decision rule |
| `04_augment_prompt_with_user_context/` | Extract intent/fields/terms, look up order + policy, build an augmented prompt |
| `05_prompt_baseline_to_desired/` | Prompt v0→v4 (role, context, constraints, format, few-shot) scored against 6 checks |
| `06_llm_guardrails/` | PII masking, injection & topic filters, output rules, grounding check, guarded chat; AI Gateway |
| `07_select_best_llm/` | Measure 4 models (accuracy, JSON, reasoning, latency); weighted choice for 2 different apps |
| `08_embedding_context_length/` | Truncation demo (512 vs 8192 tokens), chunk-to-fit, storage trade-offs |
| `09_model_cards_marketplace/` | Parse a model card, filter on must-haves, rank on metrics; list real endpoints / `system.ai` |
| `10_select_model_by_metrics/` | Accuracy, macro F1, ROUGE-1/L, latency p50/p95 logged to MLflow; quality gate then efficiency |
| `11_mlflow_agent_framework/` | Tool-calling `ResponsesAgent`, tracing, log/load, `mlflow.genai.evaluate`, register/deploy |
| `12_evaluation_vs_monitoring/` | Offline eval + release gate vs 14 days of production logs, alerts, drift, feedback loop |
| `13_multi_agent_genie/` | Genie Conversation API as a data agent + policy agent under a supervisor (stand-in if no space) |

Only notebook 13 writes tables (to `workspace.fde_app_dev`).

## Section 4 – Assembling and Deploying Applications (`section4_assembling_deploying/`)

| Folder | Objective |
|---|---|
| `01_pyfunc_chain_pre_post_processing/` | `PythonModel` chain: clean/mask input → LLM → parse/validate/trim output; `model_config`, log & load |
| `02_serving_endpoint_access_control/` | Automatic auth passthrough (`resources`), OBO auth, secrets env vars, endpoint ACLs, UC grants |
| `03_simple_chain_from_requirements/` | Requirements → components → LCEL chain (translate → summarise/classify → JSON), plain-Python equivalent |
| `04_rag_application_elements/` | Flavor, embedding model, retriever, LLM, dependencies, input example, signature in one RAG pyfunc |
| `05_register_model_unity_catalog/` | Register to UC, versions, `champion`/`challenger` aliases, load by alias, promote |
| `06_create_query_vector_search/` | Delta table (CDF) → VS endpoint → Delta Sync index (managed embeddings) → search, filters, hybrid, sync |
| `07_serve_with_foundation_model_apis/` | Pay-per-token vs provisioned throughput vs external models; call FMAPI 3 ways; deploy your app (flagged) |
| `08_vector_search_concepts/` | Components, index types, ANN vs brute force, filters, hybrid, sync – built in plain Python |
| `09_batch_inference_ai_query/` | Batch vs real-time; `ai_query` classify, JSON output, `failOnError`, embeddings, CTAS, AI functions |
| `10_configure_vector_search/` | Choose endpoint type, index type, sync mode, reranking from vectors / freshness / latency / budget |
| `11_persistent_memory_store/` | Short-term chat memory, long-term facts (MERGE), task state in Delta; Lakebase notes |
| `12_cicd_best_practices/` | Unit tests with mock LLM, Prompt Registry dev→prod promotion, index update + retrieval test, bundle CI |

## Section 5 – Governance (`section5_governance/`)

| Folder | Objective |
|---|---|
| `01_masking_guardrails_performance/` | No mask vs sentence drop vs typed placeholders scored on privacy + accuracy + latency; pseudonymisation, `ai_mask`, UC column masks |
| `02_guardrails_malicious_inputs/` | 9 attacks + 3 benign through size limit, rule filter, LLM classifier; detection matrix, hardened prompt + canary, indirect injection, least privilege |
| `03_legal_licensing_data_sources/` | Licence table, data source register with include/review/exclude rules, UC tags, licence-filtered chunks, attribution by code |
| `04_problematic_text_mitigation/` | Detect toxic/outdated/PII/dangerous/biased/spam posts; mask, detoxify, drop, replace with official source, review queue |

## Section 6 – Evaluation and Monitoring (`section6_evaluation_monitoring/`)

| Folder | Objective |
|---|---|
| `01_select_llm_size_architecture/` | 8B / 12B / MoE 80B-A3B / 70B on classification, JSON extraction, reasoning, latency; quality gate then efficiency |
| `02_key_metrics_by_scenario/` | Scenario features → metric set; 7 days of simulated logs → daily metrics, thresholds, alerts |
| `03_agent_eval_scoring_tracing/` | Traced tool agent, span inspection, `mlflow.genai.evaluate` with built-in judges + trace-based scorers |
| `04_inference_logging_rag/` | Inference table (simulated or real): parse payloads, errors/latency, empty retrieval, groundedness on logged rows |
| `05_control_llm_costs/` | Token usage, prompt/output/model/caching levers, monthly projection, usage + billing system tables, serving modes |
| `06_inference_tables_agent_monitoring/` | Enable inference tables (gated), traced app, registered monitoring scorers (gated), manual search_traces + evaluate |
| `07_judges_requiring_ground_truth/` | Judges with vs without expectations; groundedness ≠ correctness demo; which judges per situation |
| `08_ai_gateway_tracking/` | Audit AI Gateway config, configure inference table/usage/rate limits (gated), usage tables, 429 retry with backoff |
| `09_custom_scorers/` | `@scorer` code, `Feedback`, expectations, trace and LLM-judge scorers, `Guidelines`, `make_judge`, scorer unit tests |
| `10_sme_feedback/` | SME `log_feedback`/`log_expectation` on traces, labeling sessions, eval dataset + guidelines → v2, v1 vs v2 evaluation |

Section 6 also logs to MLflow experiments under `/Users/<you>/fde_section6_*`.

---

## Prerequisites

- A Databricks workspace with Unity Catalog and serverless compute (Free Edition works)
- Access to Foundation Model API endpoints (default: `databricks-meta-llama-3-3-70b-instruct`)
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.218+ to deploy with Asset Bundles
- Optional: a Genie space for `section3/13_multi_agent_genie` (the notebook falls back to a stand-in without one)

## Configuration

| Setting | Default | Where to change |
|---|---|---|
| Catalog | `workspace` | widget, or `section1_design_applications/00_setup/config.py` |
| Schema (Section 1) | `fde_customer_support` | widget, or `config.py` |
| Model endpoint | `databricks-meta-llama-3-3-70b-instruct` | widget, or `config.py` |

Sections 2–6 don't use this config: any catalog, schema or endpoint a notebook needs is set inside that notebook.

## Getting started

### Option A – Asset Bundle (recommended)

1. Edit the `targets.dev.workspace` block in `databricks.yml` (host, profile, root path) to match your workspace.
2. Validate and deploy:

   ```bash
   databricks bundle validate
   databricks bundle deploy            # uploads notebooks to /Users/<you>/databricks-fde-usecases
   ```

3. Run a whole section as a serverless job:

   | Section | Command |
   |---|---|
   | 1 | `databricks bundle run customer_support_ai_usecases` |
   | 2 | `databricks bundle run data_preparation_usecases` |
   | 3 | `databricks bundle run application_development_usecases` |
   | 4 | `databricks bundle run assembling_deploying_usecases` |
   | 5 | `databricks bundle run governance_usecases` |
   | 6 | `databricks bundle run evaluation_monitoring_usecases` |

   You can also open any deployed notebook in the workspace and run it cell by cell.

### Option B – Git folder

Clone this repo into a Databricks Git folder (**Workspace → Create → Git folder**), attach a notebook to serverless
compute and run it. For Section 1, run `00_setup/create_sample_data` first.

## Cost and safety notes

- **4.6 `create_query_vector_search`** creates a Vector Search endpoint that keeps running (and billing) until you
  delete it. Run the cleanup cell at the end of the notebook.
- **4.7 `serve_with_foundation_model_apis`** creates a serving endpoint only if you set `DEPLOY = True`.
- **Section 6** cells that change endpoints are off by default: `ENABLE_INFERENCE_TABLE`, `START_MONITORING` and
  `APPLY` are all `False`.
- Pay-per-token Foundation Model API calls are billed per token. The sample datasets are small, so a full run costs little.

## Cleanup

Each section writes only to its own schema, so you can remove a section's data by dropping its schema:

```sql
DROP SCHEMA IF EXISTS workspace.fde_customer_support CASCADE;  -- Section 1
DROP SCHEMA IF EXISTS workspace.fde_data_prep CASCADE;         -- Section 2
DROP SCHEMA IF EXISTS workspace.fde_app_dev CASCADE;           -- Section 3
DROP SCHEMA IF EXISTS workspace.fde_deploy CASCADE;            -- Section 4
DROP SCHEMA IF EXISTS workspace.fde_governance CASCADE;        -- Section 5
DROP SCHEMA IF EXISTS workspace.fde_eval_monitor CASCADE;      -- Section 6
```

Delete any Vector Search or serving endpoints you created from the **Compute → Vector Search** and **Serving** pages,
and remove the deployed bundle with `databricks bundle destroy`.

## Repository layout

```
.
├── databricks.yml                     # Asset Bundle: one serverless job per section
├── section1_design_applications/
├── section2_data_preparation/
├── section3_application_development/
├── section4_assembling_deploying/
├── section5_governance/
└── section6_evaluation_monitoring/
```
