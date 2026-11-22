# databricks-fde-usecases

Customer Support AI practice examples for Databricks (works in Free Edition), one folder per exam objective.
An e-commerce assistant that classifies complaints, analyzes sentiment, retrieves policies, recommends actions and extracts structured info.

| Folder | Objective |
|---|---|
| `00_setup/` | `config` (catalog / schema / model endpoint) and `create_sample_data` (tickets, policies, orders tables + policy docs Volume) |
| `01_prompt_formatted_response/` | Design a prompt that elicits a specifically formatted (JSON) response |
| `02_model_task_selection/` | Select model tasks: classification, sentiment, summarization, extraction, translation, generation, embeddings |
| `03_chain_components/` | LangChain chains: PromptTemplate → LLM → OutputParser, retriever, memory |
| `04_pipeline_inputs_outputs/` | Business goals → pipeline stages, output contract, AI Functions batch pipeline → Delta |
| `05_tools_multistage_reasoning/` | Define and order tools: fixed orchestration vs tool-calling agent |
| `06_agent_bricks/` | Knowledge Assistant, Multiagent Supervisor, Information Extraction – when to use each |

### Section 2 – Data Preparation (`section2_data_preparation/`)
Self-contained notebooks (no shared helpers) with beginner-friendly comments and their own sample data.

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

Section 2 writes to schema `workspace.fde_data_prep`.

### Section 3 – Application Development (`section3_application_development/`)
Self-contained notebooks with step-by-step What / Why / How-to-read explanations for junior engineers.

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

Section 3 writes to schema `workspace.fde_app_dev` (notebook 13).

Defaults: catalog `workspace`, schema `fde_customer_support`, endpoint `databricks-meta-llama-3-3-70b-instruct` (edit `00_setup/config.py` or the widgets).
Run `00_setup/create_sample_data` first; the other notebooks are independent.
Each notebook calls the model endpoint directly with plain `mlflow.deployments` code (`client.predict(...)`), no shared helper functions.

## Deploy & run
```bash
databricks bundle deploy            # uploads notebooks to /Users/<you>/databricks-fde-usecases
databricks bundle run customer_support_ai_usecases   # Section 1 notebooks on serverless
databricks bundle run data_preparation_usecases      # Section 2 notebooks on serverless
databricks bundle run application_development_usecases   # Section 3 notebooks on serverless
```
