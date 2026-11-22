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

Defaults: catalog `workspace`, schema `fde_customer_support`, endpoint `databricks-meta-llama-3-3-70b-instruct` (edit `00_setup/config.py` or the widgets).
Run `00_setup/create_sample_data` first; the other notebooks are independent.
Each notebook calls the model endpoint directly with plain `mlflow.deployments` code (`client.predict(...)`), no shared helper functions.

## Deploy & run
```bash
databricks bundle deploy            # uploads notebooks to /Users/<you>/databricks-fde-usecases
databricks bundle run customer_support_ai_usecases   # runs all notebooks on serverless
```
