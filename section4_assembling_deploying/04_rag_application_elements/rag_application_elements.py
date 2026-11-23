# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.4 – Choose the basic elements of a RAG application
# MAGIC ### model flavor · embedding model · retriever · dependencies · input example · model signature
# MAGIC
# MAGIC **The big idea for juniors:** a RAG app (Retrieval-Augmented Generation) = **find** relevant text, then **ask the
# MAGIC LLM** to answer using that text. To *deploy* it, you package it as an MLflow model – and that package needs six
# MAGIC basic elements. Think of shipping a product: the product itself, the parts inside, the instruction label, a sample,
# MAGIC and a list of batteries required.
# MAGIC
# MAGIC | Element | Plain meaning | Our choice here | Common alternatives |
# MAGIC |---|---|---|---|
# MAGIC | **Model flavor** | The package *type* MLflow uses to save & load it | `pyfunc` (custom Python) | `langchain` flavor, `ResponsesAgent` (pyfunc for agents), `openai` |
# MAGIC | **Embedding model** | Turns text into vectors for searching | `databricks-gte-large-en` (8192-token context) | `databricks-bge-large-en` (512), external models |
# MAGIC | **Retriever** | Finds the top-k most relevant chunks for a question | in-memory cosine similarity (small demo) | **Mosaic AI Vector Search** index (4.6) |
# MAGIC | **LLM** | Writes the answer from the retrieved chunks | `databricks-meta-llama-3-3-70b-instruct` | other Foundation Model endpoints |
# MAGIC | **Dependencies** | Packages + Databricks resources the model needs | `pip_requirements` + `resources` | `conda_env`, `code_paths` |
# MAGIC | **Input example** | A sample request saved with the model | `{"question": "..."}` | – |
# MAGIC | **Model signature** | The declared input/output schema | input `question: string` → output `answer`, `sources` | inferred with `infer_signature` |
# MAGIC
# MAGIC **Why signature + input example matter:** Unity Catalog **requires a signature** to register a model, and the
# MAGIC serving endpoint uses it to validate requests. The input example documents how to call the model and is used to
# MAGIC test it before deployment.
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0  install MLflow 3
# MAGIC Step 1  knowledge base (6 policy chunks) → saved to a JSON file = an ARTIFACT of the model
# MAGIC Step 2  RagChain class: load_context (read file, embed chunks) → predict (embed question → retrieve → generate)
# MAGIC Step 3  test the class in the notebook
# MAGIC Step 4  define signature, input example, dependencies, resources → log_model
# MAGIC Step 5  inspect the saved package (flavor, signature) and load + predict
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – The knowledge base (our documents)
# MAGIC **What:** 6 short chunks, each with an id and a source. We save them to a local JSON file and will attach that file
# MAGIC to the model as an **artifact** (a file packaged inside the model).
# MAGIC **In production:** chunks live in a Delta table and are searched with a **Vector Search index** instead of a file.

# COMMAND ----------

import json

knowledge_base = [
    {"id": "ret-1", "source": "returns_policy.md", "text": "Most products can be returned within 30 days of delivery."},
    {"id": "ret-2", "source": "returns_policy.md", "text": "Opened software and gift cards cannot be returned."},
    {"id": "shp-1", "source": "shipping_policy.md", "text": "Standard shipping takes 3 to 5 business days; express takes 1 to 2."},
    {"id": "shp-2", "source": "shipping_policy.md", "text": "Orders delayed more than 3 days qualify for an investigation and a $10 store credit."},
    {"id": "war-1", "source": "warranty.md", "text": "Defective products are replaced free within 15 days of delivery."},
    {"id": "pay-1", "source": "payments.md", "text": "Duplicate charges are refunded within 3 business days once confirmed."},
]
KB_PATH = "/tmp/fde_knowledge_base.json"
with open(KB_PATH, "w") as f:
    json.dump(knowledge_base, f)
print(f"Saved {len(knowledge_base)} chunks to {KB_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The RAG chain as a pyfunc model
# MAGIC **Code walk-through:**
# MAGIC - `load_context` (runs once at load time):
# MAGIC   - `context.artifacts["knowledge_base"]` = the path of the JSON file **inside the model package**,
# MAGIC   - embed all chunks once with the **embedding model** and keep the vectors in memory.
# MAGIC - `retrieve(question, k)` – the **retriever**: embed the question, cosine similarity with every chunk, top-k.
# MAGIC - `predict` – for each question: retrieve → put the chunks in the prompt → **LLM** answers → return answer + sources.
# MAGIC
# MAGIC **Why vectors are normalised:** after dividing by their length, a dot product = cosine similarity.

# COMMAND ----------

import json
import mlflow
import numpy as np
import pandas as pd

EMBEDDING_MODEL = "databricks-gte-large-en"
LLM = "databricks-meta-llama-3-3-70b-instruct"


class RagChain(mlflow.pyfunc.PythonModel):

    def _embed(self, texts):
        from mlflow.deployments import get_deploy_client
        r = get_deploy_client("databricks").predict(endpoint=EMBEDDING_MODEL, inputs={"input": texts})
        v = np.array([d["embedding"] for d in r["data"]])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def load_context(self, context):
        with open(context.artifacts["knowledge_base"]) as f:
            self.chunks = json.load(f)
        self.vectors = self._embed([c["text"] for c in self.chunks])          # embed the knowledge base once

    def retrieve(self, question, k=2):
        scores = self.vectors @ self._embed([question])[0]
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]

    def predict(self, context, model_input, params=None):
        from mlflow.deployments import get_deploy_client
        answers, sources = [], []
        for q in model_input["question"]:
            hits = self.retrieve(q)
            context_text = "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
            r = get_deploy_client("databricks").predict(endpoint=LLM, inputs={
                "messages": [{"role": "system", "content": "Answer ONLY from the context. If the answer is not there, "
                                                           "say you don't know. Keep it to 2 sentences.\n\nContext:\n" + context_text},
                             {"role": "user", "content": q}],
                "temperature": 0.0, "max_tokens": 150})
            answers.append(r["choices"][0]["message"]["content"].strip())
            sources.append(", ".join(f"{h['source']}#{h['id']}" for h in hits))
        return pd.DataFrame({"answer": answers, "sources": sources})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Test the class before logging
# MAGIC **What:** a tiny fake "context" object that has the same `artifacts` attribute MLflow would give us.
# MAGIC **How to read the output:** each answer should match its source – late order → `shipping_policy.md#shp-2`
# MAGIC ($10 credit), opened software → `returns_policy.md#ret-2`.

# COMMAND ----------

from types import SimpleNamespace

rag = RagChain()
rag.load_context(SimpleNamespace(artifacts={"knowledge_base": KB_PATH}))
questions = pd.DataFrame({"question": ["My order is 5 days late, do I get anything?", "Can I return opened software?"]})
display(rag.predict(None, questions))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Define all the elements and log the model
# MAGIC | Argument | Element | Value |
# MAGIC |---|---|---|
# MAGIC | `mlflow.pyfunc.log_model` | **model flavor** | pyfunc |
# MAGIC | `artifacts` | data files | the knowledge base JSON |
# MAGIC | `signature` | **model signature** | `question` → `answer`, `sources` |
# MAGIC | `input_example` | **input example** | one sample question |
# MAGIC | `pip_requirements` | **dependencies** (packages) | mlflow, numpy, pandas |
# MAGIC | `resources` | **dependencies** (Databricks) | the embedding endpoint + the LLM endpoint → automatic auth when served |
# MAGIC
# MAGIC **Tip:** pin versions in production (e.g. `numpy==1.26.4`) so the serving container builds the same environment.

# COMMAND ----------

from mlflow.models import infer_signature
from mlflow.models.resources import DatabricksServingEndpoint

input_example = pd.DataFrame({"question": ["How long do I have to return a product?"]})
signature = infer_signature(input_example, pd.DataFrame({"answer": ["You have 30 days."], "sources": ["returns_policy.md#ret-1"]}))

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_section4_rag_elements")
with mlflow.start_run(run_name="rag_chain_v1"):
    logged = mlflow.pyfunc.log_model(
        name="rag_chain",
        python_model=RagChain(),
        artifacts={"knowledge_base": KB_PATH},
        signature=signature,
        input_example=input_example,
        pip_requirements=["mlflow>=3.1", "numpy", "pandas"],
        resources=[DatabricksServingEndpoint(endpoint_name=EMBEDDING_MODEL), DatabricksServingEndpoint(endpoint_name=LLM)],
    )
print("Model URI:", logged.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Inspect the package, then load and predict
# MAGIC **How to read the output:**
# MAGIC - `flavors` → `python_function` = the pyfunc flavor.
# MAGIC - `signature` → inputs `question: string`, outputs `answer`, `sources`.
# MAGIC - The prediction table – the same answers as Step 3, now from the **saved** package (artifact file included).

# COMMAND ----------

info = mlflow.models.get_model_info(logged.model_uri)
print("flavors:  ", list(info.flavors))
print("signature:", info.signature)

loaded = mlflow.pyfunc.load_model(logged.model_uri)
display(loaded.predict(pd.DataFrame({"question": ["How fast is express shipping?", "Do you sell gift wrapping?"]})))

# COMMAND ----------

# MAGIC %md
# MAGIC **How to read the last answer:** "Do you sell gift wrapping?" is **not** in the knowledge base – a good RAG app says
# MAGIC it doesn't know instead of inventing an answer.
# MAGIC
# MAGIC **Exam takeaways**
# MAGIC - Basic RAG elements: **flavor** (pyfunc / langchain / ResponsesAgent), **embedding model**, **retriever**
# MAGIC   (Vector Search in production), **LLM**, **dependencies** (`pip_requirements` + `resources`), **input example**,
# MAGIC   **signature**.
# MAGIC - Use the **same embedding model** for documents and questions – vectors from different models can't be compared.
# MAGIC - A signature is **required** to register in Unity Catalog; `infer_signature` builds it from sample input/output.
