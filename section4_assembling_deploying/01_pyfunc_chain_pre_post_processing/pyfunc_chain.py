# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.1 – Code a chain using a pyfunc model with pre- and post-processing
# MAGIC
# MAGIC **The big idea for juniors:** to *deploy* a GenAI app, you package it as an **MLflow model**. The most flexible
# MAGIC package type is **`pyfunc`** ("Python function"): you write a Python class with a `predict()` method, and MLflow
# MAGIC can save it, version it, and serve it behind a REST API.
# MAGIC
# MAGIC Inside `predict()` we build a small **chain** with three stages:
# MAGIC ```
# MAGIC raw request ─► PRE-PROCESSING ─► LLM call ─► POST-PROCESSING ─► clean response
# MAGIC               • trim / clean text           • parse JSON
# MAGIC               • mask PII (emails, phones)   • validate category (allowed list)
# MAGIC               • reject empty input          • cut reply to max length
# MAGIC               • build the prompt            • safe fallback if the model misbehaves
# MAGIC ```
# MAGIC
# MAGIC **Why pre-processing?** The model should only receive clean, safe, well-formed input.
# MAGIC **Why post-processing?** The app calling us needs a *reliable* output shape, even when the LLM rambles or breaks
# MAGIC the format.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **MLflow model** | A saved, versioned package of your code + settings + dependencies |
# MAGIC | **Flavor** | The *type* of package. `pyfunc` = "any Python code with a predict method" |
# MAGIC | **`PythonModel`** | The base class you inherit from to write a custom pyfunc |
# MAGIC | **`load_context()`** | Runs **once** when the model is loaded (read settings, create clients) |
# MAGIC | **`predict()`** | Runs on **every request** |
# MAGIC | **`model_config`** | Settings saved with the model (endpoint name, max words) – change them without changing code |
# MAGIC | **Signature** | The declared input and output columns, so callers know the shape |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0  install MLflow 3
# MAGIC Step 1  sample requests (good, messy, PII, empty)
# MAGIC Step 2  SupportChain class: load_context → predict = preprocess → call_llm → postprocess
# MAGIC Step 3  test the class directly in the notebook
# MAGIC Step 4  log it to MLflow (with model_config, signature, input example)
# MAGIC Step 5  load the saved model back and predict – exactly what a serving endpoint does
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 0 – Install a recent MLflow
# MAGIC **Note:** `%restart_python` clears all variables – run this cell first, then the rest in order.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample requests
# MAGIC **What:** a pandas DataFrame with one column `message`. pyfunc models receive a **DataFrame** – one row per request,
# MAGIC so the same model handles one message or a thousand.
# MAGIC
# MAGIC **What each row tests:**
# MAGIC | Row | Tests |
# MAGIC |---|---|
# MAGIC | 1 | a normal, clean message |
# MAGIC | 2 | messy spacing and line breaks → pre-processing cleans it |
# MAGIC | 3 | an email and phone number → pre-processing masks them before the LLM sees them |
# MAGIC | 4 | an empty message → pre-processing skips the LLM call entirely |

# COMMAND ----------

import pandas as pd

requests_df = pd.DataFrame({"message": [
    "My headphones stopped working after 5 days. What can I do?",
    "   where    is my order???\n\n it is   a week late   ",
    "Please refund me, my email is jane.doe@example.com and phone 555-123-4567. I was charged twice.",
    "   ",
]})
display(requests_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The chain as a `PythonModel`
# MAGIC **Code walk-through:**
# MAGIC - `load_context(context)` – reads `model_config` (endpoint name, allowed categories, max words). Falls back to
# MAGIC   defaults so we can also test the class without saving it first.
# MAGIC - `preprocess(text)` – strips spaces, collapses repeated whitespace, masks emails/phones with regex, limits the
# MAGIC   length to 2,000 characters (protects cost and context window). Returns the clean text and which PII was found.
# MAGIC - `call_llm(text)` – the plain `client.predict` call. It asks for JSON `{"category": ..., "reply": ...}`.
# MAGIC - `postprocess(raw)` – the safety net for the output:
# MAGIC   - find and parse the JSON (if it fails → fallback answer),
# MAGIC   - category not in the allowed list → `"Other"`,
# MAGIC   - reply longer than `max_words` → cut it,
# MAGIC   - `status` tells the caller whether the answer is normal (`ok`) or a fallback.
# MAGIC - `predict(context, model_input)` – runs the three stages for **every row** and returns a DataFrame.
# MAGIC
# MAGIC This cell only defines the class – nothing is printed.

# COMMAND ----------

import json, re
import mlflow
import pandas as pd


class SupportChain(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        config = (context.model_config if context is not None else None) or {}
        self.endpoint = config.get("endpoint", "databricks-meta-llama-3-3-70b-instruct")
        self.categories = config.get("categories", ["Shipping", "Payment", "Product Defect", "Return", "Other"])
        self.max_words = int(config.get("max_words", 60))

    # ---------- PRE-PROCESSING ----------
    def preprocess(self, text):
        text = re.sub(r"\s+", " ", str(text or "")).strip()                 # collapse spaces / new lines
        pii = []
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
            pii.append("EMAIL")
            text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
        if re.search(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", text):
            pii.append("PHONE")
            text = re.sub(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", "[PHONE]", text)
        return text[:2000], pii                                              # length limit

    # ---------- MODEL CALL ----------
    def call_llm(self, text):
        from mlflow.deployments import get_deploy_client                     # created here so the model can be saved
        prompt = (f"Classify the customer message into one of {self.categories} and write a short, polite reply.\n"
                  'Return ONLY JSON: {"category": "...", "reply": "..."}\n\n'
                  f"Customer message: {text}")
        response = get_deploy_client("databricks").predict(
            endpoint=self.endpoint,
            inputs={"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 250})
        return response["choices"][0]["message"]["content"]

    # ---------- POST-PROCESSING ----------
    def postprocess(self, raw):
        try:
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception:
            return "Other", "Thanks for your message. A support agent will get back to you shortly.", "fallback: bad JSON"
        category = data.get("category") if data.get("category") in self.categories else "Other"
        words = str(data.get("reply", "")).split()
        reply = " ".join(words[: self.max_words]) + ("..." if len(words) > self.max_words else "")
        return category, reply, "ok"

    # ---------- THE CHAIN ----------
    def predict(self, context, model_input, params=None):
        if not hasattr(self, "endpoint"):                                    # allows testing without load_context
            self.load_context(None)
        rows = []
        for message in model_input["message"]:
            clean, pii = self.preprocess(message)
            if not clean:                                                    # empty → no LLM call needed
                rows.append({"category": "Other", "reply": "Could you tell us a bit more about your issue?",
                             "pii_masked": "", "status": "skipped: empty input"})
                continue
            category, reply, status = self.postprocess(self.call_llm(clean))
            rows.append({"category": category, "reply": reply, "pii_masked": ",".join(pii), "status": status})
        return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Test the class directly (before saving)
# MAGIC **Why:** catch bugs quickly. First we test the pre-processing alone (no LLM, instant), then the whole chain.
# MAGIC
# MAGIC **How to read the output:**
# MAGIC - Pre-processing test: row 2 becomes `"where is my order??? it is a week late"`; row 3 shows `[EMAIL]` and `[PHONE]`.
# MAGIC - Full chain: one row per request with `category`, `reply`, `pii_masked`, `status`. Row 4 is `skipped: empty input`
# MAGIC   (no model call was made). Row 3's category should be `Payment`.

# COMMAND ----------

chain = SupportChain()
chain.load_context(None)
for m in requests_df["message"]:
    print(repr(m[:60]), "->", chain.preprocess(m))

display(chain.predict(None, requests_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Log the chain to MLflow
# MAGIC **What we pass to `log_model`:**
# MAGIC | Argument | Why |
# MAGIC |---|---|
# MAGIC | `python_model=SupportChain()` | the code |
# MAGIC | `model_config={...}` | settings stored *with* the model – e.g. switch endpoint without changing code |
# MAGIC | `signature` | declares "input: column `message` (string); output: category, reply, pii_masked, status" |
# MAGIC | `input_example` | a sample request, shown in the UI and used to test the model |
# MAGIC | `pip_requirements` | packages the serving container must install |
# MAGIC
# MAGIC `infer_signature` looks at a sample input and output and writes the signature for us.
# MAGIC
# MAGIC **How to read the output:** the model URI (`models:/m-...`) – the address of the saved package – and the signature.

# COMMAND ----------

from mlflow.models import infer_signature

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fde_section4_pyfunc_chain")

sample_input = requests_df.head(1)
sample_output = pd.DataFrame([{"category": "Product Defect", "reply": "Sorry to hear that...", "pii_masked": "", "status": "ok"}])
signature = infer_signature(sample_input, sample_output)

with mlflow.start_run(run_name="support_chain_v1"):
    logged = mlflow.pyfunc.log_model(
        name="support_chain",
        python_model=SupportChain(),
        model_config={"endpoint": "databricks-meta-llama-3-3-70b-instruct", "max_words": 40,
                      "categories": ["Shipping", "Payment", "Product Defect", "Return", "Other"]},
        signature=signature,
        input_example=sample_input,
        pip_requirements=["mlflow>=3.1", "pandas"],
    )
print("Model URI:", logged.model_uri)
print(signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Load the saved model and predict (what serving does)
# MAGIC **Why:** proves the package works on its own – the serving endpoint will do exactly this: load → `load_context`
# MAGIC once → `predict` for each request.
# MAGIC
# MAGIC **How to read the output:** same columns as Step 3. Replies should now be **≤ 40 words**, because the logged
# MAGIC `model_config` set `max_words` to 40 (the class default is 60) – proof that config travels with the model.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(logged.model_uri)
display(loaded.predict(requests_df))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - A **pyfunc** model = class inheriting `mlflow.pyfunc.PythonModel` with `load_context` (once) and `predict` (per request).
# MAGIC - Put **pre-processing** (clean, mask PII, validate, build prompt) and **post-processing** (parse, validate, trim,
# MAGIC   fallback) inside `predict`, so every caller gets the same safe behaviour.
# MAGIC - Log with **`model_config`, signature, input example and pip requirements**; create clients inside methods, not at
# MAGIC   import time, so the model can be saved and loaded anywhere.
# MAGIC - Production teams often log **"models from code"** (`python_model="chain.py"`) instead of a Python object.
