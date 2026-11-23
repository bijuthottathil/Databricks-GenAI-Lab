# Databricks notebook source
# MAGIC %md
# MAGIC # Section 4.3 – Code a simple chain according to requirements
# MAGIC
# MAGIC **The big idea for juniors:** a **chain** is a fixed sequence of steps where each step's output feeds the next –
# MAGIC like an assembly line. Before coding, turn the business **requirements** into a list of components. Then the code
# MAGIC almost writes itself.
# MAGIC
# MAGIC ### The requirements we were given
# MAGIC > R1. Input: a customer product review (free text, any language).
# MAGIC > R2. If the review is not in English, translate it to English first.
# MAGIC > R3. Produce a one-sentence summary.
# MAGIC > R4. Classify the sentiment as positive, neutral or negative.
# MAGIC > R5. Output must be JSON: `{"english_text", "summary", "sentiment"}` so the dashboard can load it.
# MAGIC > R6. Deterministic (same input → same output) and use `databricks-meta-llama-3-3-70b-instruct`.
# MAGIC
# MAGIC ### Requirements → components
# MAGIC | Requirement | Component |
# MAGIC |---|---|
# MAGIC | R1 | input variable `review` |
# MAGIC | R2 | **step 1**: prompt "translate to English if needed" → LLM → text parser |
# MAGIC | R3 + R4 | **step 2**: prompt using the English text → LLM → **JSON output parser** |
# MAGIC | R5 | combine step 1 and step 2 outputs into one dict |
# MAGIC | R6 | `ChatDatabricks(endpoint=..., temperature=0)` |
# MAGIC
# MAGIC ```
# MAGIC {"review"} ─► translate_prompt ─► LLM ─► StrOutputParser ─► english_text
# MAGIC                                                               │
# MAGIC                         analyse_prompt ◄──────────────────────┘
# MAGIC                               │
# MAGIC                              LLM ─► JsonOutputParser ─► {summary, sentiment}
# MAGIC                                                               │
# MAGIC                               final: {english_text, summary, sentiment}
# MAGIC ```
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **LangChain** | A library of building blocks for LLM apps |
# MAGIC | **Prompt template** | A prompt with blanks like `{review}` filled in at run time |
# MAGIC | **`ChatDatabricks`** | LangChain's wrapper for a Databricks model serving endpoint |
# MAGIC | **Output parser** | Turns the model's raw text into a Python string or dict |
# MAGIC | **LCEL (the pipe symbol)** | LangChain Expression Language – writing `a` pipe `b` means "output of a goes into b" |
# MAGIC | **`RunnablePassthrough.assign`** | Keep the existing values and add a new one |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC Step 0 install          Step 1 sample reviews       Step 2 llm + autolog
# MAGIC Step 3 translate_chain  Step 4 analyse_chain        Step 5 full_chain = translate → analyse
# MAGIC Step 6 run on all reviews + check R5   Step 7 same chain in plain Python (for comparison)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q -U databricks-langchain "mlflow[databricks]>=3.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 – Sample reviews (English, German, Spanish)
# MAGIC **Why mixed languages:** R2 says we must handle any language – test data must cover it.

# COMMAND ----------

reviews = [
    "The headphones sound amazing and the battery lasts all week. Best purchase this year!",
    "Die Tastatur kam beschädigt an und der Kundenservice hat nie geantwortet.",
    "El monitor está bien, pero llegó dos días tarde.",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – The model (R6) and tracing
# MAGIC **What:** `ChatDatabricks` points at the endpoint; `temperature=0` makes it deterministic.
# MAGIC `mlflow.langchain.autolog()` records a **trace** of every chain run – expand the Trace panel under a cell's
# MAGIC output to see each step's input and output.

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks

mlflow.langchain.autolog()
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0, max_tokens=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Step 1 of the chain: translate (R2)
# MAGIC **What:** prompt → LLM → `StrOutputParser` (takes the message object and returns plain text).
# MAGIC **How to read the output:** the German review in English, e.g. *"The keyboard arrived damaged and customer service
# MAGIC never replied."*

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "If the text is not in English, translate it to English. If it is already English, return it unchanged. "
               "Return only the text, nothing else."),
    ("user", "{review}"),
])
translate_chain = translate_prompt | llm | StrOutputParser()

print(translate_chain.invoke({"review": reviews[1]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 – Step 2 of the chain: summarise + classify (R3, R4)
# MAGIC **What:** prompt → LLM → `JsonOutputParser` (parses the JSON text into a Python dict, and copes with extra
# MAGIC formatting like ```json fences).
# MAGIC **Why the allowed values are listed in the prompt:** so the dashboard only ever receives positive/neutral/negative.
# MAGIC **How to read the output:** a dict like `{'summary': '...', 'sentiment': 'negative'}`.

# COMMAND ----------

from langchain_core.output_parsers import JsonOutputParser

analyse_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Return ONLY JSON: {{"summary": "<one sentence>", "sentiment": "positive|neutral|negative"}}'),
    ("user", "Review: {english_text}"),
])
analyse_chain = analyse_prompt | llm | JsonOutputParser()

print(analyse_chain.invoke({"english_text": "The keyboard arrived damaged and customer service never replied."}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Connect the steps into one chain (R5)
# MAGIC **Code walk-through:**
# MAGIC 1. `RunnablePassthrough.assign(english_text=translate_chain)` – keeps `review` and **adds** `english_text`.
# MAGIC 2. `.assign(analysis=analyse_chain)` – runs step 2 using `english_text`, adds `analysis` (the dict).
# MAGIC 3. `RunnableLambda(...)` – reshapes everything into the exact output R5 asks for.
# MAGIC
# MAGIC Note: `{{` and `}}` in the prompt above are "escaped" braces – they print as `{` `}` instead of being treated as
# MAGIC template blanks.

# COMMAND ----------

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

full_chain = (
    RunnablePassthrough.assign(english_text=translate_chain)
    | RunnablePassthrough.assign(analysis=analyse_chain)
    | RunnableLambda(lambda x: {"english_text": x["english_text"],
                                "summary": x["analysis"].get("summary"),
                                "sentiment": x["analysis"].get("sentiment")})
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 – Run on all reviews and check the requirements
# MAGIC **What:** `full_chain.batch([...])` runs the chain on several inputs (in parallel).
# MAGIC Then we **test** R5 (exact keys) and R4 (allowed sentiment values) in code – requirements you can check
# MAGIC automatically should be checked automatically.
# MAGIC
# MAGIC **How to read the output:** a table with English text, summary and sentiment per review; then `R4 ok: True,
# MAGIC R5 ok: True`. Expected sentiments: positive, negative, neutral (or negative – late delivery).

# COMMAND ----------

import json

results = full_chain.batch([{"review": r} for r in reviews])
display(spark.createDataFrame([(r, o["english_text"], o["summary"], o["sentiment"]) for r, o in zip(reviews, results)],
                              "review string, english_text string, summary string, sentiment string"))

r5_ok = all(set(o) == {"english_text", "summary", "sentiment"} for o in results)
r4_ok = all(o["sentiment"] in {"positive", "neutral", "negative"} for o in results)
print(f"R4 ok: {r4_ok}, R5 ok: {r5_ok}")
print(json.dumps(results[0], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 – The same chain in plain Python (no LangChain)
# MAGIC **Why show this:** a chain is just "call, take the output, call again". LangChain adds reusable parts, batching,
# MAGIC tracing and easy swapping of models – but the logic is identical. Use whichever your team prefers.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def plain_chain(review):
    def ask(system, user):
        r = client.predict(endpoint="databricks-meta-llama-3-3-70b-instruct",
                           inputs={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   "temperature": 0.0, "max_tokens": 200})
        return r["choices"][0]["message"]["content"].strip()
    english = ask("If the text is not in English, translate it to English. Return only the text.", review)
    raw = ask('Return ONLY JSON: {"summary": "<one sentence>", "sentiment": "positive|neutral|negative"}', f"Review: {english}")
    analysis = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    return {"english_text": english, "summary": analysis.get("summary"), "sentiment": analysis.get("sentiment")}

print(plain_chain(reviews[2]))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Start from **requirements** → list the components (inputs, prompts, model, parsers, output shape) → wire them.
# MAGIC - LCEL: `prompt | llm | parser`; use `RunnablePassthrough.assign` to pass values between steps.
# MAGIC - `StrOutputParser` for text, `JsonOutputParser` for structured output; `temperature=0` for repeatability.
# MAGIC - Verify requirements in code (keys, allowed values) and trace runs with `mlflow.langchain.autolog()`.
