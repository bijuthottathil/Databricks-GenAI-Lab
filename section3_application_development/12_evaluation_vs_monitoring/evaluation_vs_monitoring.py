# Databricks notebook source
# MAGIC %md
# MAGIC # Section 3.12 – Compare the evaluation and monitoring phases of the GenAI application life cycle
# MAGIC
# MAGIC **The big idea for juniors:** testing a GenAI app happens in **two different phases**:
# MAGIC
# MAGIC ```
# MAGIC   BUILD ──► EVALUATION (before release) ──► DEPLOY ──► MONITORING (after release) ──┐
# MAGIC     ▲        "Is it good enough to ship?"               "Is it still working well?"  │
# MAGIC     └──────────────── problems & new examples found in monitoring feed back ─────────┘
# MAGIC ```
# MAGIC
# MAGIC **Analogy:** a driving test vs. a car's dashboard. The **test** (evaluation) happens once, on a known route, with an
# MAGIC examiner who knows the right answers. The **dashboard** (monitoring) runs every day on real roads, with no examiner –
# MAGIC just warning lights (speed, fuel, engine temperature) that tell you when something goes wrong.
# MAGIC
# MAGIC | | **Evaluation** (offline) | **Monitoring** (online) |
# MAGIC |---|---|---|
# MAGIC | When | Before deployment, and before every new version | Continuously after deployment |
# MAGIC | Data | A curated **evaluation set** with expected answers | **Real user traffic** – no expected answers |
# MAGIC | Question answered | "Is version B better than A? Is it good enough to ship?" | "Is quality, speed or cost getting worse? Are users asking new things?" |
# MAGIC | Typical metrics | Correctness vs ground truth, groundedness, retrieval recall, safety | Latency, errors, token cost, user feedback (👍/👎), LLM-judge scores on samples, topic drift |
# MAGIC | Who/what grades | Ground truth + LLM judges + expert reviewers | LLM judges on samples + user feedback + operational stats |
# MAGIC | Outcome | Go / no-go decision for a release | Alerts, dashboards, new eval examples, rollback |
# MAGIC | Databricks tools | `mlflow.genai.evaluate`, evaluation datasets, Review App | Inference tables, MLflow production monitoring (scorers on traces), Lakehouse Monitoring, dashboards & SQL alerts |
# MAGIC
# MAGIC **In this notebook:** Part A evaluates two bot versions offline and makes a release decision. Part B monitors 14 days
# MAGIC of (simulated) production logs and finds a problem evaluation could not have seen.
# MAGIC
# MAGIC ### Key words used in this notebook
# MAGIC | Word | Plain meaning |
# MAGIC |---|---|
# MAGIC | **Ground truth** | The known correct answer, written by a person |
# MAGIC | **Evaluation set** | A fixed list of test questions + their ground truth |
# MAGIC | **Regression** | Something that worked in the old version but is broken in the new one |
# MAGIC | **Quality gate** | A pass/fail rule a new version must meet before release |
# MAGIC | **Inference table** | A Delta table where a serving endpoint automatically logs every request and response |
# MAGIC | **Proxy metric** | Something measurable that hints at quality when you don't know the right answer (👎 rate, latency) |
# MAGIC | **Drift** | Real traffic slowly changing (new topics, new wording) compared to what you tested |
# MAGIC
# MAGIC ### How the cells connect
# MAGIC ```
# MAGIC PART A (evaluation)                         PART B (monitoring)
# MAGIC A1 eval_set (6 questions + v1/v2 answers)   B1 prod logs (14 days, simulated) → production_logs view
# MAGIC A2 score each answer → rows, accuracies     B2 daily metrics from the logs → daily
# MAGIC A3 quality gate → ship / don't ship         B3 compare daily with THRESHOLDS → alerts
# MAGIC                                             B4 slice by topic → find the cause
# MAGIC                                             B5 topics missing from eval_set → feed back to Part A
# MAGIC ```
# MAGIC No model calls in this notebook – answers and logs are pre-made so you can focus on the *process*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A – EVALUATION (before release)
# MAGIC ### Step A1 – The evaluation set and the answers of two bot versions
# MAGIC **What:** 6 questions with the fact a correct answer must contain, and the (pre-recorded) answers from bot **v1** and
# MAGIC a new candidate **v2**.
# MAGIC **Why pre-recorded:** to focus on the *process*. In practice `mlflow.genai.evaluate` calls your app for you.
# MAGIC
# MAGIC **Spot the difference yourself:** v1 misses the $10 credit (question 2) and wrongly says opened software can be
# MAGIC returned (question 5). v2 gets both right.

# COMMAND ----------

eval_set = [
    # question, required fact, v1 answer, v2 answer
    ("How long do I have to return an item?", "30 days",
     "You can return items within 30 days.", "Returns are accepted within 30 days of delivery."),
    ("Do I get anything if my order is late?", "$10",
     "We are sorry for the delay.", "Orders over 3 days late get a $10 store credit."),
    ("How long is express shipping?", "1 to 2",
     "Express takes 1 to 2 business days.", "Express shipping takes 1 to 2 business days."),
    ("Is a broken item replaced?", "15 days",
     "Defective items are replaced within 15 days.", "Yes, defective items are replaced free within 15 days."),
    ("Can I return opened software?", "cannot",
     "Yes, all items can be returned.", "Opened software cannot be returned."),
    ("Which payment methods do you take?", "PayPal",
     "We accept cards and PayPal.", "We accept credit cards, debit cards and PayPal."),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A2 – Score both versions against the ground truth
# MAGIC **What:** `correct` = the answer contains the required fact. We compute accuracy per version.
# MAGIC **Why this is possible only in evaluation:** we *know* the right answers here. In production we won't.
# MAGIC
# MAGIC **How to read the output:** a table with ✓/✗ per question for each version, then
# MAGIC `v1 accuracy: 67%   v2 accuracy: 100%`. (The check is a simple "does the fact appear" test; for real apps use an
# MAGIC LLM judge such as MLflow's `Correctness`.)

# COMMAND ----------

def correct(answer, fact):
    return fact.lower() in answer.lower()

rows = [(q, fact, correct(v1, fact), correct(v2, fact)) for q, fact, v1, v2 in eval_set]
display(spark.createDataFrame(rows, "question string, required_fact string, v1_correct boolean, v2_correct boolean"))

v1_acc = sum(r[2] for r in rows) / len(rows)
v2_acc = sum(r[3] for r in rows) / len(rows)
print(f"v1 accuracy: {v1_acc:.0%}   v2 accuracy: {v2_acc:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A3 – Release decision (a "quality gate")
# MAGIC **Rule:** ship v2 only if it reaches ≥ 90% accuracy **and** is not worse than v1 on any question (no regressions).
# MAGIC
# MAGIC **Why check regressions separately:** a new version can have higher accuracy overall and still break a question
# MAGIC that used to work – users notice that. `regressions` lists questions where v1 was right and v2 is wrong.
# MAGIC
# MAGIC **How to read the output:** `DECISION: ✅ ship v2 | regressions: none`. Evaluation said yes – but watch what
# MAGIC Part B finds after release.

# COMMAND ----------

regressions = [r[0] for r in rows if r[2] and not r[3]]
ship = v2_acc >= 0.9 and not regressions
print("DECISION:", "✅ ship v2" if ship else "❌ do not ship v2", "| regressions:", regressions or "none")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B – MONITORING (after release)
# MAGIC ### Step B1 – Production logs (simulated 14 days of traffic)
# MAGIC **What:** each row is one real request as an *inference table* would store it: time, question topic, latency,
# MAGIC tokens used, errors, and user feedback (👍 = 1, 👎 = 0, or none).
# MAGIC **What we planted:** from day 8, users start asking about **international shipping** – a topic the eval set never
# MAGIC covered – and the bot handles it badly (more 👎, longer answers, slower).
# MAGIC
# MAGIC **How the fake data is generated (40 requests per day):**
# MAGIC | | Normal topics | International shipping (from day 8, ~35% of traffic) |
# MAGIC |---|---|---|
# MAGIC | latency | ~1.2 s | ~2.8 s |
# MAGIC | tokens | ~180 | ~420 |
# MAGIC | error chance | 1% | 6% |
# MAGIC | 👎 chance (when rated) | 10% | 60% |
# MAGIC
# MAGIC `random.gauss(mean, spread)` adds realistic noise; `random.seed(42)` makes the "random" data the same on every run.
# MAGIC Feedback is `None` about 2 out of 3 times – most real users never click 👍/👎.

# COMMAND ----------

import random
from datetime import date, timedelta

random.seed(42)
topics = ["returns", "late order", "express shipping", "warranty", "payments"]
logs = []
for day in range(14):
    d = date(2026, 11, 1) + timedelta(days=day)
    for _ in range(40):
        new_topic = day >= 7 and random.random() < 0.35
        topic = "international shipping" if new_topic else random.choice(topics)
        latency = random.gauss(2.8 if new_topic else 1.2, 0.3)
        tokens = int(random.gauss(420 if new_topic else 180, 30))
        error = random.random() < (0.06 if new_topic else 0.01)
        feedback = random.choice([None, None, 1 if random.random() > (0.6 if new_topic else 0.1) else 0])
        logs.append((d, topic, round(max(latency, 0.2), 2), tokens, error, feedback))

prod = spark.createDataFrame(logs, "request_date date, topic string, latency_s double, total_tokens int, error boolean, feedback int")
prod.createOrReplaceTempView("production_logs")
display(prod.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B2 – Daily monitoring dashboard
# MAGIC **What:** per day: request count, p95 latency, average tokens (≈ cost), error rate, and 👎 rate (share of rated
# MAGIC requests that got a thumbs-down).
# MAGIC **Why these metrics:** none needs a "correct answer" – they come straight from the logs. That's what makes them
# MAGIC usable in production.
# MAGIC **How to read:** look for the jump around **2026-11-08**. Switch the result to a line chart to see it clearly.
# MAGIC
# MAGIC **SQL walk-through:** `percentile(latency_s, 0.95)` = p95 latency; `avg(CAST(error AS INT))` = share of errors
# MAGIC (true=1, false=0); `1 - avg(feedback)` = 👎 rate, because 👍=1 and 👎=0 and `avg` skips the `NULL` (unrated) rows.

# COMMAND ----------

daily = spark.sql("""
SELECT request_date,
       count(*)                                    AS requests,
       round(percentile(latency_s, 0.95), 2)       AS p95_latency_s,
       round(avg(total_tokens))                    AS avg_tokens,
       round(avg(CAST(error AS INT)), 3)           AS error_rate,
       round(1 - avg(feedback), 3)                 AS thumbs_down_rate
FROM production_logs
GROUP BY request_date
ORDER BY request_date
""")
display(daily)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B3 – Alerts: compare each day with agreed thresholds
# MAGIC **What:** flag any day that breaks a limit. In Databricks you'd put the query above in a **SQL alert** or a
# MAGIC **Lakehouse Monitoring** monitor and get notified automatically.
# MAGIC
# MAGIC **How to read the output:** days from 2026-11-01 to 11-07 should have no alerts; from 2026-11-08 onwards most days
# MAGIC break the latency and/or 👎 limits. The exact numbers per day vary because of the random noise.

# COMMAND ----------

THRESHOLDS = {"p95_latency_s": 2.5, "error_rate": 0.03, "thumbs_down_rate": 0.25}
alerts = []
for r in daily.collect():
    broken = [f"{m}={r[m]} > {limit}" for m, limit in THRESHOLDS.items() if r[m] is not None and r[m] > limit]
    if broken:
        alerts.append((str(r.request_date), "; ".join(broken)))
display(spark.createDataFrame(alerts or [("none", "no alerts")], "date string, alerts string"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B4 – Find *why*: topic drift
# MAGIC **What:** compare the mix of topics before and after the problems started, with their 👎 rates.
# MAGIC **Why:** monitoring tells you *something* changed; slicing by topic tells you *what*. Here a brand-new topic appears.
# MAGIC
# MAGIC **How to read the output:** in week 2, "international shipping" is the biggest topic with a high 👎 rate, while
# MAGIC the old topics stay low. That's the cause: not a broken bot, but a **new kind of question** it was never tested on.

# COMMAND ----------

display(spark.sql("""
SELECT CASE WHEN request_date < '2026-11-08' THEN 'week 1' ELSE 'week 2' END AS period,
       topic,
       count(*) AS requests,
       round(1 - avg(feedback), 2) AS thumbs_down_rate
FROM production_logs
GROUP BY ALL
ORDER BY period, requests DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B5 – Close the loop: monitoring feeds evaluation
# MAGIC **What happens next in a real team:**
# MAGIC 1. Take real "international shipping" questions from the logs (especially 👎 ones).
# MAGIC 2. Have an expert write the correct answers (e.g. in the **Review App**) → **add them to the evaluation set**.
# MAGIC 3. Fix the app (add an international-shipping policy document, adjust the prompt).
# MAGIC 4. Re-run **evaluation** – the new version must pass the bigger eval set before it ships.

# COMMAND ----------

new_eval_examples = spark.sql("""
SELECT DISTINCT topic FROM production_logs WHERE topic NOT IN ('returns','late order','express shipping','warranty','payments')
""")
print("Topics seen in production but missing from the evaluation set:", [r.topic for r in new_eval_examples.collect()])

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - **Evaluation** = offline, before release, curated data **with ground truth**, compares versions, go/no-go decision.
# MAGIC - **Monitoring** = online, after release, **real traffic without ground truth**, tracks quality proxies (LLM judges on
# MAGIC   samples, user feedback) plus latency, errors, cost and drift; triggers alerts.
# MAGIC - The phases form a **loop**: monitoring finds new failure cases → they become new evaluation examples.
# MAGIC - Databricks: `mlflow.genai.evaluate` + evaluation datasets + Review App (evaluation); inference tables, MLflow
# MAGIC   production monitoring, Lakehouse Monitoring, SQL alerts and dashboards (monitoring).
