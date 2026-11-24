# Databricks notebook source
# MAGIC %md
# MAGIC # 00 – Sample data for the Customer Support AI app
# MAGIC Scenario: an e-commerce company wants an assistant that classifies complaints, analyzes sentiment,
# MAGIC retrieves policies, recommends actions, and extracts structured info.
# MAGIC
# MAGIC Creates three Delta tables: `customer_support_tickets`, `customer_support_policies`, `customer_orders`,
# MAGIC plus a Volume with policy documents (used by the Knowledge Assistant exercise in `06_agent_bricks`).

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

data = [
    (101, "My order has not arrived. It is 5 days late.", "Laptop", 1200, "Shipping"),
    (102, "The headphones stopped working after 2 days.", "Headphones", 150, "Product Defect"),
    (103, "I want to return my laptop. It is too expensive.", "Laptop", 1200, "Return"),
    (104, "I was charged twice for the same order.", "Mobile", 800, "Payment"),
    (105, "Excellent service. My order arrived early.", "Keyboard", 75, "Positive Feedback"),
    (106, "I received the wrong product. Please help.", "Monitor", 400, "Wrong Item"),
]
columns = ["ticket_id", "customer_message", "product", "order_value", "expected_category"]
df = spark.createDataFrame(data, columns)
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("customer_support_tickets")
display(spark.table("customer_support_tickets"))

# COMMAND ----------

policies = [
    ("Shipping", "Orders delayed more than 3 days qualify for shipping investigation."),
    ("Return", "Customers can return products within 30 days."),
    ("Product Defect", "Defective products qualify for replacement within 15 days."),
    ("Payment", "Duplicate payments must be investigated and refunded if confirmed."),
    ("Wrong Item", "Incorrectly delivered products qualify for free replacement."),
]
policy_df = spark.createDataFrame(policies, ["policy_category", "policy_text"])
policy_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("customer_support_policies")
display(spark.table("customer_support_policies"))

# COMMAND ----------

orders = [
    (101, "ORD1001", "Shipped", 5),
    (102, "ORD1002", "Delivered", 0),
    (103, "ORD1003", "Processing", 2),
]
orders_df = spark.createDataFrame(orders, ["ticket_id", "order_id", "order_status", "delay_days"])
orders_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("customer_orders")
display(spark.table("customer_orders"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Policy documents in a Volume (source for the Knowledge Assistant)

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS policy_docs")
vol = f"/Volumes/{CATALOG}/{SCHEMA}/policy_docs"
for category, text in policies:
    fname = category.lower().replace(" ", "_") + "_policy.md"
    dbutils.fs.put(f"{vol}/{fname}", f"# {category} Policy\n\n{text}\n", overwrite=True)
display(dbutils.fs.ls(vol))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT t.ticket_id, t.customer_message, t.expected_category, p.policy_text, o.order_status, o.delay_days
# MAGIC FROM customer_support_tickets t
# MAGIC LEFT JOIN customer_support_policies p ON t.expected_category = p.policy_category
# MAGIC LEFT JOIN customer_orders o ON t.ticket_id = o.ticket_id
# MAGIC ORDER BY t.ticket_id
