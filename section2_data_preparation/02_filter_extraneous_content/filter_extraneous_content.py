# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.2 – Filter extraneous content that degrades a RAG application
# MAGIC
# MAGIC **The problem (plain English)**
# MAGIC Real documents are full of things that are *not* knowledge: website menus, cookie banners, "© 2026" footers,
# MAGIC page numbers, repeated page headers in PDFs, ads, tracking code, empty lines, and duplicate paragraphs.
# MAGIC If we leave that "junk" in:
# MAGIC - Every chunk starts to look alike (they all contain the same menu), so **search gets worse**.
# MAGIC - The junk takes up space in the model's limited context window – **wasted money and room**.
# MAGIC - The model may **quote junk** ("Subscribe to our newsletter!") in its answer.
# MAGIC
# MAGIC "Garbage in, garbage out": cleaning is one of the cheapest ways to improve a RAG app.

# COMMAND ----------

# MAGIC %md ### Step 1 – A messy web page (what you really get when you download a help page)

# COMMAND ----------

raw_html = """
<html><head><title>Returns | ShopEase Help</title>
<script>trackVisitor('returns-page'); var ads = loadAds();</script>
<style>.banner {color: red}</style></head>
<body>
<div class="cookie-banner">We use cookies to improve your experience. Accept all cookies?</div>
<nav>Home | Shop | Deals | Shipping | Returns | Payments | Contact Us | Login</nav>
<div class="ad">SALE! 50% off headphones this weekend only!</div>
<main>
  <h1>Return Policy</h1>
  <p>Customers can return most products within 30 days of delivery for a full refund.</p>
  <p>Laptops and monitors must be returned in their original packaging.</p>
  <p>Customers can return most products within 30 days of delivery for a full refund.</p>
  <p>Opened software and gift cards cannot be returned.</p>
  <p>   </p>
  <p>Refunds are issued to the original payment method within 5 to 7 business days.</p>
</main>
<footer>© 2026 ShopEase Inc. All rights reserved. | Privacy Policy | Terms of Use | Subscribe to our newsletter!</footer>
</body></html>
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### And text copied out of a PDF (repeated page headers, page numbers, a confidentiality line)

# COMMAND ----------

raw_pdf_text = """ShopEase Customer Handbook - CONFIDENTIAL
Page 1 of 3
Shipping
Standard shipping takes 3 to 5 business days.
Orders delayed more than 3 days qualify for a shipping investigation.
ShopEase Customer Handbook - CONFIDENTIAL
Page 2 of 3
Warranty
Defective products qualify for a free replacement within 15 days of delivery.
ShopEase Customer Handbook - CONFIDENTIAL
Page 3 of 3
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 – Clean the HTML
# MAGIC `BeautifulSoup` (package `beautifulsoup4`, pre-installed on Databricks) reads HTML like a tree, so we can
# MAGIC delete whole parts of the page by their tag (`<nav>`, `<footer>`, `<script>`...) or CSS class (`cookie-banner`, `ad`).

# COMMAND ----------

from bs4 import BeautifulSoup

soup = BeautifulSoup(raw_html, "html.parser")

# 2a. Remove parts of the page that are never knowledge
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()                       # "decompose" = delete this element and everything inside it
for css_class in ["cookie-banner", "ad"]:
    for tag in soup.find_all(class_=css_class):
        tag.decompose()

# 2b. Keep only the visible text of the main content, one line per paragraph
lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]

# 2c. Drop empty lines and exact duplicates (keep the first time we see a line)
seen, clean_lines = set(), []
for line in lines:
    if line and line not in seen:
        seen.add(line)
        clean_lines.append(line)

clean_html_text = "\n".join(clean_lines)
print(clean_html_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 – Clean the PDF text with simple rules ("regular expressions" = text patterns)

# COMMAND ----------

import re

pdf_lines = raw_pdf_text.splitlines()

# 3a. Lines that repeat on every page are almost always headers/footers -> remove them
from collections import Counter
line_counts = Counter(pdf_lines)
repeated = {line for line, count in line_counts.items() if count >= 2}

clean_pdf_lines = []
for line in pdf_lines:
    if line in repeated:
        continue                                        # repeated header
    if re.fullmatch(r"Page \d+ of \d+", line.strip()):
        continue                                        # page number like "Page 2 of 3"
    if not line.strip():
        continue                                        # empty line
    clean_pdf_lines.append(line.strip())

clean_pdf_text = "\n".join(clean_pdf_lines)
print(clean_pdf_text)

# COMMAND ----------

# MAGIC %md ### Step 4 – How much junk did we remove?

# COMMAND ----------

raw_visible = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
print(f"HTML: {len(raw_visible)} chars of visible text -> {len(clean_html_text)} chars after cleaning")
print(f"PDF : {len(raw_pdf_text)} chars -> {len(clean_pdf_text)} chars after cleaning")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 – Proof that junk hurts search
# MAGIC We turn text into **embeddings** (lists of numbers that capture meaning) with a Databricks embedding model,
# MAGIC then measure how similar a question is to each chunk (1.0 = identical meaning, 0 = unrelated).
# MAGIC
# MAGIC We build 3 chunks (returns, shipping, warranty) twice: once with the menu + footer glued on (as a careless
# MAGIC pipeline would), once clean. The question is about returns. With clean chunks, the returns chunk should win by a
# MAGIC **bigger margin** – the junk made all chunks look alike.

# COMMAND ----------

import numpy as np
from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")   # talks to model endpoints in this workspace
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

menu = "Home | Shop | Deals | Shipping | Returns | Payments | Contact Us | Login. We use cookies. Accept all cookies?"
footer = "© 2026 ShopEase Inc. All rights reserved. | Privacy Policy | Terms of Use | Subscribe to our newsletter!"
topics = {
    "returns": "Customers can return most products within 30 days of delivery for a full refund.",
    "shipping": "Standard shipping takes 3 to 5 business days.",
    "warranty": "Defective products qualify for a free replacement within 15 days of delivery.",
}
question = "How many days do I have to send back a product?"

def similarity_to_question(chunks):
    # 1. Send the question + all chunks to the embedding model in one request
    response = client.predict(endpoint=EMBEDDING_ENDPOINT, inputs={"input": [question] + chunks})
    vectors = np.array([item["embedding"] for item in response["data"]])
    # 2. Normalise each vector to length 1, so a dot product = cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors[1:] @ vectors[0]

noisy_scores = similarity_to_question([f"{menu}\n{text}\n{footer}" for text in topics.values()])
clean_scores = similarity_to_question(list(topics.values()))

for name, n, c in zip(topics, noisy_scores, clean_scores):
    print(f"{name:9s} noisy={n:.3f}  clean={c:.3f}")

def margin(scores):
    # how far the correct chunk (returns, index 0) is ahead of the best wrong chunk
    return scores[0] - max(scores[1:])

print(f"\nLead of the correct chunk -> noisy: {margin(noisy_scores):.3f}   clean: {margin(clean_scores):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways – common extraneous content and how to remove it**
# MAGIC
# MAGIC | Junk | Typical fix |
# MAGIC |---|---|
# MAGIC | Navigation menus, footers, cookie banners, ads | Remove HTML tags/classes (BeautifulSoup) |
# MAGIC | Scripts, styles, tracking code | Remove `<script>` / `<style>` tags |
# MAGIC | Repeated PDF headers/footers, page numbers | Drop lines that repeat on every page; regex for "Page X of Y" |
# MAGIC | Duplicate paragraphs / near-duplicate documents | De-duplicate (exact match or similarity) |
# MAGIC | Empty lines, odd whitespace, broken characters | Normalise whitespace / encoding |
# MAGIC | Legal boilerplate, disclaimers, signatures | Rules or an LLM classifier to drop non-knowledge paragraphs |
# MAGIC
# MAGIC Clean **before** chunking, so junk never reaches the index.
