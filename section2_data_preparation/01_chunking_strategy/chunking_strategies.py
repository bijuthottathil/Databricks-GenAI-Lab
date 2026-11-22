# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.1 – Apply a chunking strategy for a document structure and model constraints
# MAGIC
# MAGIC **What is "chunking"? (plain English)**
# MAGIC A RAG app ("Retrieval-Augmented Generation") answers questions by first *finding* the right piece of your
# MAGIC documents and then handing that piece to the AI model. Whole documents are usually too big to hand over, and
# MAGIC searching inside a giant blob is imprecise. So we cut documents into smaller pieces called **chunks**.
# MAGIC
# MAGIC **How big should a chunk be?** It depends on two things:
# MAGIC 1. **The document's structure** – a handbook with headings should be cut at the headings; a plain wall of text
# MAGIC    can be cut by size or by sentences; a table/FAQ should keep each question+answer together.
# MAGIC 2. **The model's limits** – the *embedding model* (turns text into numbers for searching) can only read so many
# MAGIC    tokens at once (e.g. `databricks-bge-large-en` reads up to 512 tokens). The *chat model* has a context window
# MAGIC    that must hold the question + all retrieved chunks + the answer.
# MAGIC    *(A "token" is roughly ¾ of a word, so 512 tokens ≈ 380 words.)*
# MAGIC
# MAGIC | Strategy | How it cuts | Best for |
# MAGIC |---|---|---|
# MAGIC | Fixed-size | Every N characters | Unstructured text, quick baseline |
# MAGIC | Fixed-size + overlap | Every N characters, repeating the last few | Keeps sentences that fall on a boundary |
# MAGIC | Sentence / recursive | At paragraph → sentence → word boundaries | Normal prose |
# MAGIC | Structure-aware (headers) | At document headings | Manuals, handbooks, Markdown/HTML |
# MAGIC | One-record-per-chunk | Each FAQ item / table row | FAQs, tables, tickets |

# COMMAND ----------

# MAGIC %md ### Step 0 – Install the text-splitting library
# MAGIC `langchain-text-splitters` is a small, popular package that already knows how to split text in the ways above.

# COMMAND ----------

# MAGIC %pip install -q langchain-text-splitters
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md ### Step 1 – Our sample document: a small company handbook written in Markdown
# MAGIC Markdown uses `#` for headings – that structure will matter later.

# COMMAND ----------

handbook = """# ShopEase Customer Handbook

## 1. Shipping
Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs $15.
Orders delayed more than 3 days beyond the promised date qualify for a shipping investigation.
If an investigation confirms the delay was our fault, the customer receives a $10 store credit.

## 2. Returns
Customers can return most products within 30 days of delivery for a full refund.
Laptops and monitors must be returned in their original packaging.
Opened software and gift cards cannot be returned.
Refunds are issued to the original payment method within 5 to 7 business days.

## 3. Product Defects and Warranty
Defective products qualify for a free replacement within 15 days of delivery.
After 15 days, defects are covered by the manufacturer warranty, which lasts 1 year for electronics.
Customers must provide a photo or video of the defect.

## 4. Payments
We accept credit cards, debit cards, and PayPal.
Duplicate payments must be investigated and are refunded within 3 business days if confirmed.

## 5. Wrong Items
If a customer receives the wrong product, we send the correct item for free and provide a prepaid return label.
"""

print(f"Document length: {len(handbook)} characters, ~{int(len(handbook.split()) * 1.3)} tokens")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A helper to *look* at chunks
# MAGIC (Defined right here in the notebook – it only prints things.)

# COMMAND ----------

def show_chunks(chunks, title):
    print(f"===== {title}: {len(chunks)} chunks =====")
    for i, c in enumerate(chunks):
        text = c if isinstance(c, str) else c.page_content
        # word count × 1.3 is a quick, rough estimate of the token count
        print(f"--- chunk {i}  ({len(text)} chars, ~{int(len(text.split()) * 1.3)} tokens)\n{text}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1 – Fixed-size chunks (every 200 characters, no overlap)
# MAGIC Simple, but notice how it cuts **in the middle of words and sentences**. A chunk may start halfway through
# MAGIC the returns policy and end halfway through the warranty policy – confusing for search and for the model.

# COMMAND ----------

chunk_size = 200
fixed_chunks = [handbook[i:i + chunk_size] for i in range(0, len(handbook), chunk_size)]
show_chunks(fixed_chunks[:4], "Fixed size (first 4 shown)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 2 – Fixed size **with overlap**
# MAGIC Each chunk repeats the last 50 characters of the previous one, so a sentence cut at a boundary still appears
# MAGIC whole in at least one chunk. Cost: more chunks and some duplicated text.

# COMMAND ----------

overlap = 50
step = chunk_size - overlap
overlap_chunks = [handbook[i:i + chunk_size] for i in range(0, len(handbook), step)]
show_chunks(overlap_chunks[:3], "Fixed size + overlap (first 3 shown)")
print(f"No overlap: {len(fixed_chunks)} chunks  |  with overlap: {len(overlap_chunks)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 3 – Recursive (paragraph → sentence → word) splitting
# MAGIC Tries to cut at a blank line first; if a piece is still too big, at a new line; then at ". "; then at spaces.
# MAGIC Chunks now end at natural boundaries.

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # maximum characters per chunk
    chunk_overlap=30,      # characters repeated between neighbouring chunks
    separators=["\n\n", "\n", ". ", " "],  # the order of "good places to cut"
)
recursive_chunks = recursive_splitter.split_text(handbook)
show_chunks(recursive_chunks, "Recursive")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 4 – Structure-aware: split on the document's headings
# MAGIC Because the handbook has `##` headings, each policy becomes exactly one chunk, and the heading is kept as
# MAGIC **metadata** (extra labels stored next to the chunk – handy for filtering and for showing citations).

# COMMAND ----------

from langchain_text_splitters import MarkdownHeaderTextSplitter

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "document"), ("##", "section")])
header_chunks = header_splitter.split_text(handbook)
for c in header_chunks:
    print(c.metadata, "->", c.page_content[:80], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 5 – One record per chunk (FAQ)
# MAGIC For a FAQ, each question and its answer belong together. Splitting by size could separate a question
# MAGIC from its answer, so we split on the "Q:" marker instead.

# COMMAND ----------

faq = """Q: How long does standard shipping take?
A: 3 to 5 business days.

Q: Can I return opened software?
A: No, opened software cannot be returned.

Q: Which payment methods do you accept?
A: Credit cards, debit cards and PayPal."""

faq_chunks = [block.strip() for block in faq.split("\n\n")]
show_chunks(faq_chunks, "FAQ: one Q&A per chunk")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Respecting model constraints
# MAGIC Rule of thumb: **every chunk must fit in the embedding model's limit**, and
# MAGIC `top_k × chunk size + question + answer` must fit in the chat model's context window.
# MAGIC Below we check each strategy against a 512-token embedding limit and a small 2,000-token budget for
# MAGIC "retrieved text we are allowed to send to the chat model".

# COMMAND ----------

EMBEDDING_TOKEN_LIMIT = 512     # databricks-bge-large-en reads at most 512 tokens per input
CONTEXT_BUDGET_TOKENS = 2000    # how much retrieved text we allow in the chat prompt
TOP_K = 3                       # how many chunks we plan to retrieve per question

def approx_tokens(text):
    return int(len(text.split()) * 1.3)

for name, chunks in [("fixed", fixed_chunks), ("overlap", overlap_chunks),
                     ("recursive", recursive_chunks), ("headers", [c.page_content for c in header_chunks])]:
    biggest = max(approx_tokens(c) for c in chunks)
    fits_embedding = biggest <= EMBEDDING_TOKEN_LIMIT
    fits_context = biggest * TOP_K <= CONTEXT_BUDGET_TOKENS
    print(f"{name:10s} chunks={len(chunks):3d}  biggest≈{biggest:4d} tokens  "
          f"fits embedding limit: {fits_embedding}  top-{TOP_K} fits context budget: {fits_context}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What if a section is too big for the model?
# MAGIC Combine strategies: split by headings first (keeps topics together), then split any section that is still too
# MAGIC large with the recursive splitter, carrying the heading metadata along.

# COMMAND ----------

small_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
combined = small_splitter.split_documents(header_chunks)   # keeps each chunk's metadata
for c in combined[:6]:
    print(c.metadata.get("section"), "|", c.page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Structured documents (headings, HTML, Markdown) → split on structure, keep headings as metadata.
# MAGIC - Unstructured prose → recursive/sentence splitting with a little overlap.
# MAGIC - FAQs / tables → one logical record per chunk.
# MAGIC - Chunk size is bounded by the embedding model's token limit and by the chat model's context window ÷ top-k.
# MAGIC - Too small = chunks lose context; too large = imprecise search and wasted context window.
