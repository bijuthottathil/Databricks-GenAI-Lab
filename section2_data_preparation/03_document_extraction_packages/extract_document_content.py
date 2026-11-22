# Databricks notebook source
# MAGIC %md
# MAGIC # Section 2.3 – Choose the right Python package to extract document content
# MAGIC
# MAGIC **Plain English:** before we can chunk and search documents, we must pull the *text* out of them.
# MAGIC Every file format stores text differently (a PDF is drawing instructions, a Word file is zipped XML, a web page
# MAGIC is HTML), so each format has its own "reader" package.
# MAGIC
# MAGIC | Source format | Recommended package | Why |
# MAGIC |---|---|---|
# MAGIC | PDF (digital text) | `pypdf` (or `PyMuPDF` / `pdfplumber` for layout & tables) | Reads the text layer page by page |
# MAGIC | PDF (scanned image) | OCR: `pytesseract`, `unstructured`, or Databricks `ai_parse_document` | No text layer – needs image-to-text |
# MAGIC | Word `.docx` | `python-docx` | Reads paragraphs, headings and tables |
# MAGIC | PowerPoint `.pptx` | `python-pptx` | Reads text from each slide's shapes |
# MAGIC | HTML / web pages | `beautifulsoup4` | Parses tags so you can drop menus and keep content |
# MAGIC | CSV / Excel | `pandas` (`openpyxl` for `.xlsx`) | Already tabular – read rows directly |
# MAGIC | JSON | built-in `json` | Already structured |
# MAGIC | Markdown / plain text | built-in `open()` | Already text |
# MAGIC | Many mixed formats | `unstructured` | One API that detects and partitions many formats |
# MAGIC
# MAGIC In this notebook we **create one sample file of each format** in a Unity Catalog **Volume** (a folder for files
# MAGIC that lives in Unity Catalog), then **read each one back with the right package**.

# COMMAND ----------

# MAGIC %md ### Step 0 – Install the reader/writer packages

# COMMAND ----------

# MAGIC %pip install -q pypdf fpdf2 python-docx python-pptx beautifulsoup4
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md ### Step 1 – Choose where to store the files (catalog → schema → volume)

# COMMAND ----------

CATALOG = "workspace"            # change if your workspace uses another catalog
SCHEMA = "fde_data_prep"
VOLUME = "source_documents"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
folder = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print("Files will be written to", folder)

# COMMAND ----------

# MAGIC %md ### Step 2 – Create one sample file per format (pretend these came from different company teams)

# COMMAND ----------

# NOTE: PDF, Word and PowerPoint writers jump back and forth inside the file while saving (Word/PowerPoint files
# are zip archives). Volumes only support writing files from start to end, so we save these three to the
# machine's local disk (/tmp) first and then copy the finished file into the Volume.
import shutil

def save_via_local_disk(save_function, file_name):
    local_path = f"/tmp/{file_name}"
    save_function(local_path)                          # e.g. doc.save("/tmp/returns_policy.docx")
    shutil.copy(local_path, f"{folder}/{file_name}")   # copy the finished file into the Volume

# --- PDF: the shipping policy -------------------------------------------------
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(0, 8, "Shipping Policy\n\nStandard shipping takes 3 to 5 business days. "
                     "Orders delayed more than 3 days qualify for a shipping investigation.")
pdf.add_page()
pdf.multi_cell(0, 8, "Express shipping takes 1 to 2 business days and costs $15.")
save_via_local_disk(pdf.output, "shipping_policy.pdf")

# --- Word: the returns policy, including a small table -------------------------
from docx import Document
doc = Document()
doc.add_heading("Returns Policy", level=1)
doc.add_paragraph("Customers can return most products within 30 days of delivery for a full refund.")
table = doc.add_table(rows=3, cols=2)
for row, (item, rule) in zip(table.rows, [("Item", "Rule"), ("Laptop", "Original packaging required"), ("Gift card", "Not returnable")]):
    row.cells[0].text, row.cells[1].text = item, rule
save_via_local_disk(doc.save, "returns_policy.docx")

# --- PowerPoint: a training deck about warranty --------------------------------
from pptx import Presentation
deck = Presentation()
slide = deck.slides.add_slide(deck.slide_layouts[1])      # layout 1 = "title + content"
slide.shapes.title.text = "Warranty Training"
slide.placeholders[1].text = "Defective products: free replacement within 15 days.\nAfter 15 days: 1-year manufacturer warranty."
save_via_local_disk(deck.save, "warranty_training.pptx")

# --- HTML: the payments help page ----------------------------------------------
with open(f"{folder}/payments_help.html", "w") as f:
    f.write("<html><body><nav>Home | Help</nav><h1>Payments</h1>"
            "<p>We accept credit cards, debit cards and PayPal.</p>"
            "<p>Duplicate payments are refunded within 3 business days.</p><footer>© ShopEase</footer></body></html>")

# --- CSV: an FAQ exported from a spreadsheet ------------------------------------
with open(f"{folder}/faq.csv", "w") as f:
    f.write("question,answer\n"
            "Can I change my delivery address?,Yes - before the order ships.\n"
            "Do you ship internationally?,Only to Canada and Mexico.\n")

# --- JSON: product data from an API --------------------------------------------
import json
with open(f"{folder}/products.json", "w") as f:
    json.dump([{"product": "Laptop", "warranty_years": 1, "return_days": 30},
               {"product": "Headphones", "warranty_years": 1, "return_days": 30}], f)

# --- Markdown: an internal wiki page --------------------------------------------
with open(f"{folder}/wrong_items.md", "w") as f:
    f.write("# Wrong Items\n\nIf a customer receives the wrong product, send the correct item for free.\n")

display(dbutils.fs.ls(folder))

# COMMAND ----------

# MAGIC %md ### Step 3 – Read each format back with the right package

# COMMAND ----------

# PDF -> pypdf: open the file and read the text layer of every page
from pypdf import PdfReader
reader = PdfReader(f"{folder}/shipping_policy.pdf")
pdf_text = "\n".join(page.extract_text() for page in reader.pages)
print(f"PDF has {len(reader.pages)} pages:\n{pdf_text}")

# COMMAND ----------

# Word -> python-docx: paragraphs AND tables are stored separately, so read both
from docx import Document
doc = Document(f"{folder}/returns_policy.docx")
paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
table_rows = [" | ".join(cell.text for cell in row.cells) for t in doc.tables for row in t.rows]
docx_text = "\n".join(paragraphs + table_rows)
print(docx_text)

# COMMAND ----------

# PowerPoint -> python-pptx: each slide has "shapes"; some shapes hold text
from pptx import Presentation
deck = Presentation(f"{folder}/warranty_training.pptx")
pptx_text = "\n".join(shape.text_frame.text
                      for slide in deck.slides
                      for shape in slide.shapes if shape.has_text_frame)
print(pptx_text)

# COMMAND ----------

# HTML -> BeautifulSoup: drop the menu/footer, keep the real content
from bs4 import BeautifulSoup
soup = BeautifulSoup(open(f"{folder}/payments_help.html").read(), "html.parser")
for tag in soup(["nav", "footer", "script", "style"]):
    tag.decompose()
html_text = soup.get_text("\n", strip=True)
print(html_text)

# COMMAND ----------

# CSV -> pandas: each row becomes one "question + answer" text (one chunk per FAQ item)
import pandas as pd
faq = pd.read_csv(f"{folder}/faq.csv")
csv_texts = [f"Q: {r.question}\nA: {r.answer}" for r in faq.itertuples()]
print("\n\n".join(csv_texts))

# COMMAND ----------

# JSON -> json: turn each record into a readable sentence so the model can use it
records = json.load(open(f"{folder}/products.json"))
json_texts = [f"{r['product']}: {r['warranty_years']}-year warranty, returnable within {r['return_days']} days." for r in records]
print("\n".join(json_texts))

# COMMAND ----------

# Markdown / text -> plain open()
md_text = open(f"{folder}/wrong_items.md").read()
print(md_text)

# COMMAND ----------

# MAGIC %md ### Step 4 – Put everything in one table: one row per document, ready for chunking (see 2.4)

# COMMAND ----------

extracted = [
    ("shipping_policy.pdf", "pdf", "pypdf", pdf_text),
    ("returns_policy.docx", "docx", "python-docx", docx_text),
    ("warranty_training.pptx", "pptx", "python-pptx", pptx_text),
    ("payments_help.html", "html", "beautifulsoup4", html_text),
    ("faq.csv", "csv", "pandas", "\n\n".join(csv_texts)),
    ("products.json", "json", "json", "\n".join(json_texts)),
    ("wrong_items.md", "markdown", "open()", md_text),
]
docs_df = spark.createDataFrame(extracted, "file_name string, file_format string, package_used string, text string")
docs_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.extracted_documents")
display(spark.table(f"{CATALOG}.{SCHEMA}.extracted_documents"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional – Databricks' built-in parser (no Python package needed)
# MAGIC `ai_parse_document` is a SQL function that reads PDFs, Word, PowerPoint and images (including scanned pages)
# MAGIC directly from a Volume. It may not be enabled in every workspace, so we try it and explain if it is missing.

# COMMAND ----------

try:
    parsed = spark.sql(f"""
        SELECT path, ai_parse_document(content) AS parsed
        FROM READ_FILES('{folder}/*.pdf', format => 'binaryFile')
    """)
    display(parsed)
except Exception as e:
    print("ai_parse_document is not available in this workspace:", str(e)[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC **Exam takeaways**
# MAGIC - Pick the package by **format**: PDF → `pypdf`/`PyMuPDF`; DOCX → `python-docx`; PPTX → `python-pptx`;
# MAGIC   HTML → `BeautifulSoup`; CSV/XLSX → `pandas`; JSON → `json`.
# MAGIC - Scanned documents have no text layer → OCR (`pytesseract`, `unstructured`, `ai_parse_document`).
# MAGIC - Mixed formats at scale → `unstructured` or `ai_parse_document`.
# MAGIC - Don't forget **tables** (Word/PDF) – plain text extraction can skip or scramble them.
