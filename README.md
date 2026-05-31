# Audit-Agent

AI-powered financial document auditor for Indian businesses. Detects anomalies, flags policy violations, and explains suspicious transactions in plain English.

---

## Quick Start

1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY to .env
   ```
4. Download Ollama from [ollama.com/download/windows](https://ollama.com/download/windows).
5. Pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```
6. Run the app:
   ```bash
   run.bat
   ```
7. Open [http://localhost:8501](http://localhost:8501).

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Gemini is used for parsing image-based PDF bank statements (e.g. scanned SBI, BOB statements).

---

## What It Does

Upload financial documents and the agent will:

- Automatically detect the document type
- Extract and parse transactions
- Run anomaly and policy violation checks
- Score overall risk from 0–100
- Generate a full audit report (PDF or JSON)
- Explain individual findings in plain English via Ollama

---

## Supported Documents

| Document Type | Formats |
|---|---|
| Bank Statements | CSV, PDF (native + scanned) |
| Invoices | PDF, Excel (auto-detects GST vs regular) |
| Expense Sheets | Excel, CSV |
| GST Documents | Excel |
| Ledger Exports | Excel, CSV |

## Supported Banks

HDFC, ICICI, SBI, BOB, PNB, Axis Bank, Kotak, Yes Bank

---

## Anomaly Detection

### Bank Statements

**Amount**
- Amount outliers — z-score and IQR based, flags unusually high/low transactions compared to account history

**Timing**
- Weekend payments above ₹10,000
- Late night / outside business hours transactions

**Frequency**
- Vendor transaction frequency spike — current month count exceeds 2–3x historical monthly average

**Pattern**
- Split billing — multiple payments to same vendor within rolling window, all below approval threshold but combined amount exceeds it
- Just-below-threshold payments — amounts clustered between 95–99% of approval threshold
- Rapid sequential payments to same beneficiary

**Multivariate**
- IsolationForest on combined amount + timing + frequency — flags transactions that appear normal individually but anomalous together

**Vendor**
- Blacklisted vendor check — fuzzy name match and exact GSTIN match against blacklist
- Related party detection — vendor name similarity ≥ 80% match to employee name

**GST Cross-reference** (when GST docs also uploaded)
- Unmatched invoices — invoices with no corresponding bank payment
- Unmatched payments — bank payments with no corresponding invoice

---

### Invoices

Invoices are auto-detected as GST or regular based on file content (presence of GSTIN, CGST/SGST, HSN/SAC codes). All invoices go through the same `evaluate_invoice` function — GST checks simply return no finding if GST data is absent.

**Duplicate Detection**
- Exact duplicate — same invoice number and vendor
- Near-duplicate — similar invoice number (≥80% similarity), same amount, within 30 days
- Same amount from same vendor — different invoice numbers but identical amounts

**Vendor Risk**
- One-time vendor with high value and no GSTIN
- Vendor sudden activity — 3+ invoices in 30 days with no prior history
- Vendor name similar to employee name — possible related party fraud

**Amount**
- Unusually high invoice — z-score ≥ 2.5 standard deviations above vendor mean
- Amount clustering — multiple invoices just below approval threshold from same vendor
- Invoice amount exceeds approval threshold

**Frequency**
- Invoice frequency anomaly — 3x spike vs monthly average for the vendor

**GST** (runs on all invoices; no finding returned if GST data absent)
- GST consistency — CGST + SGST arithmetic checked against total within tolerance
- CGST/SGST split — CGST ≠ SGST flagged for intra-state supply
- Interstate tax type — CGST/SGST used instead of IGST for cross-state supply (and vice versa)
- GSTIN name mismatch — same GSTIN appears with different vendor names across invoices

**Timing**
- Weekend invoice — invoice dated on Saturday/Sunday above ₹10,000

**Payments**
- Early payment — paid before due date under Net terms for amounts above ₹50,000
- Late payment — paid more than 15 days after due date

---

### Expense Sheets

- Weekend high-amount expense — expense above ₹5,000 on a weekend
- Consecutive day submissions — same employee submits expenses for 3+ consecutive days
- Late submission — expense submitted more than 30 days after expense date
- Same-day possible split — 3+ expenses by same employee on same day
- Round large amount — expense ≥ ₹10,000 that is a multiple of ₹5,000
- Category daily limit breach — amount exceeds per-category daily cap
- Employee monthly limit breach — total monthly spend exceeds per-employee cap
- Missing receipt — amount above receipt threshold (default ₹5,000) with no receipt attached

---

## App Features

- Upload PDF, CSV, XLSX documents via UI
- Automatic document type detection
- Native PDF parsing via pdfplumber
- Scanned/image PDF parsing via Gemini Flash Lite
- OCR pipeline via EasyOCR for image PDFs
- Debit / credit / balance extraction
- Risk score 0–100 per document
- Vendor token extraction
- **Findings tab** — filterable by severity and type
- **Analytics tab** — severity distribution and finding type breakdown
- **Documents tab** — findings grouped by document, GST invoice detail cards
- **Full Report tab** — executive summary and risk assessment
- PDF and JSON report download
- On-demand finding explanation via Ollama

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Pipeline | LangGraph 1.1.9 |
| LLM (explanations) | Llama 3.1 8B via Ollama |
| LLM (image PDFs) | Gemini Flash Lite |
| Entity Extraction | spaCy NER |
| Anomaly Detection | Isolation Forest, z-score, IQR |
| OCR Engine | EasyOCR 1.7.2 |
| PDF Rendering | pypdfium2 |
| Native PDF Parsing | pdfplumber |
| Table Detection | OpenCV |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Vector Search | ChromaDB + sentence-transformers |
| Job Persistence | SQLAlchemy |
| Background Jobs | ThreadPoolExecutor |
| Report Generation | xhtml2pdf |
