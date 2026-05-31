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
| Invoices | PDF |
| Expense Sheets | Excel, CSV |
| GST Documents | Excel |
| Ledger Exports | Excel, CSV |

## Supported Banks

HDFC, ICICI, SBI, BOB, PNB, Axis Bank, Kotak, Yes Bank

---

## Anomaly Detection

### Invoice Checks
- Duplicate invoice detection
- Near-duplicate invoices
- Same amount from same vendor
- Unusually high invoice amounts
- Amount clustering patterns
- Vendor sudden activity spikes
- One-time vendor flagging
- Transaction frequency anomalies

### GST Validation
- GSTIN format and name mismatch
- CGST + SGST arithmetic correctness
- GST rate consistency across line items
- Interstate vs intrastate tax classification

### Bank Statement Checks
- Amount outliers (z-score + IQR)
- Split billing detection
- Timing anomalies (weekend payments, off-hours)

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
