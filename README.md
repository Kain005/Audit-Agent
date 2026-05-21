# Audit-Agent

AI-powered financial document auditor for Indian businesses. Detects anomalies, flags policy violations, and explains suspicious transactions in plain English.

## Quick Start
1. Clone the repo.
2. Install dependencies with `pip install -r requirements.txt`.
3. Download Ollama from `ollama.com/download/windows`.
4. Pull the model with `ollama pull llama3.1:8b`.
5. Run `run.bat`.
6. Open `http://localhost:8501`.

## Supported Banks
HDFC, ICICI, SBI, PNB, Axis Bank, Kotak, Yes Bank

## Supported Documents
- Bank statements (CSV/PDF)
- Invoices (PDF)
- Expense sheets (Excel/CSV)
- GST documents (Excel)
- Ledger exports (Excel/CSV)

## Tech Stack
| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Llama 3.1 8B via Ollama |
| Entity Extraction | spaCy NER |
| Anomaly Detection | Isolation Forest + Statistical |
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector Search | ChromaDB + sentence-transformers |
| PDF Generation | xhtml2pdf |
