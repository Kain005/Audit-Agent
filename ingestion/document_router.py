"""Document routing logic for finance audit ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_bank_parser import parse_image_pdf_with_llm
import pandas as pd

from .bank_statement_parser import BankStatementParser
from .expense_parser import ExpenseParser
from .invoice_parser import InvoiceParser
from .gst_parser import GSTParser
from . import gst_invoice_parser
from .ledger_parser import LedgerParser
from .pdf_parser import PDFParser
from .bank_pdf_parser import parse_bank_pdf, to_standard_transactions
from ingestion.pdf_parser import extract_image_pdf_full, detect_pdf_type_and_scale
from ingestion.bank_pdf_parser import parse_dataframe_transactions


class DocumentRouter:
    """Route financial documents to the best available parser."""

    EXPENSE_STANDARD_COLUMNS = [
        "date",
        "description",
        "debit",
        "credit",
        "balance",
        "transaction_id",
        "bank_name",
        "category",
        "submitted_by",
    ]

    def __init__(self) -> None:
        self.bank_parser = BankStatementParser()
        self.expense_parser = ExpenseParser()
        self.invoice_parser = InvoiceParser()
        self.gst_parser = GSTParser()
        self.ledger_parser = LedgerParser()
        self.pdf_parser = PDFParser()

    def route(self, file_path: str) -> dict[str, Any]:
        """Route a file to matching parser and return parsed payload."""
        path = Path(file_path)
        parse_errors: list[str] = []

        result: dict[str, Any] = {
            "document_type": "unknown",
            "parser_used": "none",
            "data": pd.DataFrame(),
            "raw_text": None,
            "bank_name": None,
            "file_name": path.name,
            "parse_errors": parse_errors,
        }

        try:
            suffix = path.suffix.lower()

            if suffix == ".pdf":
                pdf_type, render_scale = detect_pdf_type_and_scale(str(path))

                if pdf_type == "image_pdf":
                    pages = extract_image_pdf_full(str(path))
                    raw_text = " ".join(
                        p.get("raw_text", "") for p in pages if p.get("raw_text")
                    )
                    all_rows: list[dict] = []
                    for page in pages:
                        df = page.get("dataframe")
                        if df is not None and not df.empty:
                            transactions = parse_dataframe_transactions(df)
                            all_rows.extend(transactions)
                    if all_rows:
                        data = pd.DataFrame(all_rows)
                    else:
                        data = pd.DataFrame()
                else:
                    pdf_result = self.pdf_parser.parse(str(path))
                    raw_text = str(pdf_result.get("raw_text") or "")
                    data = pdf_result.get("data", pd.DataFrame())

                result["raw_text"] = raw_text
                result["data"] = data
                result["parser_used"] = "ImagePDFParser" if pdf_type == "image_pdf" else "PDFParser"

                if gst_invoice_parser.looks_like_gst_invoice_pdf(str(path)):
                    try:
                        invoice_entities = gst_invoice_parser.parse_pdf(str(path))
                        result["document_type"] = "gst_invoice"
                        result["invoice_data"] = invoice_entities.model_dump()
                        result["data"] = pd.DataFrame()
                        result["parser_used"] = "GSTInvoiceParser"
                        return result
                    except Exception as exc:
                        result["parse_errors"].append(f"GSTInvoiceParser failed: {exc}")

                if self._looks_like_invoice(path.name, raw_text):
                    llm_result = self.invoice_parser.parse_pdf_with_llm(str(path))
                    if "error" in llm_result:
                        result["document_type"] = "invoice_error"
                        result["parse_errors"].append(llm_result["error"])
                        result["parser_used"] = "InvoiceParser"
                    else:
                        result["document_type"] = "invoice"
                        result["invoice_data"] = llm_result["invoice_data"]
                        result["data"] = pd.DataFrame()
                        result["parser_used"] = "InvoiceParserLLM"
                    return result

                if self._looks_like_bank_statement(raw_text=raw_text, df=data):
                    result["document_type"] = "bank_statement"

                    if pdf_type == "image_pdf":
                        try:
                            bank_name_hint = "generic"
                            llm_df = parse_image_pdf_with_llm(str(path), bank_name=bank_name_hint)
                            if not llm_df.empty:
                                result["transactions"] = llm_df.to_dict(orient="records")
                                result["bank_name"] = bank_name_hint
                                result["parser_used"] = "LLMBankParser"
                                return result
                        except Exception as exc:
                            result["parse_errors"].append(f"LLMBankParser failed: {exc}")
                    
                        # fallback: regex OCR text parser
                        ocr_text = raw_text
                        if not ocr_text:
                            try:
                                pages = extract_image_pdf_full(str(path))
                                ocr_text = " ".join(p.get("raw_text", "") for p in pages if p.get("raw_text"))
                            except Exception:
                                ocr_text = ""
                        ocr_df = self._parse_ocr_text_to_transactions(ocr_text)
                        if not ocr_df.empty:
                            result["data"] = ocr_df
                            result["parser_used"] = "OCRTextParser"
                        return result

                    # native text PDF — use existing parse_bank_pdf path
                    try:
                        parsed = parse_bank_pdf(str(path))
                        bank_name = str(parsed.get("bank", "generic") or "generic")
                        used_ocr_fallback = False

                        if not parsed.get("transactions") and parsed.get("raw_text"):
                            df = self._parse_ocr_text_to_transactions(
                                str(parsed.get("raw_text") or ""), bank_name
                            )
                            used_ocr_fallback = True
                        else:
                            df = pd.DataFrame(to_standard_transactions(parsed))

                        result["bank_name"] = bank_name
                        result["data"] = df
                        result["parser_used"] = "OCRTextParser" if used_ocr_fallback else "BankStatementParser"

                        if parsed.get("partial"):
                            result["parse_warnings"] = parsed.get("parse_warnings", [])

                        return result
                    except Exception:
                        result["document_type"] = "bank_statement"
                        if result["data"].empty and raw_text:
                            ocr_df = self._parse_ocr_text_to_transactions(raw_text)
                            if not ocr_df.empty:
                                result["data"] = ocr_df
                                result["parser_used"] = "OCRTextParser"
                            else:
                                result["parser_used"] = "PDFParser"
                        else:
                            result["parser_used"] = "PDFParser"
                        return result

                if self.expense_parser.looks_like_expense_sheet(data):
                    result["data"] = self.expense_parser.parse(str(path))
                    result["document_type"] = "expense_sheet"
                    result["parser_used"] = "ExpenseParser"
                    return result

                result["document_type"] = "unknown"
                return result

            if suffix == ".csv":
                try:
                    preview_df = pd.read_csv(path)

                    if self.expense_parser.looks_like_expense_sheet(preview_df):
                        data = self.expense_parser.parse(str(path))
                        result["data"] = data
                        result["document_type"] = "expense_sheet"
                        result["parser_used"] = "ExpenseParser"
                        return result

                    data = self.bank_parser.parse_csv(str(path))
                    if data.empty:
                        raise ValueError("Bank parser returned empty dataframe")

                    result["data"] = data
                    result["document_type"] = "bank_statement"
                    result["parser_used"] = "BankStatementParser"
                    # propagate detected bank name when available
                    if "bank_name" in data.columns and not data.empty:
                        result["bank_name"] = str(data["bank_name"].iloc[0])
                    return result
                except Exception as exc:
                    parse_errors.append(f"CSV routing failed: {exc}")
                    generic_df = pd.read_csv(path)
                    result["data"] = generic_df
                    result["document_type"] = "expense_sheet" if self.expense_parser.looks_like_expense_sheet(generic_df) else "unknown"
                    result["parser_used"] = "pandas.read_csv"
                    return result

            if suffix in {".xlsx", ".xls", ".xlsm"}:
                if self.gst_parser.looks_like_gst_file(str(path)):
                    gst_result = self.gst_parser.parse(str(path))
                    result["data"] = gst_result["data"]
                    result["document_type"] = "gst"
                    result["parser_used"] = "GSTParser"
                    result["gst_type"] = gst_result["gst_type"]
                    return result

                preview_df = pd.read_excel(path, nrows=50)
                if self.invoice_parser.looks_like_invoice_xlsx(str(path), preview_df):
                    invoice_result = self.invoice_parser.parse(str(path))
                    result["data"] = invoice_result["data"]
                    result["document_type"] = "invoice"
                    result["parser_used"] = "InvoiceParser"
                    result["invoice_data"] = invoice_result["data"].to_dict(orient="records")
                    return result

                workbook = pd.ExcelFile(path)

                expense_detected = False
                for sheet_name in workbook.sheet_names:
                    sheet_df = workbook.parse(sheet_name, nrows=50)
                    if self.expense_parser.looks_like_expense_sheet(sheet_df):
                        expense_detected = True
                        break

                if expense_detected:
                    data = self.expense_parser.parse(str(path))
                    result["data"] = data
                    result["document_type"] = "expense_sheet"
                    result["parser_used"] = "ExpenseParser"
                else:
                    data = self.ledger_parser.parse(str(path))
                    result["data"] = data
                    result["document_type"] = "ledger"
                    result["parser_used"] = "LedgerParser"
                return result

            parse_errors.append(f"Unsupported file extension: {suffix or '<none>'}")
            return result

        except Exception as exc:
            parse_errors.append(str(exc))
            return result

    def process_batch(self, file_paths: list[str]) -> list[dict[str, Any]]:
        """Process multiple files and continue on per-file failures."""
        results: list[dict[str, Any]] = []
        for file_path in file_paths:
            try:
                results.append(self.route(file_path))
            except Exception as exc:
                path = Path(file_path)
                results.append(
                    {
                        "document_type": "unknown",
                        "parser_used": "none",
                        "data": pd.DataFrame(),
                        "raw_text": None,
                        "file_name": path.name,
                        "parse_errors": [f"Unhandled error while routing file: {exc}"],
                    }
                )
        return results

    def _parse_ocr_text_to_transactions(self, raw_text: str, bank_name: str = "generic") -> pd.DataFrame:
        """Parse OCR text into a standard transactions DataFrame when pdfplumber tables are empty.

        Scans lines for date + amount patterns typical of Indian bank statements.
        Returns empty DataFrame if no parseable rows found.
        """
        import re
        from ingestion.parsers.normalizers import clean_amount, clean_date

        DATE_RE = re.compile(
            r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})"
            r"|\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
            re.IGNORECASE,
        )
        AMOUNT_RE = re.compile(r"[\d,]+\.?\d*")

        rows: list[dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split("|") if part.strip()]
            if not parts:
                continue

            date_value = None
            date_index = None
            for index, part in enumerate(parts):
                date_match = DATE_RE.search(part)
                if date_match:
                    matched_str = date_match.group(1) or date_match.group(2) or ""
                    date_value = clean_date(matched_str.strip())
                    date_index = index
                    break

            if not date_value:
                continue

            amount_index = None
            amount_token = ""
            balance_token = ""
            for index in range(len(parts) - 1, -1, -1):
                token = parts[index]
                if re.search(r"\b(?:DR|CR)\b", token, re.IGNORECASE):
                    amount_index = index
                    amount_token = token
                    break

            if amount_index is None:
                continue

            for index in range(len(parts) - 1, amount_index, -1):
                token = parts[index]
                if re.search(r"\b(?:DR|CR)\b", token, re.IGNORECASE) or AMOUNT_RE.search(token):
                    balance_token = token
                    break

            if not balance_token:
                balance_token = parts[-1]

            middle_parts = parts[date_index + 1 : amount_index]
            description = " ".join(middle_parts)
            description = description.replace("|", " ")
            description = re.sub(r"\s+", " ", description).strip()

            amount_match = AMOUNT_RE.search(amount_token)
            if not amount_match:
                continue
            amount_value = clean_amount(amount_match.group(0))
            if not amount_value:
                continue

            balance_match = AMOUNT_RE.search(balance_token)
            if not balance_match:
                continue
            balance_value = clean_amount(balance_match.group(0))

            debit = amount_value if re.search(r"\bDR\b", amount_token, re.IGNORECASE) else 0.0
            credit = amount_value if re.search(r"\bCR\b", amount_token, re.IGNORECASE) else 0.0

            rows.append({
                "date": pd.to_datetime(date_value, errors="coerce"),
                "description": description,
                "debit": debit,
                "credit": credit,
                "balance": balance_value,
                "transaction_id": "",
                "bank_name": bank_name,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _looks_like_invoice(self, file_name: str, raw_text: str) -> bool:
        filename = file_name.lower()
        if any(marker in filename for marker in ("invoice", "inv")):
            return True

        text = raw_text.lower()
        invoice_markers = [
            "invoice no",
            "bill to",
            "amount due",
            "total amount",
            "tax invoice",
            "invoice",
        ]
        return any(marker in text for marker in invoice_markers)

    def _looks_like_bank_statement(self, raw_text: str, df: pd.DataFrame) -> bool:
        text = raw_text.lower()
        statement_markers = [
            "account statement",
            "opening balance",
            "closing balance",
            "transaction",
            "debit",
            "credit",
        ]

        if any(marker in text for marker in statement_markers):
            return True

        if not df.empty:
            bank_guess = self.bank_parser.detect_bank(df)
            if bank_guess != "generic":
                return True

            cols = {str(col).lower() for col in df.columns}
            expected = {"date", "description", "debit", "credit", "balance"}
            if len(cols.intersection(expected)) >= 3:
                return True

        return False

    def _normalize_expense_transaction_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        normalized = df.copy()
        for column in self.EXPENSE_STANDARD_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = None

        return normalized
