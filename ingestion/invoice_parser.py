"""Invoice parsing utilities for XLSX and PDF invoice documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

from .parsers.normalizers import clean_amount, clean_date


INVOICE_PROMPT = """Extract invoice data from the text below. Return ONLY a JSON object with these keys: vendor_name, vendor_gst, vendor_address, invoice_number, invoice_date, due_date, subtotal, gst_amount, total_amount, payment_terms, currency, line_items (array of: description, quantity, unit_price, amount, gst_rate). No markdown, no explanation.

{raw_text}"""

class InvoiceParser:
    """Parse invoice files into a normalized dataframe."""

    XLSX_COLUMN_ALIASES: dict[str, list[str]] = {
        "invoice_number": ["invoice_number", "invoice_no", "inv_no", "bill_number", "bill no", "doc no"],
        "date": ["date", "invoice_date", "bill_date", "inv_date", "document_date"],
        "vendor": ["vendor", "supplier", "bill_to", "bill to", "party", "customer"],
        "amount": ["amount", "amount_due", "net_amount", "subtotal", "taxable_amount"],
        "tax": ["tax", "gst", "tax_amount", "igst", "cgst", "sgst"],
        "total": ["total", "grand_total", "invoice_total", "amount_due", "total_amount"],
    }

    XLSX_FILENAME_MARKERS = ("invoice", "inv")

    def parse(self, file_path: str) -> dict[str, Any]:
        """Parse invoice file and return normalized dataframe payload."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".xls", ".xlsm"}:
            data = self.parse_xlsx(file_path)
            return {"document_type": "invoice", "parser_used": "InvoiceParser", "data": data, "source_file": path.name}

        if suffix == ".pdf":
            data = self.parse_pdf(file_path)
            return {"document_type": "invoice", "parser_used": "InvoiceParser", "data": data, "source_file": path.name}

        raise ValueError(f"Unsupported invoice file type: {suffix or '<none>'}")

    def looks_like_invoice_xlsx(self, file_path: str, df: pd.DataFrame) -> bool:
        """Detect invoice-like spreadsheets by filename keywords or invoice columns."""
        path = Path(file_path)
        filename = path.stem.lower()
        if any(marker in filename for marker in self.XLSX_FILENAME_MARKERS):
            return True

        if df.empty:
            return False

        columns = {str(column).strip().lower() for column in df.columns}
        hits = 0
        for aliases in self.XLSX_COLUMN_ALIASES.values():
            if any(alias in columns or any(alias in column for column in columns) for alias in aliases):
                hits += 1
        return hits >= 3

    def parse_xlsx(self, file_path: str) -> pd.DataFrame:
        """Normalize invoice spreadsheets into a standard dataframe."""
        path = Path(file_path)
        workbook = pd.ExcelFile(path)

        best_sheet = None
        best_score = -1
        best_df = pd.DataFrame()

        for sheet_name in workbook.sheet_names:
            candidate = workbook.parse(sheet_name)
            score = self._score_invoice_columns(candidate)
            if score > best_score:
                best_score = score
                best_sheet = sheet_name
                best_df = candidate

        if best_sheet is None or best_df.empty:
            return pd.DataFrame(columns=["invoice_number", "date", "vendor", "amount", "tax", "total", "source_file"])

        normalized = self._normalize_xlsx(best_df)
        normalized["source_file"] = path.name
        return normalized

    def parse_pdf(self, file_path: str) -> pd.DataFrame:
        """Extract invoice fields from PDF text using regex and tables when available."""
        path = Path(file_path)
        raw_text, table_df = self._extract_pdf_content(path)

        invoice_number = self._extract_first(
            raw_text,
            [
                r"Invoice\s*No[\.:]?\s*([A-Z0-9/-]+)",
                r"Invoice\s*#\s*([A-Z0-9/-]+)",
                r"Bill\s*No[\.:]?\s*([A-Z0-9/-]+)",
                r"INV[-/]([A-Z0-9-]+)",
            ],
        )
        invoice_date = self._extract_first(
            raw_text,
            [
                r"Invoice\s*Date[\s:.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"Date[\s:.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
                r"Invoice\s*Date[\s:.-]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            ],
        )
        vendor = self._extract_first(
            raw_text,
            [r"Bill\s*To[:\s]+(.+)", r"Vendor[:\s]+(.+)", r"From[:\s]+(.+)"],
        )
        amount = self._extract_amount(raw_text, [r"Amount\s*Due[\s:]+(?:₹\s*)?([\d,]+\.?\d*)", r"Total\s*Amount[\s:]+(?:₹\s*)?([\d,]+\.?\d*)", r"Grand\s*Total[\s:]+(?:₹\s*)?([\d,]+\.?\d*)"])
        tax = self._extract_amount(raw_text, [r"Tax\s*[:\s]+(?:₹\s*)?([\d,]+\.?\d*)", r"GST\s*[:\s]+(?:₹\s*)?([\d,]+\.?\d*)", r"IGST\s*[:\s]+(?:₹\s*)?([\d,]+\.?\d*)"])
        total = self._extract_amount(raw_text, [r"Total\s*[:\s]+(?:₹\s*)?([\d,]+\.?\d*)", r"Grand\s*Total[\s:]+(?:₹\s*)?([\d,]+\.?\d*)", r"Amount\s*Due[\s:]+(?:₹\s*)?([\d,]+\.?\d*)"])

        if not table_df.empty:
            normalized_table = self._normalize_xlsx(table_df)
            for column in ["invoice_number", "date", "vendor", "amount", "tax", "total"]:
                if column not in normalized_table.columns:
                    normalized_table[column] = None
            normalized_table["source_file"] = path.name
            if invoice_number and normalized_table["invoice_number"].isna().all():
                normalized_table["invoice_number"] = invoice_number
            if invoice_date and normalized_table["date"].isna().all():
                normalized_table["date"] = invoice_date
            if vendor and normalized_table["vendor"].isna().all():
                normalized_table["vendor"] = vendor
            if amount is not None and normalized_table["amount"].isna().all():
                normalized_table["amount"] = amount
            if tax is not None and normalized_table["tax"].isna().all():
                normalized_table["tax"] = tax
            if total is not None and normalized_table["total"].isna().all():
                normalized_table["total"] = total
            return normalized_table[["invoice_number", "date", "vendor", "amount", "tax", "total", "source_file"]]

        return pd.DataFrame([
            {
                "invoice_number": invoice_number,
                "date": invoice_date,
                "vendor": vendor,
                "amount": amount,
                "tax": tax,
                "total": total,
                "source_file": path.name,
                "raw_text": raw_text,
            }
        ])

    def extract_pdf_text(self, file_path: str) -> str:
        """Return combined text extracted from the invoice PDF."""
        raw_text, _ = self._extract_pdf_content(Path(file_path))
        return raw_text

    def _extract_pdf_content(self, path: Path) -> tuple[str, pd.DataFrame]:
        raw_text_parts: list[str] = []
        tables: list[pd.DataFrame] = []

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    raw_text_parts.append(page_text)

                page_tables = page.extract_tables() or []
                for table in page_tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(cell).strip() if cell is not None else "" for cell in table[0]]
                    rows = table[1:]
                    tables.append(pd.DataFrame(rows, columns=header))

        table_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        return "\n".join(raw_text_parts).strip(), table_df

    def _score_invoice_columns(self, df: pd.DataFrame) -> int:
        columns = {str(column).strip().lower() for column in df.columns}
        score = 0
        for aliases in self.XLSX_COLUMN_ALIASES.values():
            if any(alias in columns or any(alias in column for column in columns) for alias in aliases):
                score += 1
        return score

    def _normalize_xlsx(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = pd.DataFrame(index=df.index)

        for standard_name, aliases in self.XLSX_COLUMN_ALIASES.items():
            source_column = self._find_column(df, aliases)
            if source_column is None:
                normalized[standard_name] = None
            else:
                normalized[standard_name] = df[source_column]

        if "date" in normalized.columns:
            normalized["date"] = normalized["date"].apply(self._parse_date)
        for column in ["amount", "tax", "total"]:
            if column in normalized.columns:
                normalized[column] = normalized[column].apply(self._parse_amount)

        if "total" in normalized.columns:
            missing_total = normalized["total"].isna()
            if missing_total.any() and {"amount", "tax"}.issubset(normalized.columns):
                normalized.loc[missing_total, "total"] = (
                    normalized.loc[missing_total, "amount"].fillna(0) + normalized.loc[missing_total, "tax"].fillna(0)
                )

        return normalized[["invoice_number", "date", "vendor", "amount", "tax", "total"]]

    def _find_column(self, df: pd.DataFrame, aliases: list[str]) -> str | None:
        columns = [str(column) for column in df.columns]
        lower_lookup = {column.lower().strip(): column for column in columns}

        for alias in aliases:
            alias_lower = alias.lower().strip()
            if alias_lower in lower_lookup:
                return lower_lookup[alias_lower]

        for alias in aliases:
            alias_lower = alias.lower().strip()
            for column in columns:
                column_lower = column.lower().strip()
                if alias_lower in column_lower or column_lower in alias_lower:
                    return column
        return None

    def _extract_first(self, text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                return value or None
        return None

    def _extract_amount(self, text: str, patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = clean_amount(match.group(1).strip())
                if value != 0.0:
                    return value
        return None

    def _parse_date(self, value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        cleaned = clean_date(value)
        return cleaned or None

    def _parse_amount(self, value: Any) -> float | None:
        cleaned = clean_amount(value)
        return cleaned if cleaned != 0.0 else None
    
    def parse_pdf_with_llm(self, file_path: str) -> dict:
        raw_text, _ = self._extract_pdf_content(Path(file_path))

        if len(raw_text.strip()) < 50:
            return {"error": "Image-based invoice — text extraction not supported yet"}

        words = raw_text.split()
        if len(words) > 1500:
            raw_text = " ".join(words[:1500])

        from extraction.models import InvoiceEntities, LineItem
        import requests, json

        prompt = INVOICE_PROMPT.format(raw_text=raw_text)

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=400,
            )
            response.raise_for_status()
            raw_response = response.json().get("response", "").strip()

            cleaned = raw_response
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            parsed["line_items"] = [LineItem(**item) for item in parsed.get("line_items", [])]
            parsed["raw_text"] = raw_text
            invoice = InvoiceEntities(**parsed)
            return {"invoice_data": invoice, "doc_type": "invoice"}

        except json.JSONDecodeError as e:
            return {"error": f"Ollama returned invalid JSON: {e}"}
        except Exception as e:
            return {"error": f"Invoice parsing failed: {e}"}