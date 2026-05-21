"""Bank statement parsing utilities for CSV and PDF files."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import logging

import dateparser
import pandas as pd
import pdfplumber

from ingestion.parsers.normalizers import clean_amount, clean_date
from utils.errors import AuditError, ERROR_CODES


class BankStatementParser:
    """Parser that normalizes bank statements from multiple banks."""

    STANDARD_COLUMNS = [
        "date",
        "description",
        "debit",
        "credit",
        "balance",
        "transaction_id",
        "bank_name",
    ]

    BANK_HEADER_PATTERNS: dict[str, set[str]] = {
        "hdfc": {"Narration", "Chq./Ref.No."},
        "icici": {"Transaction Remarks", "Cheque Number"},
        "sbi": {"Txn Date", "Description", "Ref No./Cheque No."},
        "axis": {"Tran Date", "PARTICULARS", "CHQNO"},
        "pnb": {"DATE", "PARTICULARS", "CHEQUE NO"},
        "kotak": {"Transaction Date", "Description"},
        "yesbank": {"Date", "Transaction Details"},
    }

    BANK_COLUMN_MAPS: dict[str, dict[str, list[str]]] = {
        "hdfc": {
            "date": ["Date", "Value Dt", "Txn Date"],
            "description": ["Narration"],
            "debit": ["Withdrawal Amt.", "Debit", "Dr Amount"],
            "credit": ["Deposit Amt.", "Credit", "Cr Amount"],
            "balance": ["Closing Balance", "Balance"],
            "transaction_id": ["Chq./Ref.No.", "Reference No", "Ref No"],
        },
        "icici": {
            "date": ["Transaction Date", "Date", "Value Date"],
            "description": ["Transaction Remarks", "Remarks", "Description"],
            "debit": ["Withdrawal Amount (INR )", "Debit Amount", "Debit"],
            "credit": ["Deposit Amount (INR )", "Credit Amount", "Credit"],
            "balance": ["Balance (INR )", "Balance"],
            "transaction_id": ["Cheque Number", "Reference Number", "Ref No"],
        },
        "sbi": {
            "date": ["Txn Date", "Date"],
            "description": ["Description"],
            "debit": ["Debit", "Withdrawal"],
            "credit": ["Credit", "Deposit"],
            "balance": ["Balance"],
            "transaction_id": ["Ref No./Cheque No.", "Ref No", "Cheque No"],
        },
        "axis": {
            "date": ["Tran Date", "Date", "Value Date"],
            "description": ["Particulars", "Description"],
            "debit": ["Debit", "Withdrawal"],
            "credit": ["Credit", "Deposit"],
            "balance": ["Balance", "Running Bal"],
            "transaction_id": ["Chq/Ref Number", "Ref No", "Cheque Number"],
        },
        "pnb": {
            "date": ["DATE", "Date"],
            "description": ["PARTICULARS", "Particulars", "Description"],
            "debit": ["DEBIT", "Debit"],
            "credit": ["CREDIT", "Credit"],
            "balance": ["BALANCE", "Balance"],
            "transaction_id": ["CHEQUE NO", "Cheque No", "Ref No"],
        },
        "kotak": {
            "date": ["Transaction Date", "Date"],
            "description": ["Description", "Narration", "Remarks"],
            "debit": ["Debit Amount", "DEBIT", "Debit"],
            "credit": ["Credit Amount", "CREDIT", "Credit"],
            "balance": ["Balance"],
            "transaction_id": ["Ref No", "Reference", "Cheque No"],
        },
        "yesbank": {
            "date": ["Date", "Transaction Date"],
            "description": ["Transaction Details", "Description", "Remarks"],
            "debit": ["Withdrawal Amount", "Withdrawal Amount (INR )", "Debit"],
            "credit": ["Deposit Amount", "Deposit Amount (INR )", "Credit"],
            "balance": ["Balance"],
            "transaction_id": ["Ref No", "Reference", "Cheque No"],
        },
        "generic": {
            "date": ["date", "txn date", "transaction date", "value date"],
            "description": ["description", "narration", "remarks", "particulars"],
            "debit": ["debit", "withdrawal", "dr"],
            "credit": ["credit", "deposit", "cr"],
            "balance": ["balance", "closing balance", "running balance"],
            "transaction_id": ["reference", "ref", "cheque", "utr", "chq"],
        },
    }

    UPI_PATTERNS = {
        "phonepe": re.compile(
            r"UPI/(?P<ref>[A-Za-z0-9]+)/.*?(PHONEPE|PPBL).*?(?P<recipient>[\w.\-]+@[\w]+)?",
            re.IGNORECASE,
        ),
        "gpay": re.compile(
            r"UPI/(?P<ref>[A-Za-z0-9]+)/.*?(GPAY|GOOGLE ?PAY).*?(?P<recipient>[\w.\-]+@[\w]+)?",
            re.IGNORECASE,
        ),
        "paytm": re.compile(
            r"UPI/(?P<ref>[A-Za-z0-9]+)/.*?(PAYTM).*?(?P<recipient>[\w.\-]+@[\w]+)?",
            re.IGNORECASE,
        ),
        "bhim": re.compile(
            r"UPI/(?P<ref>[A-Za-z0-9]+)/.*?(BHIM).*?(?P<recipient>[\w.\-]+@[\w]+)?",
            re.IGNORECASE,
        ),
    }

    def clean_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw statement rows before bank-specific normalization."""
        if df.empty:
            return df

        df = df.copy()

        # Step 1: Strip whitespace from all column names
        df.columns = df.columns.map(lambda c: str(c).strip())

        # Step 2: Drop completely empty rows
        df = df.dropna(how="all")

        # Step 3: Skip summary/header rows by description keywords
        lower_to_col = {str(col).lower(): col for col in df.columns}
        desc_col = None
        if "description" in lower_to_col:
            desc_col = lower_to_col["description"]
        elif "narration" in lower_to_col:
            desc_col = lower_to_col["narration"]

        if desc_col is not None:
            skip_keywords = [
                "opening balance",
                "closing balance",
                "total",
                "brought forward",
                "carried forward",
                "as on",
                "statement of",
                "account no",
                "branch",
            ]
            mask = df[desc_col].astype(str).str.lower().apply(
                lambda x: not any(keyword in x for keyword in skip_keywords)
            )
            df = df[mask]

        # Step 4: Skip rows where date cannot be parsed
        date_col = None
        for candidate in ["date", "txn date", "transaction date", "value date"]:
            if candidate in lower_to_col:
                date_col = lower_to_col[candidate]
                break

        if date_col is None:
            for col in df.columns:
                col_text = str(col).lower()
                if any(token in col_text for token in ["date", "txn", "transaction", "value"]):
                    date_col = col
                    break

        if date_col is not None:
            date_mask = df[date_col].apply(lambda val: clean_date(val) is not None)
            df = df[date_mask]

        # Step 5: Reset index
        df = df.reset_index(drop=True)
        return df

    def clean_raw_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible wrapper for legacy calls."""
        return self.clean_raw_dataframe(df)

    def detect_bank(self, df: pd.DataFrame, raw_text: str = "") -> str:
        """Detect bank from OCR text or known column header signatures."""
        text = str(raw_text or "").lower()
        if text:
            text_patterns = [
                ("sbi", ["state bank of india", "sbi"]),
                ("hdfc", ["hdfc bank", "hdfc"]),
                ("icici", ["icici bank", "icici"]),
                ("axis", ["axis bank", "axis"]),
                ("kotak", ["kotak mahindra", "kotak"]),
                ("pnb", ["punjab national", "pnb"]),
                ("yesbank", ["yes bank"]),
            ]
            for bank_name, patterns in text_patterns:
                if any(pattern in text for pattern in patterns):
                    return bank_name

        # Normalize column names by stripping whitespace
        try:
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass

        headers = {str(col).strip() for col in df.columns}
        headers_lower = {h.lower() for h in headers}

        # Special-case: ICICI variations where indicator columns may have trailing spaces
        icici_indicators = [
            "Transaction Remarks",
            "Withdrawal Amount (INR)",
            "Withdrawal Amount (INR )",
            "Deposit Amount (INR)",
            "Deposit Amount (INR )",
        ]
        icici_lower = {s.lower() for s in icici_indicators}
        icici_count = sum(1 for ind in icici_lower if ind in headers_lower)
        if icici_count >= 2:
            return "icici"

        for bank_name, required_headers in self.BANK_HEADER_PATTERNS.items():
            required_lower = {h.lower() for h in required_headers}
            if required_lower.issubset(headers_lower):
                return bank_name

        # attempt generic detection: if any date/debit/credit-like columns present
        keywords = {"date", "txn", "transaction", "debit", "credit", "withdrawal", "deposit"}
        if any(any(k in col for k in keywords) for col in headers_lower):
            logging.getLogger(__name__).warning("Unknown bank format, attempting generic parse")
            return "generic"

        logging.getLogger(__name__).warning("Unknown bank format, no obvious debit/credit/date columns found, attempting generic parse")
        return "generic"

    def parse_csv(self, file_path: str) -> pd.DataFrame:
        """Parse CSV statement and normalize fields."""
        csv_path = Path(file_path)
        dataframe = pd.read_csv(csv_path)
        
        # Run generic row/column cleanup before bank-specific mapping.
        dataframe = self.clean_raw_dataframe(dataframe)
        
        bank_name = self.detect_bank(dataframe)
        normalized = self._normalize_dataframe(dataframe, bank_name)
        return normalized

    def parse_pdf_statement(self, file_path: str) -> pd.DataFrame:
        """Extract tables from PDF statement and normalize fields."""
        pdf_path = Path(file_path)
        rows: list[list[Any]] = []
        header: list[str] | None = None

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    candidate_header = [str(cell).strip() if cell is not None else "" for cell in table[0]]
                    if header is None:
                        header = candidate_header
                    if candidate_header == header:
                        table_rows = table[1:]
                    else:
                        table_rows = table
                    for row in table_rows:
                        if row and any(cell not in (None, "") for cell in row):
                            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        if header is None:
            raise ValueError("Could not detect header row in PDF tables.")

        extracted = pd.DataFrame(rows, columns=header)
        
        # Run generic row/column cleanup before bank-specific mapping.
        extracted = self.clean_raw_dataframe(extracted)
        
        bank_name = self.detect_bank(extracted)
        return self._normalize_dataframe(extracted, bank_name)

    def parse_upi_description(self, description: str) -> dict[str, str | None]:
        """Parse common UPI description formats and extract metadata."""
        raw_text = (description or "").strip()
        result = {
            "upi_app": None,
            "recipient": None,
            "reference_number": None,
            "raw": raw_text,
        }

        if not raw_text:
            return result

        for app_name, pattern in self.UPI_PATTERNS.items():
            match = pattern.search(raw_text)
            if match:
                result["upi_app"] = app_name
                result["reference_number"] = match.groupdict().get("ref")
                result["recipient"] = match.groupdict().get("recipient")
                return result

        if "UPI/" in raw_text.upper():
            parts = [part for part in raw_text.split("/") if part]
            if len(parts) > 1:
                result["reference_number"] = parts[1]
            recipient_match = re.search(r"([\w.\-]+@[\w]+)", raw_text)
            if recipient_match:
                result["recipient"] = recipient_match.group(1)

        return result

    def parse(self, file_path: str) -> pd.DataFrame:
        """Main entry point for statement parsing."""
        statement_path = Path(file_path)
        suffix = statement_path.suffix.lower()

        if suffix == ".csv":
            return self.parse_csv(str(statement_path))
        if suffix == ".pdf":
            return self.parse_pdf_statement(str(statement_path))

        raise ValueError(f"Unsupported file type: {statement_path.suffix}")

    def _normalize_dataframe(self, df: pd.DataFrame, bank_name: str) -> pd.DataFrame:
        mapping = self.BANK_COLUMN_MAPS.get(bank_name, self.BANK_COLUMN_MAPS["generic"])
        normalized = pd.DataFrame(index=df.index)

        for std_col in ["date", "description", "debit", "credit", "balance", "transaction_id"]:
            source_col = self._find_column(df, mapping.get(std_col, []))
            if source_col is not None:
                normalized[std_col] = df[source_col]
            else:
                normalized[std_col] = None

        normalized["bank_name"] = bank_name
        normalized["date"] = normalized["date"].apply(self._parse_date)

        # Clean debit/credit using robust normalizer
        normalized["debit"] = normalized["debit"].apply(lambda x: clean_amount(x) if x is not None else None)
        normalized["credit"] = normalized["credit"].apply(lambda x: clean_amount(x) if x is not None else None)
        normalized["balance"] = normalized["balance"].apply(lambda x: clean_amount(x) if x is not None else None)
        normalized["description"] = normalized["description"].fillna("").astype(str).str.strip()
        normalized["transaction_id"] = (
            normalized["transaction_id"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )

        upi_info = normalized["description"].apply(self.parse_upi_description)
        normalized["transaction_id"] = normalized.apply(
            lambda row: row["transaction_id"]
            if row["transaction_id"]
            else (upi_info[row.name].get("reference_number") or ""),
            axis=1,
        )

        # Ensure numeric amounts and treat missing debit/credit as 0.0
        normalized["debit"] = pd.to_numeric(normalized["debit"], errors="coerce").fillna(0.0)
        normalized["credit"] = pd.to_numeric(normalized["credit"], errors="coerce").fillna(0.0)
        # balance may legitimately be missing; coerce to numeric if present
        normalized["balance"] = pd.to_numeric(normalized["balance"], errors="coerce")

        normalized = normalized[self.STANDARD_COLUMNS]
        return normalized

    def _find_column(self, df: pd.DataFrame, candidate_names: list[str]) -> str | None:
        if not candidate_names:
            return None

        columns = [str(col) for col in df.columns]
        lower_lookup = {col.lower().strip(): col for col in columns}

        for candidate in candidate_names:
            candidate_lower = candidate.lower().strip()
            if candidate_lower in lower_lookup:
                return lower_lookup[candidate_lower]

        for candidate in candidate_names:
            candidate_lower = candidate.lower().strip()
            for col in columns:
                col_lower = col.lower().strip()
                if candidate_lower in col_lower or col_lower in candidate_lower:
                    return col

        return None

    def _parse_date(self, value: Any) -> pd.Timestamp | pd.NaT:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return pd.NaT
        # Use robust clean_date normalizer
        cleaned = clean_date(value)
        if cleaned:
            return pd.to_datetime(cleaned, errors="coerce")
        return pd.NaT

    def _clean_amount(self, value: Any, expected: str | None = None) -> float | None:
        # Use robust clean_amount normalizer
        cleaned = clean_amount(value)
        
        # If expected direction and value is 0, return None (missing)
        if cleaned == 0.0 and value is None:
            return None
        
        return cleaned if cleaned != 0.0 else None

    def validate_parsed_df(self, df: pd.DataFrame, file_name: str | None = None) -> dict[str, Any]:
        """Validate parsed dataframe for data quality issues.
        
        Args:
            df: Parsed dataframe to validate
            file_name: Optional file name for error reporting
            
        Returns:
            Dict with 'valid': bool, 'warnings': list[str]
            
        Raises:
            AuditError: If critical data quality issues found
        """
        warnings: list[str] = []
        
        # Check minimum rows
        if len(df) < 2:
            raise AuditError(
                message=ERROR_CODES["EMPTY_FILE"]["message"],
                error_code="EMPTY_FILE",
                file_name=file_name,
                suggestion=ERROR_CODES["EMPTY_FILE"]["suggestion"],
            )
        
        # Check date column for excessive NaT (>20%)
        if "date" in df.columns:
            nat_count = df["date"].isna().sum()
            nat_percentage = (nat_count / len(df)) * 100 if len(df) > 0 else 0
            if nat_percentage > 20:
                raise AuditError(
                    message=ERROR_CODES["DATE_PARSE_FAILED"]["message"],
                    error_code="DATE_PARSE_FAILED",
                    file_name=file_name,
                    suggestion=ERROR_CODES["DATE_PARSE_FAILED"]["suggestion"],
                )
            if nat_percentage > 0:
                warnings.append(f"Date parsing failed for {nat_percentage:.1f}% of rows")
        
        # Check amount columns for non-numeric (>20%)
        for amount_col in ["debit", "credit", "balance"]:
            if amount_col in df.columns:
                non_numeric = pd.to_numeric(df[amount_col], errors="coerce").isna().sum()
                non_numeric_percentage = (non_numeric / len(df)) * 100 if len(df) > 0 else 0
                if non_numeric_percentage > 20:
                    raise AuditError(
                        message=ERROR_CODES["AMOUNT_PARSE_FAILED"]["message"],
                        error_code="AMOUNT_PARSE_FAILED",
                        file_name=file_name,
                        suggestion=ERROR_CODES["AMOUNT_PARSE_FAILED"]["suggestion"],
                    )
                if non_numeric_percentage > 0:
                    warnings.append(f"Amount parsing failed for {non_numeric_percentage:.1f}% of {amount_col} values")
        
        # Check bank name
        if "bank_name" in df.columns:
            bank_name = df["bank_name"].iloc[0] if len(df) > 0 else None
            if bank_name == "unknown":
                warnings.append(
                    f"{ERROR_CODES['UNKNOWN_BANK_FORMAT']['message']} - {ERROR_CODES['UNKNOWN_BANK_FORMAT']['suggestion']}"
                )
        
        return {"valid": True, "warnings": warnings}

    def _extract_debit_credit_from_text(self, description: Any) -> tuple[float | None, float | None]:
        text = str(description or "")
        amount_match = re.search(r"([0-9][0-9,]*\.?[0-9]*)\s*(Dr|Cr)\b", text, flags=re.IGNORECASE)
        if not amount_match:
            return (None, None)

        amount_text = amount_match.group(1)
        suffix = amount_match.group(2).lower()
        amount = self._clean_amount(amount_text)
        if amount is None:
            return (None, None)
        if suffix == "dr":
            return (amount, None)
        return (None, amount)


if __name__ == "__main__":
    sample_hdfc_csv = StringIO(
        """Date,Narration,Chq./Ref.No.,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/04/2026,UPI/123456789/Payment/PHONEPE/merchant@ybl,123456789,"1,500.00",,"24,500.00"
02-04-2026,NEFT CREDIT SALARY/ABC LTD/9988776655,9988776655,,"75,000.00","99,500.00"
03 Apr 2026,ATM WDL DR / 8899001122,8899001122,"5,000 Dr",,"94,500.00"
"""
    )

    parser = BankStatementParser()

    # parse_csv expects a path; create a temporary file for the inline test.
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
        tmp_file.write(sample_hdfc_csv.getvalue())

    parsed_df = parser.parse_csv(str(temp_path))
    print(parsed_df)

    upi_info = parser.parse_upi_description("UPI/123456789/Payment/PHONEPE/merchant@ybl")
    print(upi_info)

    temp_path.unlink(missing_ok=True)
