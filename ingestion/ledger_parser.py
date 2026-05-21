"""Ledger parser for CSV and Excel formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dateparser
import pandas as pd

from ingestion.parsers.normalizers import clean_amount, clean_date


class LedgerParser:
    """Parse ledger exports from common accounting tools."""

    STANDARD_COLUMNS = [
        "date",
        "voucher_number",
        "party_name",
        "category",
        "debit",
        "credit",
        "narration",
        "reference",
    ]

    MANDATORY_FIELDS = ["date", "narration"]

    LEDGER_PATTERNS: dict[str, set[str]] = {
        "tally": {"Particulars", "Vch Type", "Vch No.", "Debit", "Credit"},
        "zoho": {"Date", "Account", "Debit", "Credit", "Description"},
        "quickbooks": {"Date", "Num", "Name", "Memo", "Debit", "Credit"},
    }

    COLUMN_MAPS: dict[str, dict[str, list[str]]] = {
        "tally": {
            "date": ["Date"],
            "voucher_number": ["Vch No."],
            "party_name": ["Particulars"],
            "category": ["Vch Type"],
            "debit": ["Debit", "Dr"],
            "credit": ["Credit", "Cr"],
            "narration": ["Particulars", "Narration"],
            "reference": ["Ref", "Reference", "Vch No."],
        },
        "zoho": {
            "date": ["Date"],
            "voucher_number": ["Voucher Number", "Voucher #", "Num"],
            "party_name": ["Contact Name", "Account", "Party", "Name"],
            "category": ["Account", "Category", "Type"],
            "debit": ["Debit"],
            "credit": ["Credit"],
            "narration": ["Description", "Notes", "Narration"],
            "reference": ["Reference", "Ref No", "Doc No"],
        },
        "quickbooks": {
            "date": ["Date"],
            "voucher_number": ["Num", "No.", "Doc Num"],
            "party_name": ["Name", "Customer/Vendor"],
            "category": ["Account", "Type", "Detail Type"],
            "debit": ["Debit"],
            "credit": ["Credit"],
            "narration": ["Memo", "Description"],
            "reference": ["Ref No", "Transaction No", "Num"],
        },
        "generic": {
            "date": ["date", "transaction date", "entry date"],
            "voucher_number": ["voucher", "vch", "num", "bill no", "invoice no"],
            "party_name": ["party", "name", "account", "particulars"],
            "category": ["category", "type", "account", "vch type"],
            "debit": ["debit", "dr"],
            "credit": ["credit", "cr"],
            "narration": ["narration", "description", "memo", "remarks", "particulars"],
            "reference": ["ref", "reference", "transaction id", "cheque"],
        },
    }

    def detect_ledger_type(self, df: pd.DataFrame) -> str:
        """Detect ledger source based on columns."""
        headers = {str(col).strip() for col in df.columns}
        for ledger_type, required_headers in self.LEDGER_PATTERNS.items():
            if required_headers.issubset(headers):
                return ledger_type
        return "generic"

    def parse(self, file_path: str) -> pd.DataFrame:
        """Parse CSV/Excel ledger and return normalized dataframe."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            raw_df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            raw_df = self._read_ledger_excel(path)
        else:
            raise ValueError(f"Unsupported ledger file type: {path.suffix}")

        ledger_type = self.detect_ledger_type(raw_df)
        normalized = self._normalize_dataframe(raw_df, ledger_type)
        normalized["data_quality_issues"] = normalized.apply(self._collect_quality_issues, axis=1)
        return normalized

    def _read_ledger_excel(self, path: Path) -> pd.DataFrame:
        """Read the "Ledger" sheet if present; otherwise choose largest sheet."""
        workbook = pd.ExcelFile(path)
        sheet_names = workbook.sheet_names

        ledger_sheet = None
        for name in sheet_names:
            if name.strip().lower() == "ledger":
                ledger_sheet = name
                break

        if ledger_sheet is not None:
            return workbook.parse(ledger_sheet)

        largest_sheet = None
        largest_size = -1
        for name in sheet_names:
            candidate_df = workbook.parse(name)
            size = candidate_df.shape[0] * max(candidate_df.shape[1], 1)
            if size > largest_size:
                largest_size = size
                largest_sheet = name

        if largest_sheet is None:
            raise ValueError(f"No readable sheets found in Excel file: {path}")

        return workbook.parse(largest_sheet)

    def _normalize_dataframe(self, df: pd.DataFrame, ledger_type: str) -> pd.DataFrame:
        mapping = self.COLUMN_MAPS.get(ledger_type, self.COLUMN_MAPS["generic"])
        normalized = pd.DataFrame(index=df.index)

        for standard_column in self.STANDARD_COLUMNS:
            source_col = self._find_column(df, mapping.get(standard_column, []))
            if source_col is None:
                normalized[standard_column] = None
            else:
                normalized[standard_column] = df[source_col]

        normalized["date"] = normalized["date"].apply(self._parse_date)
        normalized["debit"] = normalized["debit"].apply(self._clean_amount)
        normalized["credit"] = normalized["credit"].apply(self._clean_amount)

        for text_col in ["voucher_number", "party_name", "category", "narration", "reference"]:
            normalized[text_col] = normalized[text_col].fillna("").astype(str).str.strip()
            normalized[text_col] = normalized[text_col].replace("nan", "")

        return normalized

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        columns = [str(col) for col in df.columns]
        lowered = {col.lower().strip(): col for col in columns}

        for candidate in candidates:
            key = candidate.lower().strip()
            if key in lowered:
                return lowered[key]

        for candidate in candidates:
            candidate_key = candidate.lower().strip()
            for col in columns:
                col_key = col.lower().strip()
                if candidate_key in col_key or col_key in candidate_key:
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

    def _clean_amount(self, value: Any) -> float | None:
        # Use robust clean_amount normalizer
        cleaned = clean_amount(value)
        return cleaned if cleaned != 0.0 else None

    def _collect_quality_issues(self, row: pd.Series) -> str:
        issues: list[str] = []

        for field in self.MANDATORY_FIELDS:
            value = row[field]
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                issues.append(f"missing_{field}")

        if row["debit"] is None and row["credit"] is None:
            issues.append("missing_amount")

        if row["debit"] is not None and row["credit"] is not None:
            issues.append("both_debit_and_credit_present")

        return "|".join(issues)
