"""Expense sheet parser for CSV and Excel exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dateparser
import pandas as pd

from ingestion.parsers.normalizers import clean_amount, clean_date


class ExpenseParser:
    """Parse expense sheets and normalize records to a standard schema."""

    STANDARD_COLUMNS = [
        "date",
        "description",
        "debit",
        "credit",
        "balance",
        "transaction_id",
        "bank_name",
        "category",
        "submitted_by",
        "approved_by",
        "amount",
        "receipt_attached",
        "expense_id",
    ]

    COLUMN_ALIASES: dict[str, list[str]] = {
        "date": [
            "Date",
            "Expense Date",
            "Transaction Date",
            "Created Time",
            "Created Date",
            "Spent On",
            "Expense Start Date",
        ],
        "description": [
            "Description",
            "Purpose",
            "Merchant",
            "Merchant Name",
            "Expense Name",
            "Item Name",
            "Notes",
            "Comments",
            "Expense Description",
        ],
        "category": [
            "Category",
            "Expense Category",
            "Category Name",
            "Type",
            "Expense Type",
        ],
        "amount": [
            "Amount",
            "Total",
            "Expense Amount",
            "Claim Amount",
            "Approved Amount",
            "Net Amount",
            "Amount (INR)",
            "Total Amount",
        ],
        "submitted_by": [
            "Submitted By",
            "Employee Name",
            "Employee",
            "User",
            "Owner",
            "Created By",
            "Claimed By",
            "Submitter",
        ],
        "approved_by": [
            "Approved By",
            "Approver",
            "Approved by",
            "Reporting Manager",
            "Approved Manager",
            "Approver Name",
            "Manager",
        ],
        "receipt_attached": [
            "Receipt Attached",
            "Receipt Attached Y/N",
            "Receipt",
            "Has Receipt",
            "Receipt Available",
            "Is Receipt Attached",
            "Bill Attached",
            "Attachment",
        ],
        "expense_id": [
            "Expense ID",
            "Expense Id",
            "ID",
            "Reference ID",
            "Report Entry ID",
            "Expense Number",
            "Transaction ID",
            "Claim ID",
        ],
    }

    SOURCE_PATTERNS: dict[str, set[str]] = {
        "zoho_expense": {"Expense ID", "Category", "Amount"},
        "happay": {"Expense Id", "Category", "Amount"},
        "walnut": {"Date", "Category", "Amount"},
        "common_expense_sheet": {
            "Date",
            "Description",
            "Category",
            "Amount",
            "Submitted By",
            "Approved By",
            "Receipt Attached Y/N",
        },
    }

    def parse(self, file_path: str) -> pd.DataFrame:
        """Parse CSV/Excel expense data and return normalized dataframe."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            raw_df = self._read_csv(path)
        elif suffix in {".xlsx", ".xls", ".xlsm"}:
            raw_df = self._read_expense_excel(path)
        else:
            raise ValueError(f"Unsupported expense file type: {path.suffix}")

        normalized = self._normalize_dataframe(raw_df)
        normalized["flag_missing_receipt_high_amount"] = (
            (~normalized["receipt_attached"].fillna(False)) & (normalized["amount"].fillna(0) > 5000)
        )
        normalized["flag_missing_approval_very_high_amount"] = (
            normalized["approved_by"].fillna("").astype(str).str.strip().eq("")
            & (normalized["amount"].fillna(0) > 10000)
        )

        return normalized

    def looks_like_expense_sheet(self, df: pd.DataFrame) -> bool:
        """Return True if dataframe columns resemble a known expense schema."""
        if df.empty and len(df.columns) == 0:
            return False

        headers = {str(col).strip() for col in df.columns}
        lowered_headers = {header.lower() for header in headers}

        for required_headers in self.SOURCE_PATTERNS.values():
            required_lower = {item.lower() for item in required_headers}
            if required_lower.issubset(lowered_headers):
                return True

        matched_standard_columns = 0
        for standard_col, aliases in self.COLUMN_ALIASES.items():
            if self._find_column(df, aliases) is not None:
                matched_standard_columns += 1

        return matched_standard_columns >= 5

    def categorize_expense(self, description: str) -> str:
        """Categorize expense description using keyword rules."""
        text = (description or "").lower()

        category_rules: dict[str, set[str]] = {
            "travel": {"flight", "hotel", "cab", "uber", "ola", "taxi"},
            "food": {"restaurant", "swiggy", "zomato", "lunch", "dinner"},
            "office": {"stationery", "printer", "supplies"},
            "software": {"subscription", "saas", "license", "aws", "google"},
        }

        for category, keywords in category_rules.items():
            if any(keyword in text for keyword in keywords):
                return category

        return "uncategorized"

    def _read_csv(self, path: Path) -> pd.DataFrame:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_error: Exception | None = None

        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as exc:
                last_error = exc

        raise ValueError(f"Unable to read CSV file {path}: {last_error}")

    def _read_expense_excel(self, path: Path) -> pd.DataFrame:
        workbook = pd.ExcelFile(path)

        best_sheet_name: str | None = None
        best_sheet_score = -1
        best_sheet_df = pd.DataFrame()

        for sheet_name in workbook.sheet_names:
            candidate_df = workbook.parse(sheet_name)
            score = self._expense_sheet_score(candidate_df)
            if score > best_sheet_score:
                best_sheet_score = score
                best_sheet_name = sheet_name
                best_sheet_df = candidate_df

        if best_sheet_name is None:
            raise ValueError(f"No readable sheets found in Excel file: {path}")

        return best_sheet_df

    def _expense_sheet_score(self, df: pd.DataFrame) -> int:
        if df.empty and len(df.columns) == 0:
            return 0

        score = 0
        for aliases in self.COLUMN_ALIASES.values():
            if self._find_column(df, aliases) is not None:
                score += 1

        return score

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = pd.DataFrame(index=df.index)

        for standard_col in self.STANDARD_COLUMNS:
            source_col = self._find_column(df, self.COLUMN_ALIASES.get(standard_col, []))
            if source_col is None:
                normalized[standard_col] = None
            else:
                normalized[standard_col] = df[source_col]

        normalized["date"] = normalized["date"].apply(self._parse_date)
        normalized["description"] = normalized["description"].fillna("").astype(str).str.strip()
        normalized["category"] = normalized["category"].fillna("").astype(str).str.strip()
        normalized["amount"] = normalized["amount"].apply(self._clean_amount).fillna(0.0)

        normalized["submitted_by"] = normalized["submitted_by"].fillna("").astype(str).str.strip()
        normalized["approved_by"] = normalized["approved_by"].fillna("").astype(str).str.strip()

        normalized["receipt_attached"] = normalized["receipt_attached"].apply(self._to_bool)

        normalized["category"] = normalized.apply(
            lambda row: row["category"] if row["category"] else self.categorize_expense(row["description"]),
            axis=1,
        )

        normalized["expense_id"] = normalized["expense_id"].fillna("").astype(str).str.strip()
        missing_id_mask = normalized["expense_id"].eq("")
        normalized.loc[missing_id_mask, "expense_id"] = normalized.loc[missing_id_mask].index.map(
            lambda idx: f"EXP-{idx + 1:06d}"
        )

        normalized["debit"] = normalized["amount"].astype(float)
        normalized["credit"] = 0.0
        normalized["balance"] = 0.0
        normalized["transaction_id"] = normalized["expense_id"]
        normalized["bank_name"] = "Expense Sheet"

        return normalized[self.STANDARD_COLUMNS]

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

    def _clean_amount(self, value: Any) -> float | None:
        # Use robust clean_amount normalizer
        return clean_amount(value)

    def _to_bool(self, value: Any) -> bool | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None

        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()
        if text in {"yes", "y", "true", "1", "attached"}:
            return True
        if text in {"no", "n", "false", "0", "not attached"}:
            return False

        return None
