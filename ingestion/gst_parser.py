"""GST Excel parser and reconciliation utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import dateparser
import pandas as pd

from .parsers.normalizers import clean_amount, clean_date


class GSTParser:
    """Parse GST Excel exports and validate GST entries."""

    GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]$")

    GSTR1_ALIASES: dict[str, list[str]] = {
        "buyer_gstin": [
            "GSTIN of Recipient",
            "Recipient GSTIN",
            "GSTIN/UIN of Recipient",
            "Buyer GSTIN",
            "GSTIN",
        ],
        "invoice_number": ["Invoice Number", "Invoice No", "Inv No", "Invoice #", "Doc No"],
        "invoice_date": ["Invoice Date", "Inv Date", "Document Date", "Date"],
        "taxable_value": ["Taxable Value", "Taxable Amount", "Taxable"],
        "igst": ["IGST Amount", "IGST", "Integrated Tax"],
        "cgst": ["CGST Amount", "CGST", "Central Tax"],
        "sgst": ["SGST Amount", "SGST", "State/UT Tax", "UTGST Amount", "UTGST"],
        "total": ["Invoice Value", "Total", "Total Amount", "Gross Total"],
        "gst_rate": ["Rate", "GST Rate", "Tax Rate", "IGST Rate"],
    }

    GSTR2A_ALIASES: dict[str, list[str]] = {
        "supplier_gstin": [
            "GSTIN of Supplier",
            "Supplier GSTIN",
            "GSTIN of Seller",
            "GSTIN/UIN of Supplier",
            "GSTIN",
        ],
        "invoice_number": ["Invoice Number", "Invoice No", "Inv No", "Doc No"],
        "invoice_date": ["Invoice Date", "Inv Date", "Document Date", "Date"],
        "taxable_value": ["Taxable Value", "Taxable Amount", "Taxable"],
        "igst": ["IGST Amount", "IGST", "Integrated Tax"],
        "cgst": ["CGST Amount", "CGST", "Central Tax"],
        "sgst": ["SGST Amount", "SGST", "State/UT Tax", "UTGST Amount", "UTGST"],
        "tax_amount": ["Total Tax", "Tax Amount", "Tax", "Total GST"],
        "gst_rate": ["Rate", "GST Rate", "Tax Rate", "IGST Rate"],
    }

    GST_INVOICE_ALIASES: dict[str, list[str]] = {
        "buyer_gstin": ["Buyer GSTIN", "Bill To GSTIN", "GSTIN of Recipient", "Recipient GSTIN", "GSTIN"],
        "invoice_number": ["Invoice Number", "Invoice No", "Inv No", "Doc No"],
        "invoice_date": ["Invoice Date", "Date", "Inv Date"],
        "taxable_value": ["Taxable Value", "Taxable Amount", "Taxable"],
        "igst": ["IGST Amount", "IGST", "Integrated Tax"],
        "cgst": ["CGST Amount", "CGST", "Central Tax"],
        "sgst": ["SGST Amount", "SGST", "State/UT Tax", "UTGST Amount", "UTGST"],
        "total": ["Invoice Value", "Total", "Grand Total", "Total Amount"],
        "gst_rate": ["Rate", "GST Rate", "Tax Rate"],
    }

    def parse(self, file_path: str) -> dict[str, Any]:
        """Parse GST workbook and return gst_type with normalized data."""
        path = Path(file_path)
        gst_type = self.detect_gst_type(path)

        if gst_type == "gstr1":
            data = self.parse_gstr1(str(path))
        elif gst_type == "gstr2a":
            data = self.parse_gstr2a(str(path))
        elif gst_type == "gst_invoice":
            data = self.parse_gst_invoices(str(path))
        else:
            raise ValueError("Could not identify GST workbook type")

        return {"gst_type": gst_type, "data": data}

    def detect_gst_type(self, file_path: Path) -> str:
        """Detect GST workbook type from sheet names and headers."""
        if file_path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
            return "unknown"

        workbook = pd.ExcelFile(file_path)

        for sheet_name in workbook.sheet_names:
            name = sheet_name.strip().lower()
            if "gstr-1" in name or "gstr1" in name or "outward" in name:
                return "gstr1"
            if "gstr-2a" in name or "gstr2a" in name or "purchase" in name:
                return "gstr2a"
            if "invoice" in name and "gst" in name:
                return "gst_invoice"

        best_kind = "unknown"
        best_score = -1

        for sheet_name in workbook.sheet_names:
            sample_df = workbook.parse(sheet_name, nrows=50)
            gstr1_score = self._score_mapping(sample_df, self.GSTR1_ALIASES)
            gstr2a_score = self._score_mapping(sample_df, self.GSTR2A_ALIASES)
            invoice_score = self._score_mapping(sample_df, self.GST_INVOICE_ALIASES)

            if gstr1_score > best_score:
                best_score = gstr1_score
                best_kind = "gstr1"
            if gstr2a_score > best_score:
                best_score = gstr2a_score
                best_kind = "gstr2a"
            if invoice_score > best_score:
                best_score = invoice_score
                best_kind = "gst_invoice"

        return best_kind if best_score >= 4 else "unknown"

    def looks_like_gst_file(self, file_path: str) -> bool:
        """Return True if workbook appears to be any supported GST format."""
        path = Path(file_path)
        return self.detect_gst_type(path) != "unknown"

    def parse_gstr1(self, file_path: str) -> pd.DataFrame:
        """Extract outward supply entries from GSTR-1 export."""
        path = Path(file_path)
        raw_df = self._read_best_sheet(path, self.GSTR1_ALIASES)
        normalized = self._normalize(raw_df, self.GSTR1_ALIASES)

        normalized["buyer_gstin"] = normalized.get("buyer_gstin", pd.Series(index=normalized.index, dtype=object))
        normalized["gstin"] = normalized["buyer_gstin"]
        columns = [
            "buyer_gstin",
            "gstin",
            "invoice_number",
            "invoice_date",
            "taxable_value",
            "igst",
            "cgst",
            "sgst",
            "total",
            "gst_rate",
        ]
        return normalized[columns]

    def parse_gstr2a(self, file_path: str) -> pd.DataFrame:
        """Extract purchase register entries from GSTR-2A export."""
        path = Path(file_path)
        raw_df = self._read_best_sheet(path, self.GSTR2A_ALIASES)
        normalized = self._normalize(raw_df, self.GSTR2A_ALIASES)

        normalized["supplier_gstin"] = normalized.get("supplier_gstin", pd.Series(index=normalized.index, dtype=object))
        normalized["gstin"] = normalized["supplier_gstin"]
        if "tax_amount" not in normalized.columns:
            normalized["tax_amount"] = normalized[["igst", "cgst", "sgst"]].fillna(0).sum(axis=1)

        columns = [
            "supplier_gstin",
            "gstin",
            "invoice_number",
            "invoice_date",
            "taxable_value",
            "igst",
            "cgst",
            "sgst",
            "tax_amount",
            "gst_rate",
        ]
        return normalized[columns]

    def parse_gst_invoices(self, file_path: str) -> pd.DataFrame:
        """Extract standard GST invoice rows from Excel export."""
        path = Path(file_path)
        raw_df = self._read_best_sheet(path, self.GST_INVOICE_ALIASES)
        normalized = self._normalize(raw_df, self.GST_INVOICE_ALIASES)

        normalized = normalized.rename(columns={"buyer_gstin": "gstin"})
        columns = [
            "gstin",
            "invoice_number",
            "invoice_date",
            "taxable_value",
            "igst",
            "cgst",
            "sgst",
            "total",
            "gst_rate",
        ]
        return normalized[columns]

    def validate_gst_entries(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Validate GST arithmetic and GSTIN format for each row."""
        failures: list[dict[str, Any]] = []

        for index, row in df.iterrows():
            invoice_number = str(row.get("invoice_number", "")).strip()
            gstin_raw = (
                row.get("gstin")
                or row.get("buyer_gstin")
                or row.get("supplier_gstin")
                or ""
            )
            gstin = str(gstin_raw).strip().upper()

            igst = self._as_float(row.get("igst")) or 0.0
            cgst = self._as_float(row.get("cgst")) or 0.0
            sgst = self._as_float(row.get("sgst")) or 0.0
            taxable_value = self._as_float(row.get("taxable_value")) or 0.0

            tax_amount = self._as_float(row.get("tax_amount"))
            if tax_amount is None:
                tax_amount = igst if igst > 0 else (cgst + sgst)

            if gstin and not self.GSTIN_PATTERN.match(gstin):
                failures.append(
                    {
                        "row_index": int(index),
                        "invoice_number": invoice_number,
                        "reason": "invalid_gstin_format",
                        "details": {"gstin": gstin},
                    }
                )

            expected_igst = cgst + sgst
            igst_tolerance = max(1.0, 0.01 * max(abs(expected_igst), 1.0))
            if abs(igst - expected_igst) > igst_tolerance:
                failures.append(
                    {
                        "row_index": int(index),
                        "invoice_number": invoice_number,
                        "reason": "cgst_sgst_not_equal_igst",
                        "details": {"cgst": cgst, "sgst": sgst, "igst": igst},
                    }
                )

            gst_rate = self._as_float(row.get("gst_rate"))
            if gst_rate is None or gst_rate <= 0:
                if taxable_value > 0:
                    failures.append(
                        {
                            "row_index": int(index),
                            "invoice_number": invoice_number,
                            "reason": "gst_rate_missing_for_tax_validation",
                            "details": {"taxable_value": taxable_value, "tax_amount": tax_amount},
                        }
                    )
            else:
                expected_tax = taxable_value * (gst_rate / 100.0)
                tolerance = max(1.0, 0.01 * max(abs(expected_tax), 1.0))
                if abs(tax_amount - expected_tax) > tolerance:
                    failures.append(
                        {
                            "row_index": int(index),
                            "invoice_number": invoice_number,
                            "reason": "tax_amount_not_matching_taxable_into_rate",
                            "details": {
                                "taxable_value": taxable_value,
                                "gst_rate": gst_rate,
                                "tax_amount": tax_amount,
                                "expected_tax": expected_tax,
                            },
                        }
                    )

        return failures

    def cross_check_gstr1_gstr2a(self, gstr1_df: pd.DataFrame, gstr2a_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """Cross-check invoices between GSTR-1 and GSTR-2A."""
        gstr1_norm = gstr1_df.copy()
        gstr2a_norm = gstr2a_df.copy()

        for frame in [gstr1_norm, gstr2a_norm]:
            frame["invoice_number"] = frame.get("invoice_number", "").fillna("").astype(str).str.strip()
            frame["invoice_date"] = pd.to_datetime(frame.get("invoice_date"), errors="coerce")
            frame["taxable_value"] = frame.get("taxable_value", 0).apply(self._as_float).fillna(0.0)

            if "total" not in frame.columns:
                frame["total"] = frame[[col for col in ["igst", "cgst", "sgst", "taxable_value"] if col in frame.columns]].fillna(0).sum(axis=1)
            else:
                frame["total"] = frame["total"].apply(self._as_float).fillna(0.0)

        gstr1_keys = set(gstr1_norm["invoice_number"])
        gstr2a_keys = set(gstr2a_norm["invoice_number"])

        missing_in_gstr2a_keys = sorted(key for key in gstr1_keys if key and key not in gstr2a_keys)
        missing_in_gstr1_keys = sorted(key for key in gstr2a_keys if key and key not in gstr1_keys)

        missing_in_gstr2a = (
            gstr1_norm[gstr1_norm["invoice_number"].isin(missing_in_gstr2a_keys)]
            [["invoice_number", "invoice_date", "gstin", "taxable_value", "total"]]
            .to_dict(orient="records")
            if "gstin" in gstr1_norm.columns
            else gstr1_norm[gstr1_norm["invoice_number"].isin(missing_in_gstr2a_keys)]
            [["invoice_number", "invoice_date", "taxable_value", "total"]]
            .to_dict(orient="records")
        )

        missing_in_gstr1 = (
            gstr2a_norm[gstr2a_norm["invoice_number"].isin(missing_in_gstr1_keys)]
            [["invoice_number", "invoice_date", "gstin", "taxable_value", "total"]]
            .to_dict(orient="records")
            if "gstin" in gstr2a_norm.columns
            else gstr2a_norm[gstr2a_norm["invoice_number"].isin(missing_in_gstr1_keys)]
            [["invoice_number", "invoice_date", "taxable_value", "total"]]
            .to_dict(orient="records")
        )

        gstr1_amount_map = gstr1_norm.groupby("invoice_number", dropna=False)["total"].sum().to_dict()
        gstr2a_amount_map = gstr2a_norm.groupby("invoice_number", dropna=False)["total"].sum().to_dict()

        common_keys = sorted(key for key in gstr1_keys.intersection(gstr2a_keys) if key)
        amount_mismatches: list[dict[str, Any]] = []
        for invoice_number in common_keys:
            total_1 = float(gstr1_amount_map.get(invoice_number, 0.0))
            total_2 = float(gstr2a_amount_map.get(invoice_number, 0.0))
            tolerance = max(1.0, 0.01 * max(abs(total_1), abs(total_2), 1.0))
            if abs(total_1 - total_2) > tolerance:
                amount_mismatches.append(
                    {
                        "invoice_number": invoice_number,
                        "gstr1_total": total_1,
                        "gstr2a_total": total_2,
                        "difference": total_1 - total_2,
                    }
                )

        return {
            "missing_in_gstr2a": missing_in_gstr2a,
            "missing_in_gstr1": missing_in_gstr1,
            "amount_mismatches": amount_mismatches,
        }

    def _read_best_sheet(self, path: Path, aliases: dict[str, list[str]]) -> pd.DataFrame:
        if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
            raise ValueError(f"Unsupported GST file type: {path.suffix}")

        workbook = pd.ExcelFile(path)

        best_sheet_name: str | None = None
        best_score = -1
        best_df = pd.DataFrame()

        for sheet_name in workbook.sheet_names:
            candidate = workbook.parse(sheet_name)
            score = self._score_mapping(candidate, aliases)
            if score > best_score:
                best_score = score
                best_sheet_name = sheet_name
                best_df = candidate

        if best_sheet_name is None:
            raise ValueError(f"No readable sheets found in Excel file: {path}")

        return best_df

    def _score_mapping(self, df: pd.DataFrame, aliases: dict[str, list[str]]) -> int:
        score = 0
        for candidate_names in aliases.values():
            if self._find_column(df, candidate_names) is not None:
                score += 1
        return score

    def _normalize(self, df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
        normalized = pd.DataFrame(index=df.index)

        for standard_col, candidate_names in aliases.items():
            source_col = self._find_column(df, candidate_names)
            if source_col is None:
                normalized[standard_col] = None
            else:
                normalized[standard_col] = df[source_col]

        normalized["invoice_number"] = normalized["invoice_number"].fillna("").astype(str).str.strip()
        normalized["invoice_date"] = normalized["invoice_date"].apply(self._parse_date)

        for col in ["taxable_value", "igst", "cgst", "sgst", "total", "tax_amount", "gst_rate"]:
            if col in normalized.columns:
                normalized[col] = normalized[col].apply(self._as_float)

        for col in ["buyer_gstin", "supplier_gstin"]:
            if col in normalized.columns:
                normalized[col] = normalized[col].fillna("").astype(str).str.strip().str.upper()

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

    def _as_float(self, value: Any) -> float | None:
        # Use robust clean_amount normalizer
        cleaned = clean_amount(value)
        return cleaned if cleaned != 0.0 else None
