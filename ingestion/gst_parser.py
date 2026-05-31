"""GST Excel parser and reconciliation utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


import pandas as pd

from .parsers.normalizers import clean_amount, clean_date


class GSTParser:
    """Parse GST Excel exports and validate GST entries."""

    GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]$")

    def looks_like_gst_file(self, file_path: str) -> bool:
        """Return True if workbook appears to be any supported GST format."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.looks_like_gst_pdf(file_path)
        return False

    def looks_like_gst_pdf(self, file_path: str) -> bool:
        """Return True if a PDF looks like a GST invoice with HSN/SAC line items."""
        gstin_pattern = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]", re.IGNORECASE)
        # Must have HSN or SAC to be a true GST invoice — regular invoices won't have these
        required_tokens = {"HSN", "SAC"}
        supporting_tokens = ["CGST", "SGST", "IGST", "GSTIN", "TAX INVOICE", "TAXABLE VALUE"]

        try:
            import pdfplumber

            gstin_found = False
            required_hits = set()
            supporting_hits = 0

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:2]:
                    text = page.extract_text() or ""
                    upper_text = text.upper()
                    if gstin_pattern.search(text):
                        gstin_found = True
                    for token in required_tokens:
                        if token in upper_text:
                            required_hits.add(token)
                    supporting_hits += sum(1 for token in supporting_tokens if token in upper_text)

            # Must have GSTIN + at least one of HSN/SAC + 2 supporting tokens
            return gstin_found and len(required_hits) >= 1 and supporting_hits >= 2
        except Exception:
            return False

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

    def _parse_date(self, value: Any) -> Any:
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
