"""GST invoice PDF parser — extracts structured invoice data from GST-compliant PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

try:
    from ..extraction.models import InvoiceEntities, LineItem
    from .parsers.normalizers import clean_amount, clean_date
except ImportError:
    from extraction.models import InvoiceEntities, LineItem
    from ingestion.parsers.normalizers import clean_amount, clean_date

GSTIN_RE = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]", re.IGNORECASE)
GST_KEYWORDS = ["CGST", "SGST", "IGST", "HSN", "SAC", "GSTIN", "TAX INVOICE"]


def looks_like_gst_invoice_pdf(file_path: str) -> bool:
    try:
        gstin_found = False
        keyword_hits = 0

        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:2]
            text_chunks: list[str] = []
            for page in pages:
                text_chunks.append(page.extract_text() or "")

        text = "\n".join(text_chunks)
        gstin_found = bool(GSTIN_RE.search(text))
        upper_text = text.upper()
        keyword_hits = sum(1 for keyword in GST_KEYWORDS if keyword in upper_text)
        return gstin_found and keyword_hits >= 2
    except Exception:
        return False


def parse_pdf(file_path: str) -> InvoiceEntities:
    try:
        pages_text: list[str] = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)

        full_text = "\n".join(t for t in pages_text if t.strip())

        def _first_match(patterns: list[str], text: str) -> str:
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return ""

        def _extract_amount(text: str) -> float | None:
            match = re.search(r"[₹Rs\.\s]*(\d[\d,]*(?:\.\d{1,2})?)", text)
            if not match:
                return None
            return clean_amount(match.group(1))

        # --- GSTINs ---
        all_gstins = GSTIN_RE.findall(full_text)
        vendor_gstin = all_gstins[0].upper() if all_gstins else ""
        buyer_gstin = all_gstins[1].upper() if len(all_gstins) >= 2 else ""

        # --- Vendor name ---
        # Look for explicit "For <name>" signature block first
        vendor_name = ""
        for_match = re.search(r"\bFor\s+([A-Za-z][\w\s&\.]+?)(?:\n|$)", full_text)
        if for_match:
            vendor_name = for_match.group(1).strip()
        if not vendor_name:
            # Fall back to line immediately before first GSTIN occurrence
            if vendor_gstin:
                idx = full_text.upper().find(vendor_gstin.upper())
                prefix = full_text[:idx] if idx > 0 else ""
                for line in reversed(prefix.splitlines()):
                    stripped = line.strip()
                    if stripped and not any(k in stripped.upper() for k in ("GSTIN", "PAN", "TAX", "INVOICE")):
                        vendor_name = stripped
                        break

        # --- Buyer name ---
        buyer_name = _first_match(
            [
                r"M/S\s+([A-Za-z][\w\s&\.]+?)(?:\s+Challan|\s+GSTIN|\s+Address|\n|$)",
                r"Bill\s+To\s*[:\-]?\s*([A-Za-z][\w\s&\.]+?)(?:\n|GSTIN|$)",
                r"Buyer\s*[:\-]?\s*([A-Za-z][\w\s&\.]+?)(?:\n|GSTIN|$)",
                r"Consignee\s*[:\-]?\s*([A-Za-z][\w\s&\.]+?)(?:\n|GSTIN|$)",
            ],
            full_text,
        )

        # --- Invoice number ---
        invoice_number = _first_match(
            [
                r"Invoice\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"Invoice\s+#\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
                r"Bill\s+No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
            ],
            full_text,
        )

        # --- Invoice date ---
        invoice_date_raw = _first_match(
            [
                r"Invoice\s+Date\s*[:\-]?\s*(\d{1,2}[\-/]\w+[\-/]\d{2,4})",
                r"Invoice\s+Date\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{2,4})",
                r"Date\s*[:\-]?\s*(\d{1,2}[\-/]\w+[\-/]\d{2,4})",
                r"Date\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{2,4})",
            ],
            full_text,
        )
        invoice_date = clean_date(invoice_date_raw.strip()) if invoice_date_raw else None

        # --- Place of supply ---
        pos_match = re.search(
            r"Place\s+of\s+Supply\s*[:\-]?\s*([A-Za-z\s]+(?:\(\s*\d+\s*\))?)"
            r"|Place\s+of\s*\n\s*Supply\s*\n?\s*([A-Za-z\s]+(?:\(\s*\d+\s*\))?)",
            full_text, flags=re.IGNORECASE
        )
        place_of_supply = None
        if pos_match:
            place_of_supply = (pos_match.group(1) or pos_match.group(2) or "").strip() or None

        # --- Taxable value / subtotal ---
        subtotal = None
        taxable_patterns = [
            r"Taxable\s+Value\s*[:\-]?\s*([₹Rs\.?\s0-9,]+)",
            r"Taxable\s+Amount\s*[:\-]?\s*([₹Rs\.?\s0-9,]+)",
            # last standalone number on a line that looks like a subtotal row
            r"(\d[\d,]+\.\d{2})\s*\n\s*(?:IGST|CGST|SGST|GST)",
        ]
        for pat in taxable_patterns:
            m = re.search(pat, full_text, flags=re.IGNORECASE)
            if m:
                subtotal = clean_amount(m.group(1))
                if subtotal and subtotal > 1:
                    break

        # --- GST amounts ---
        def _gst_component(label: str) -> float | None:
            # Match "IGST (18.00 %) 684.90" or "IGST 18% 684.90" style
            m = re.search(
                rf"{label}\s*[\(]?\s*[\d\.]+\s*%?\s*[\)]?\s*([₹Rs\.\s]*[\d,]+\.\d{{2}})",
                full_text, flags=re.IGNORECASE
            )
            if m:
                return clean_amount(m.group(1))
            return None

        cgst_amount = _gst_component("CGST")
        sgst_amount = _gst_component("SGST")
        igst_amount = _gst_component("IGST")

        gst_amount_values = [v for v in [cgst_amount, sgst_amount, igst_amount] if v is not None]
        gst_amount = sum(gst_amount_values) if gst_amount_values else None

        # --- Total amount ---
        total_amount = None
        total_patterns = [
            r"Grand\s+Total\s*[:\-]?\s*[₹Rs\.?\s]*([\d,]+\.\d{2})",
            r"Total\s+Amount\s*[:\-]?\s*[₹Rs\.?\s]*([\d,]+\.\d{2})",
            r"Invoice\s+Value\s*[:\-]?\s*[₹Rs\.?\s]*([\d,]+\.\d{2})",
            r"Total\s+\d+\s+\w+\s+[₹\s]*([\d,]+\.\d{2})",
            r"[₹]\s*([\d,]+\.\d{2})",
        ]
        for pat in total_patterns:
            m = re.search(pat, full_text, flags=re.IGNORECASE)
            if m:
                total_amount = clean_amount(m.group(1))
                if total_amount and total_amount > 1:
                    break

        # --- GST rate ---
        gst_rate_raw = _first_match(
            [
                r"IGST\s*[\(]?\s*([\d\.]+)\s*%",
                r"CGST\s*[\(]?\s*([\d\.]+)\s*%",
                r"GST\s+Rate\s*[:\-]?\s*([\d\.]+)\s*%",
                r"Tax\s+Rate\s*[:\-]?\s*([\d\.]+)\s*%",
            ],
            full_text,
        )
        gst_rate = clean_amount(gst_rate_raw) if gst_rate_raw else None

        # --- Line items from tables ---
        line_items: list[LineItem] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    for table in (page.extract_tables() or []):
                        for row in table:
                            if not row or len(row) < 3:
                                continue
                            first_cell = str(row[0] or "").strip()
                            if not first_cell or first_cell.isdigit() is False and len(first_cell) < 2:
                                continue
                            if any(h in first_cell.lower() for h in ("description", "item", "name", "product", "sr", "s.no")):
                                continue
                            amount_raw = str(row[-1] or "").strip()
                            amount = clean_amount(amount_raw)
                            if not amount or amount < 0.01:
                                continue
                            quantity = clean_amount(str(row[2] or "")) if len(row) > 2 else None
                            unit_price = clean_amount(str(row[3] or "")) if len(row) > 3 else None
                            line_items.append(LineItem(
                                description=first_cell,
                                quantity=quantity,
                                unit_price=unit_price,
                                amount=amount,
                                gst_rate=gst_rate,
                            ))
        except Exception:
            line_items = []

        if not line_items:
            line_items = [LineItem(
                description="(from invoice total)",
                quantity=None,
                unit_price=None,
                amount=subtotal or total_amount or 0.0,
                gst_rate=gst_rate,
            )]

        return InvoiceEntities(
            vendor_name=vendor_name or None,
            vendor_gst=vendor_gstin or None,
            vendor_address=None,
            invoice_number=invoice_number or None,
            invoice_date=invoice_date or None,
            due_date=None,
            line_items=line_items,
            subtotal=subtotal,
            gst_amount=gst_amount,
            total_amount=total_amount,
            payment_terms=None,
            currency="INR",
            buyer_name=buyer_name or None,
            buyer_gst=buyer_gstin or None,
            place_of_supply=place_of_supply or None,
            cgst_amount=cgst_amount,
            sgst_amount=sgst_amount,
            igst_amount=igst_amount,
            raw_text=full_text[:2000] if full_text else None,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse GST invoice PDF: {exc}") from exc

def _find_column(df: pd.DataFrame, candidate_names: list[str]) -> str | None:
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


def _parse_date(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    # Use robust clean_date normalizer
    cleaned = clean_date(value)
    if cleaned:
        return pd.to_datetime(cleaned, errors="coerce")
    return pd.NaT


def _as_float(value: Any) -> float | None:
    # Use robust clean_amount normalizer
    cleaned = clean_amount(value)
    return cleaned if cleaned != 0.0 else None
