from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .bank_layouts.hdfc_layout import HDFCLayoutParser
from .bank_layouts.sbi_layout import SBILayoutParser
from .bank_statement_parser import BankStatementParser
from .pdf_parser import extract_text_by_page as extract_pages_as_dicts

logger = logging.getLogger(__name__)

# Single shared instance - stateless, safe to reuse
_bsp = BankStatementParser()

# Registry: maps detect_bank() return values to layout parser classes.
# Add new banks here as layout parsers are written.
_LAYOUT_PARSERS: dict[str, Any] = {
    "hdfc": HDFCLayoutParser,
    "sbi": SBILayoutParser,
}

def _detect_bank_from_ocr(pages: list[dict[str, Any]]) -> str:
    """
    Identify the bank from combined OCR page text.

    Delegates to BankStatementParser.detect_bank(), which already has
    keyword signatures for SBI, HDFC, ICICI, Axis, Kotak, PNB, Yes Bank.
    Passing an empty DataFrame forces the text-only path.
    """
    combined = "\n".join(str(page.get("raw_text", "") or "") for page in pages)
    return _bsp.detect_bank(pd.DataFrame(), raw_text=combined)


def parse_bank_pdf(pdf_path: str) -> dict:
    """
    Parse a bank statement PDF and return transactions + metadata.

    Detection strategy (in order):
      1. OCR keyword matching  ->  fast, works for scanned PDFs
      2. Layout parser for the detected bank  ->  precise column extraction
      3. All remaining layout parsers  ->  fallback for structured PDFs whose
         header text was not picked up cleanly by OCR
      4. Return detected bank + raw_text so document_router can run regex
         extraction as a last resort (Bug 1 fix hook)
    """
    pages_full = extract_pages_as_dicts(pdf_path)
    combined_text = "\n".join(str(page.get("raw_text", "") or "") for page in pages_full)

    # Step 1: text-based bank detection
    detected_bank = _detect_bank_from_ocr(pages_full)
    logger.info("OCR bank detection: '%s' for %s", detected_bank, pdf_path)

    # Step 2: run the matching layout parser first
    if detected_bank in _LAYOUT_PARSERS:
        layout_cls = _LAYOUT_PARSERS[detected_bank]
        try:
            if layout_cls.detect(pages_full):
                parsed = layout_cls.extract(pages_full)
                parsed["parser_used"] = detected_bank.upper()
                parsed.setdefault("bank", detected_bank)
                logger.info("Layout parser '%s' succeeded", detected_bank)
                return parsed
        except Exception as exc:
            logger.warning("Layout parser '%s' raised: %s", detected_bank, exc)

    # Step 3: try remaining layout parsers (structured PDFs)
    for bank_key, layout_cls in _LAYOUT_PARSERS.items():
        if bank_key == detected_bank:
            continue  # already tried
        try:
            if layout_cls.detect(pages_full):
                parsed = layout_cls.extract(pages_full)
                parsed["parser_used"] = bank_key.upper()
                parsed.setdefault("bank", bank_key)
                logger.info("Fallback layout parser '%s' succeeded", bank_key)
                return parsed
        except Exception as exc:
            logger.warning("Fallback layout parser '%s' raised: %s", bank_key, exc)

    # Step 4: all layout parsers failed
    # Return detected_bank + raw_text so document_router._parse_ocr_text_to_transactions()
    # can still extract transactions via regex.
    logger.warning(
        "All layout parsers failed for '%s'. Detected bank: '%s'. "
        "Returning raw_text for regex fallback.",
        pdf_path,
        detected_bank,
    )
    return {
        "bank": detected_bank,
        "transactions": [],
        "raw_text": combined_text,
        "pages_full": pages_full,
        "parse_warnings": [
            f"No layout parser matched. Bank identified as '{detected_bank}' "
            "via OCR keywords. Regex fallback will be attempted."
        ],
        "partial": True,
        "parser_used": None,
    }


def to_standard_transactions(parsed: dict) -> list[dict[str, Any]]:
    """
    Convert parsed bank output to the pipeline's standard schema.

    Standard schema (matches what analyze_node expects):
        date, description, debit, credit, balance,
        transaction_id, bank_name, category
    """
    bank = parsed.get("bank", "unknown")
    txs = parsed.get("transactions") or []

    standard: list[dict[str, Any]] = []
    for t in txs:
        # transaction_id: accept several field names that layout parsers may use
        txn_id = (
            t.get("transaction_id")
            or t.get("ref_no")
            or t.get("cheque_no")
            or t.get("reference")
            or ""
        )
        standard.append(
            {
                "date": t.get("date"),
                "description": t.get("description"),
                "debit": t.get("debit"),
                "credit": t.get("credit"),
                "balance": t.get("balance"),
                "transaction_id": txn_id,
                "bank_name": bank,
                "category": t.get("category") or "",
            }
        )

    return standard
