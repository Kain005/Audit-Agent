from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .bank_statement_parser import BankStatementParser
from .pdf_parser import (
    detect_pdf_type_and_scale,
    extract_image_pdf_as_dataframe,
    extract_image_pdf_full,
    extract_native_pdf_as_dataframe,
    extract_text_by_page as extract_pages_as_dicts,
)
from .universal_bank_parser import clean_amount as clean_amount_with_suffix
from .universal_bank_parser import _normalize_date, parse_description

logger = logging.getLogger(__name__)

# Single shared instance - stateless, safe to reuse
_bsp = BankStatementParser()


def resolve_debit_credit(
    row: dict,
    col_map: dict[str, str],
    previous_balance: float | None,
) -> tuple[float, float, float]:
    debit = 0.0
    credit = 0.0
    balance = 0.0

    def _extract_clean_amount(cell: str) -> tuple[float, str | None]:
        """Extract only a clean currency amount from a cell, ignoring bleed text."""
        cell = str(cell or "").strip()
        if not cell:
            return 0.0, None
        # Handle prefix DR/CR
        upper = cell.upper().strip()
        prefix_suffix = None
        if upper.startswith("DR ") or upper.startswith("DR\t"):
            prefix_suffix = "DR"
            cell = cell[3:].strip()
        elif upper.startswith("CR ") or upper.startswith("CR\t"):
            prefix_suffix = "CR"
            cell = cell[3:].strip()
        # Strip leading row-number token e.g. "43 130.00" -> "130.00"
        cell = re.sub(r'^\d{1,3}\s+', '', cell).strip()
        # Normalize common OCR substitutions before matching
        cell = cell.replace("D0", "00").replace("O0", "00").replace("0o", "00")
        cell = cell.replace("Qu", "00").replace("uQ", "00").replace("uu", "00")
        cell = cell.replace("Co", "00").replace("cO", "00").replace("co", "00")
        cell = cell.replace("do", "00").replace("dO", "00")
        cell = re.sub(r'[A-Za-z](?=\d)', '', cell)  # letter immediately before digit
        cell = re.sub(r'(?<=\d)[A-Za-z]', '', cell)  # letter immediately after digit
        # Normalize large whole numbers — OCR strips commas+decimal e.g. 942,184.41 -> 94218441
        # Reinsert decimal point 2 places from right
        if re.fullmatch(r'\d{5,}', cell.strip()):
            raw = cell.strip()
            cell = raw[:-2] + '.' + raw[-2:]
        # Find all clean currency patterns: digits with optional commas and decimal
        matches = list(re.finditer(r"\d[\d,]*\.\d{2}", cell))
        if not matches:
            # Allow whole numbers under 100000 with no decimal
            whole = re.search(r"(?<![.\d])\d{1,6}(?![.\d])", cell)
            if whole:
                try:
                    val = float(whole.group(0))
                    if 0.01 <= val <= 99999:
                        return val, prefix_suffix
                except ValueError:
                    pass
            return 0.0, None
        # Use last match as the amount (rightmost currency value in cell)
        amount_str = matches[-1].group(0).replace(",", "")
        try:
            val = float(amount_str)
        except ValueError:
            return 0.0, None
        if val > 500_000_000:
            return 0.0, None
        # Detect suffix after the amount match
        after = cell[matches[-1].end():].strip().upper()
        suffix = prefix_suffix
        if after.startswith("DR"):
            suffix = "DR"
        elif re.match(r"^CR\b", after):
            suffix = "CR"
        return val, suffix

    if "balance" in col_map:
        balance, _ = _extract_clean_amount(row.get(col_map["balance"], ""))

    if "debit" in col_map:
        debit_val, debit_suffix = _extract_clean_amount(row.get(col_map["debit"], ""))
        if debit_val and debit_suffix != "CR":
            debit = debit_val

    if "credit" in col_map:
        credit_cell = str(row.get(col_map["credit"], "") or "")
        credit_val, credit_suffix = _extract_clean_amount(credit_cell)
        if credit_val and credit_suffix != "DR":
            if credit_suffix == "CR":
                credit = credit_val
        elif debit == 0:
            credit = credit_val

    if "amount" in col_map and debit == 0 and credit == 0:
        amount_val, amount_suffix = _extract_clean_amount(row.get(col_map["amount"], ""))
        if amount_val:
            if amount_suffix == "DR":
                debit = amount_val
            elif amount_suffix == "CR":
                credit = amount_val
            elif previous_balance is not None and balance > 0:
                delta = balance - previous_balance
                if delta > 0:
                    credit = amount_val
                else:
                    debit = amount_val

    if debit == 0 and credit == 0 and previous_balance is not None and balance > 0:
        delta = round(balance - previous_balance, 2)
        if abs(delta) > 0.5:
            if delta > 0:
                credit = round(abs(delta), 2)
            elif delta < 0:
                debit = round(abs(delta), 2)
                
    if credit > 0 and previous_balance is not None and balance > 0:
     expected_delta = round(balance - previous_balance, 2)
     if abs(expected_delta - credit) > 1.0 and abs(expected_delta + credit) > 1.0:
        if debit == 0:
            credit = round(abs(expected_delta), 2) if expected_delta > 0 else 0.0
    return debit, credit, balance

def _detect_bank_from_ocr(pages: list[dict[str, Any]]) -> str:
    """
    Identify the bank from combined OCR page text.

    Delegates to BankStatementParser.detect_bank(), which already has
    keyword signatures for SBI, HDFC, ICICI, Axis, Kotak, PNB, Yes Bank.
    Passing an empty DataFrame forces the text-only path.
    """
    combined = "\n".join(str(page.get("raw_text", "") or "") for page in pages)
    return _bsp.detect_bank(pd.DataFrame(), raw_text=combined)


def _parse_raw_text_transactions(raw_text: str) -> list[dict]:
    """Parse transactions from raw OCR text when dataframe extraction fails."""
    import re
    transactions = []
    DATE_RE = re.compile(r"\b(\d{1,2}[-/]\d{2}[-/]\d{2,4})\b")
    AMOUNT_RE = re.compile(r"\d[\d,]*\.\d{2}")

    # Split on date patterns when text is a single blob (OCR page 1 style)
    lines_raw = raw_text.splitlines()
    if len(lines_raw) <= 2:
        # re-split on every occurrence of a date pattern
        segments = re.split(r'(?=\b\d{2}[-/]\d{2}[-/ ]\d{2,4}\b)', raw_text)
        lines = [s.strip() for s in segments if s.strip()]
    else:
        lines = [l.strip() for l in lines_raw if l.strip()]
    previous_balance = None

    for line in lines:
        dates = DATE_RE.findall(line)
        if not dates:
            continue
        amounts = AMOUNT_RE.findall(line)
        if not amounts:
            continue
        amounts_clean = []
        for a in amounts:
            try:
                amounts_clean.append(float(a.replace(",", "")))
            except ValueError:
                continue
        if not amounts_clean:
            continue

        date_value = _normalize_date(dates[0])
        if not date_value:
            try:
                parsed_dt = pd.to_datetime(dates[0], dayfirst=True, errors="coerce")
                date_value = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else None
            except Exception:
                pass
        if not date_value:
            # try extracting date from description field
            desc_raw = str(row_dict.get(col_map.get("description", ""), "") or "").strip()
            # normalize OCR noise in date e.g. 04-€8-2022 -> 04-08-2022
            desc_clean = re.sub(r'[^\d\-/\s]', '0', desc_raw)
            date_match = re.search(r"\b(\d{2}[-/]\d{2}[-/]\d{2,4})\b", desc_clean)
            if date_match:
                date_value = _normalize_date(date_match.group(1))
                if not date_value:
                    try:
                        parsed_dt = pd.to_datetime(date_match.group(1), dayfirst=True, errors="coerce")
                        date_value = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else None
                    except Exception:
                        pass
            if not date_value:
                if transactions:
                    date_value = transactions[-1]["date"]
                else:
                    continue

        balance = amounts_clean[-1] if len(amounts_clean) >= 1 else 0.0
        debit = 0.0
        credit = 0.0

        if len(amounts_clean) >= 2:
            amount = amounts_clean[-2]
            if previous_balance is not None and balance > 0:
                delta = round(balance - previous_balance, 2)
                if delta > 0:
                    credit = amount
                else:
                    debit = amount
            else:
                debit = amount

        if balance > 0:
            previous_balance = balance

        desc_part = re.sub(r"\d[\d,]*\.\d{2}", "", line)
        desc_part = re.sub(r"\b\d{1,2}[-/]\d{2}[-/]\d{2,4}\b", "", desc_part)
        description = re.sub(r"\s+", " ", desc_part).strip()

        transactions.append({
            "date": date_value,
            "description": description,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "ref_no": "",
            "tags": {},
        })

    return transactions


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
    pdf_type, _ = detect_pdf_type_and_scale(pdf_path)
    print("parse_bank_pdf: pdf_type=%s path=%s", pdf_type, pdf_path)
    if pdf_type == "image_pdf":
     pages_full = extract_image_pdf_full(pdf_path)
    else:
     pages_full = extract_pages_as_dicts(pdf_path)
    combined_text = "\n".join(str(page.get("raw_text", "") or "") for page in pages_full)

    # Step 1: text-based bank detection
    detected_bank = _detect_bank_from_ocr(pages_full)
    logger.info("OCR bank detection: '%s' for %s", detected_bank, pdf_path)

    structured_transactions: list[dict[str, Any]] = []
    structured_pages: list[dict[str, Any] | None]
    if pdf_type == "image_pdf":
     structured_pages = [
        {"page_number": p["page_number"], "dataframe": p.get("dataframe"), "raw_text": p.get("raw_text", "")}
        for p in pages_full
    ]
    else:
     structured_pages = extract_native_pdf_as_dataframe(pdf_path)

    last_balance: float | None = None
    for page_result in structured_pages or []:
        if not page_result:
            continue
        dataframe = page_result.get("dataframe")
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            page_num = page_result.get("page_number", "?")
            raw = str(page_result.get("raw_text", "") or "")
            if raw:
                fallback_txns = _parse_raw_text_transactions(raw)
                logger.info("Page %s dataframe empty — raw text fallback: %d transactions", page_num, len(fallback_txns))
                structured_transactions.extend(fallback_txns)
                if fallback_txns:
                    last_balance = float(fallback_txns[-1].get("balance") or 0) or last_balance
            continue
        page_txns = parse_dataframe_transactions(dataframe, filename=pdf_path, seed_balance=last_balance)
        structured_transactions.extend(page_txns)
        if page_txns:
            last_balance = float(page_txns[-1].get("balance") or 0) or last_balance

    print("parse_bank_pdf: dataframe path yielded %d transactions", len(structured_transactions))

    if structured_transactions:
        deduped_transactions: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for transaction in structured_transactions:
            key = (
                transaction.get("date", ""),
                transaction.get("description", ""),
                float(transaction.get("debit", 0.0) or 0.0),
                float(transaction.get("credit", 0.0) or 0.0),
                float(transaction.get("balance", 0.0) or 0.0),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_transactions.append(transaction)

        if deduped_transactions:
            return {
                "bank": detected_bank,
                "transactions": deduped_transactions,
                "raw_text": combined_text,
                "pages_full": pages_full,
                "partial": True,
                "parser_used": "STRUCTURED_DATAFRAME",
            }

    # Step 4: all layout parsers failed
    # Return detected_bank + raw_text so document_router._parse_ocr_text_to_transactions()
    # can still extract transactions via regex.
    # Log that the text/regex fallback will run and report zero transactions here
    transactions = []
    print("parse_bank_pdf: text fallback yielded %d transactions", len(transactions))

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


def parse_dataframe_transactions(df: pd.DataFrame, filename: str = "", seed_balance: float | None = None) -> list[dict]:
    if df is None or df.empty:
        return []

    normalized_df = df.copy()
    normalized_df.columns = [str(column).strip().lower() for column in normalized_df.columns]

    def _first_matching_column(candidates: set[str]) -> str | None:
     for column in normalized_df.columns:
        if column in candidates:
            return column
     for column in normalized_df.columns:
        if any(column == candidate for candidate in candidates):
            return column
     return None

    col_map: dict[str, str] = {}
    mappings = {
        "date": {"date", "txn date", "transaction date", "value date", "posting date"},
        "description": {"description", "narration", "particulars", "details", "remarks", "txn remarks"},
        "debit": {"debit", "withdrawal", "withdrawl", "debit amount", "dr"},
        "credit": {"credit", "deposit", "credit amount", "cr"},
        "balance": {"balance", "closing", "running"},
        "amount": {"amount", "txn amount"},
        "ref_no": {"ref", "chq", "cheque", "chq/ref", "ref no", "cheque no", "reference"},
    }
    for canonical_name, candidates in mappings.items():
        matched_column = _first_matching_column(candidates)
        if matched_column is not None:
            col_map[canonical_name] = matched_column

    if "date" not in col_map:
        return []

    transactions: list[dict[str, Any]] = []
    previous_balance: float | None = seed_balance

    def _cell_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value or "").strip()
        
    for _, row in normalized_df.iterrows():
        row_dict = row.to_dict()
        date_value = _normalize_date(row_dict.get(col_map["date"], ""))
        if not date_value:
            raw_date = str(row_dict.get(col_map["date"], "") or "").strip()
            if raw_date:
                raw_date = re.split(r'\s{2,}', raw_date)[0].strip()
                raw_date = re.sub(r'(\d{2}-\d{2})\s(\d{4})', r'\1-\2', raw_date)
                # Handle "29-06-2022 29-06-2022" duplicate — take first date only
                raw_date = re.split(r'\s+\d{2}[-/]\d{2}[-/]\d{2,4}', raw_date)[0].strip()
                try:
                    parsed_dt = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
                    if pd.notna(parsed_dt):
                        date_value = parsed_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        if not date_value:
            desc_raw = str(row_dict.get(col_map.get("description", ""), "") or "").strip()
            desc_clean = re.sub(r"[^\d\-/\s]", "0", desc_raw)
            date_match = re.search(r"\b(\d{2}[-/]\d{2}[-/]\d{2,4})\b", desc_clean)
            if date_match:
                date_value = _normalize_date(date_match.group(1))
                if not date_value:
                    try:
                        parsed_dt = pd.to_datetime(date_match.group(1), dayfirst=True, errors="coerce")
                        date_value = parsed_dt.strftime("%Y-%m-%d") if pd.notna(parsed_dt) else None
                    except Exception:
                        pass
        if not date_value:
            if transactions:
                date_value = transactions[-1]["date"]
            else:
                continue

        description = ""
        if "description" in col_map:
            description = re.sub(r"\s+", " ", _cell_text(row_dict.get(col_map["description"], ""))).strip()
        if not description:
            description = ""


        debit, credit, balance = resolve_debit_credit(row_dict, col_map, previous_balance)
        if debit == 0 and credit == 0 and balance == 0:
            continue
        if debit == 0 and credit == 0 and balance > 0:
            transactions.append({
                "date": date_value,
                "description": description,
                "debit": 0.0,
                "credit": 0.0,
                "balance": float(balance),
                "ref_no": "",
                "tags": {},
            })
            previous_balance = balance
            continue

        ref_no = ""
        if "ref_no" in col_map:
            ref_no = re.sub(r"\s+", " ", _cell_text(row_dict.get(col_map["ref_no"], ""))).strip()

        transaction = {
            "date": date_value,
            "description": description,
            "debit": float(debit or 0.0),
            "credit": float(credit or 0.0),
            "balance": float(balance or 0.0),
            "ref_no": ref_no,
            "tags": {},
        }
        transactions.append(transaction)

        if balance > 0:
            previous_balance = balance
    
    if transactions and transactions[0]["debit"] == 0 and transactions[0]["credit"] == 0:
     first_bal = transactions[0]["balance"]
     second_bal = transactions[1]["balance"] if len(transactions) > 1 else None
     if second_bal is not None:
        desc = transactions[0]["description"]
        amt_match = re.search(r"(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)(?!\d)", desc)
        if amt_match:
            amt = float(amt_match.group(1).replace(",", ""))
            delta = round(first_bal - (second_bal + transactions[1]["debit"] - transactions[1]["credit"]), 2)
            if amt > 0:
                if first_bal > second_bal:
                    transactions[0]["credit"] = amt
                else:
                    transactions[0]["debit"] = amt

    for transaction in transactions:
        try:
            transaction["tags"] = parse_description(str(transaction.get("description", "") or ""))
        except Exception:
            transaction["tags"] = {}

    deduped: list[dict] = []
    seen: set[tuple[Any, ...]] = set()
    for transaction in transactions:
        key = (
            transaction.get("date", ""),
            transaction.get("description", ""),
            float(transaction.get("debit", 0.0) or 0.0),
            float(transaction.get("credit", 0.0) or 0.0),
            float(transaction.get("balance", 0.0) or 0.0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(transaction)

    return deduped


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


