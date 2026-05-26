from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import pandas as pd

from .pdf_parser import extract_text_by_page

logger = logging.getLogger(__name__)

LAST_PARSE_DEBUG: dict[str, Any] = {
    "skip_log": [],
    "pages_processed": 0,
    "rows_attempted": 0,
    "rows_skipped": 0,
}

DATE_RE = re.compile(r"\d{1,2}[-/]\d{2}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]{2,4}\s+\d{2,4}")
BALANCE_END_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\s*(?:CR|DR)?\s*$", flags=re.IGNORECASE)
AMOUNT_RE = re.compile(r"\d[\d,]*\.\d{2}\s*(?:CR|DR)?", flags=re.IGNORECASE)

HEADER_KEYWORDS = {
    "date",
    "txn",
    "txn date",
    "value",
    "narration",
    "description",
    "particulars",
    "details",
    "remarks",
    "debit",
    "dr",
    "withdrawal",
    "credit",
    "cr",
    "deposit",
    "balance",
    "ref",
    "chq",
    "cheque",
    "reference",
    "amount",
    "txn amount",
}

KNOWN_BANK_KEYWORDS = [
    "hdfc",
    "icici",
    "sbi",
    "axis",
    "kotak",
    "pnb",
    "bob",
    "canara",
    "union",
    "idbi",
    "yes",
    "indusind",
    "federal",
    "iob",
    "uco",
    "bandhan",
    "rbl",
    "idfc",
]


def _default_description_tags() -> dict[str, Any]:
    return {
        "payment_mode": None,
        "transaction_id": None,
        "receiver_name": None,
        "receiver_bank": None,
        "receiver_upi": None,
        "phone_number": None,
        "category": None,
        "is_debit_reversal": False,
        "is_salary": False,
        "is_emi": False,
    }


def _split_row(line: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\s{2,}|\t|\|", line) if part.strip()]
    return parts


def _has_numeric_content(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def _normalize_date(text: str) -> str:
    match = DATE_RE.search(str(text or ""))
    if not match:
        return ""

    value = match.group(0)
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue

    # Try textual month forms (e.g. "12 Oct 2020"), including common OCR-truncated
    # month tokens produced by EasyOCR (e.g. "Oc", "Ja", "Fe", "Se").
    try:
        value_text = value
        # Normalize common truncations to full 3-letter month abbreviations
        value_text = re.sub(r"\bOc\b", "Oct", value_text)
        value_text = re.sub(r"\bJa\b", "Jan", value_text)
        value_text = re.sub(r"\bFe\b", "Feb", value_text)
        value_text = re.sub(r"\bSe\b", "Sep", value_text)

        return datetime.strptime(value_text, "%d %b %Y").strftime("%d-%m-%Y")
    except ValueError:
        try:
            return datetime.strptime(value_text, "%d %b %y").strftime("%d-%m-%Y")
        except Exception:
            pass

    if len(value.split("-")[-1]) == 2 or len(value.split("/")[-1]) == 2:
        separator = "-" if "-" in value else "/"
        day, month, year = value.split(separator)
        year = f"20{year}"
        try:
            return datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y").strftime("%d-%m-%Y")
        except ValueError:
            return ""

    return ""


def _clean_description_text(text: str) -> str:
    cleaned = str(text or "").replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_transaction_start(line: str) -> bool:
    return bool(DATE_RE.search(str(line or "")))


def _is_balance_like_end(line: str) -> bool:
    return bool(BALANCE_END_RE.search(str(line or "")))


def _is_orphan_description_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if _is_transaction_start(text) or _is_balance_like_end(text):
        return False
    if AMOUNT_RE.search(text):
        return False
    if any(keyword in text.lower() for keyword in ("txn date", "statement", "account", "opening balance", "closing balance")):
        return False
    if not re.fullmatch(r"[A-Za-z0-9\s/._&\-,:+@()'\"-]+", text):
        return False
    return any(character.isalpha() for character in text)


def _clean_merged_line(line: str) -> str:
    cleaned = str(line or "").replace("|", " ")
    cleaned = re.sub(r"\b[oO0]{4,}\b", " ", cleaned)
    cleaned = cleaned.replace("½", " ").replace("æ", " ").replace("¿", " ").replace("░", " ")
    cleaned = "".join(character if (ord(character) < 128 or character == "₹") else " " for character in cleaned)
    matches = list(DATE_RE.finditer(cleaned))
    if len(matches) > 1:
        first_match = matches[0]
        prefix = cleaned[: first_match.end()]
        suffix = DATE_RE.sub(" ", cleaned[first_match.end() :])
        cleaned = prefix + suffix
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def merge_transaction_lines(lines: list[str]) -> list[str]:
    merged_lines: list[str] = []
    buffer: list[str] = []
    transaction_started = False
    index = 0

    def append_orphans(start_index: int) -> int:
        orphan_count = 0
        current_index = start_index
        while current_index < len(lines) and orphan_count < 2:
            orphan_line = str(lines[current_index] or "").strip()
            if not _is_orphan_description_line(orphan_line):
                break
            if not merged_lines:
                break
            merged_lines[-1] = _clean_merged_line(f"{merged_lines[-1]} {orphan_line}")
            orphan_count += 1
            current_index += 1
        return current_index

    while index < len(lines or []):
        line = str(lines[index] or "").strip()
        if not line:
            index += 1
            continue

        if not transaction_started:
            if _is_transaction_start(line):
                transaction_started = True
                buffer = [line]
                if _is_balance_like_end(line):
                    merged_lines.append(_clean_merged_line(" ".join(buffer)))
                    buffer = []
                    transaction_started = False
                    index = append_orphans(index + 1)
                    continue
            else:
                merged_lines.append(line)
                index += 1
            continue

        if _is_transaction_start(line):
            if buffer:
                merged_lines.append(_clean_merged_line(" ".join(buffer)))
            buffer = [line]
            if _is_balance_like_end(line):
                merged_lines.append(_clean_merged_line(" ".join(buffer)))
                buffer = []
                transaction_started = False
                index = append_orphans(index + 1)
                continue
            index += 1
            continue

        buffer.append(line)
        if len(buffer) >= 4 or _is_balance_like_end(line):
            merged_lines.append(_clean_merged_line(" ".join(buffer)))
            buffer = []
            transaction_started = False
            index = append_orphans(index + 1)
            continue

        index += 1

    if buffer:
        merged_lines.append(_clean_merged_line(" ".join(buffer)))

    return [line for line in merged_lines if line]


def clean_amount(text: Any) -> tuple[float, str | None]:
    raw = str(text or "").strip()
    if not raw:
        return 0.0, None

    prefix_suffix: str | None = None
    stripped = raw.lstrip()
    if stripped.upper().startswith("DR ") or stripped.upper().startswith("DR\t"):
        prefix_suffix = "DR"
        raw = stripped[2:]
    elif stripped.upper().startswith("CR ") or stripped.upper().startswith("CR\t"):
        prefix_suffix = "CR"
        raw = stripped[2:]

    compact = raw.upper()
    compact = compact.replace("₹", "")
    compact = re.sub(r"\b(?:RS\.?|INR)\b", "", compact)
    compact = compact.replace(",", "").replace(" ", "")
    compact = compact.replace("O", "0").replace("B", "8")

    suffix: str | None = None
    if compact.endswith("CR"):
        suffix = "CR"
        compact = compact[:-2]
    elif compact.endswith("DR"):
        suffix = "DR"
        compact = compact[:-2]

    compact = compact.replace("+", "")
    negative = compact.startswith("-")
    compact = compact.lstrip("-")

    match = re.search(r"[\d,]+\.?\d*", compact)
    if not match:
        return 0.0, None

    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return 0.0, None

    if prefix_suffix:
        suffix = prefix_suffix
    if negative and not suffix:
        suffix = "DR"

    return value, suffix


def parse_description(description: str) -> dict[str, Any]:
    try:
        text = _clean_description_text(description)
        if not text:
            return _default_description_tags()

        lower = text.lower()
        tags = _default_description_tags()

        payment_checks = [
            ("UPI", ["upi"]),
            ("NEFT", ["neft"]),
            ("RTGS", ["rtgs"]),
            ("IMPS", ["imps"]),
            ("ATM", ["atm", "wdl", "atw"]),
            ("ACH", ["ach"]),
            ("NACH", ["nach"]),
            ("CLG", ["clg", "clearing"]),
            ("POS", ["pos", "ecom"]),
            ("EMI", ["emi"]),
            ("SI", ["si/", "si ", "standing inst"]),
        ]
        for payment_mode, patterns in payment_checks:
            if any(pattern in lower for pattern in patterns):
                tags["payment_mode"] = payment_mode
                break

        match = None
        upi_match = re.search(r"[a-zA-Z0-9._-]+@[a-zA-Z]{3,}", text, flags=re.IGNORECASE)
        if upi_match:
            tags["receiver_upi"] = upi_match.group(0)

        phone_match = re.search(r"\b[6-9]\d{9}\b", text, flags=re.IGNORECASE)
        if phone_match:
            tags["phone_number"] = phone_match.group(0)

        tags["receiver_bank"] = None
        for bank_keyword in KNOWN_BANK_KEYWORDS:
            bank_match = re.search(rf"\b{re.escape(bank_keyword)}\b", text, flags=re.IGNORECASE)
            if bank_match:
                tags["receiver_bank"] = bank_match.group(0).upper()
                break

        transaction_id = None
        if tags["payment_mode"] == "UPI":
            parts = [part.strip() for part in re.split(r"/", text) if part.strip()]
            for part in parts:
                numeric_part = re.search(r"\b\d{12}\b", part)
                if numeric_part:
                    transaction_id = numeric_part.group(0)
                    break
            if transaction_id is None:
                for part in parts:
                    token_match = re.search(r"[A-Z0-9]{8,}", part, flags=re.IGNORECASE)
                    if token_match:
                        transaction_id = token_match.group(0)
                        break
        elif tags["payment_mode"] in {"NEFT", "RTGS", "IMPS"}:
            parts = [part.strip() for part in re.split(r"/", text) if part.strip()]
            for part in parts[1:]:
                token_match = re.search(r"[A-Z0-9]{8,}", part, flags=re.IGNORECASE)
                if token_match:
                    transaction_id = token_match.group(0)
                    break
        elif tags["payment_mode"] in {"ACH", "NACH"}:
            token_match = re.search(r"\b[A-Z0-9]{8,}\b", text, flags=re.IGNORECASE)
            if token_match:
                transaction_id = token_match.group(0)
        else:
            token_matches = re.findall(r"\b[A-Z0-9]{8,}\b", text, flags=re.IGNORECASE)
            for token in token_matches:
                if not any(bank in token.lower() for bank in KNOWN_BANK_KEYWORDS):
                    transaction_id = token
                    break

        if transaction_id:
            tags["transaction_id"] = transaction_id

        receiver_name = None
        if tags["payment_mode"] == "UPI":
            parts = [part.strip() for part in re.split(r"/", text) if part.strip()]
            if len(parts) >= 3:
                candidate = parts[2]
                candidate = re.sub(r"[^A-Za-z\s]", " ", candidate)
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if candidate and (len(candidate) >= 4 or " " in candidate):
                    receiver_name = candidate.title()
        elif tags["payment_mode"] == "NEFT":
            parts = [part.strip() for part in re.split(r"/", text) if part.strip()]
            if parts:
                candidate = parts[-1]
                candidate = re.sub(r"[^A-Za-z\s]", " ", candidate)
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if candidate and any(ch.isalpha() for ch in candidate) and not any(ch.isdigit() for ch in candidate):
                    if len(candidate.replace(" ", "")) >= 4 and not (candidate.isupper() and len(candidate.replace(" ", "")) < 4):
                        receiver_name = candidate.title()

        transfer_match = re.search(r"(?:\bTO\\?/?\b|\bTRF TO\b)\s*([^|/-]+)", text, flags=re.IGNORECASE)
        if transfer_match:
            candidate = transfer_match.group(1)
            candidate = re.sub(r"[^A-Za-z\s]", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate and any(ch.isalpha() for ch in candidate):
                if not (candidate.isupper() and len(candidate.replace(" ", "")) < 4):
                    receiver_name = candidate.title()

        if receiver_name is None:
            fallback = re.search(r"\bTO\b\s+(.+)$", text, flags=re.IGNORECASE)
            if fallback:
                candidate = fallback.group(1)
                candidate = re.sub(r"[^A-Za-z\s]", " ", candidate)
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if candidate and any(ch.isalpha() for ch in candidate):
                    if not (candidate.isupper() and len(candidate.replace(" ", "")) < 4):
                        receiver_name = candidate.title()

        if receiver_name:
            tags["receiver_name"] = receiver_name

        debit_reversal = bool(re.search(r"reversal|refund|return|chargeback|\bREV\b", text, flags=re.IGNORECASE))
        tags["is_debit_reversal"] = debit_reversal

        category_checks = [
            ("Salary / Income", ["salary", "sal/", "payroll", "pay/sal"]),
            ("Food & Dining", ["swiggy", "zomato", "food", "restaurant", "cafe", "dominos", "mcdonalds"]),
            ("Shopping / POS", ["netflix", "amazon", "flipkart", "myntra", "meesho", "ajio", "nykaa", "shopping"]),
            (
                "Utilities & Bills",
                ["electricity", "elec", "bescom", "bses", "tata power", "gas", "water", "broadband", "jio", "airtel", "vi ", "bsnl", "recharge", "bill", "utility"],
            ),
            ("EMI / Loan", ["emi", "loan", "lic", "insurance", "insur", "premium"]),
            ("Investments", ["mutual fund", "mf/", "sip", "zerodha", "groww", "kuvera", "nps", "ppf", "fd ", "fixed deposit", "stock", "demat"]),
            ("ATM / Cash", ["atm", "wdl", "cash", "atw"]),
            ("Transfers & P2P", ["upi", "neft", "rtgs", "imps", "transfer", "trf", "p2p"]),
        ]

        category = None
        for candidate_category, patterns in category_checks:
            if any(pattern in lower for pattern in patterns):
                category = candidate_category
                break
        if category is None and tags["payment_mode"] == "POS":
            category = "Shopping / POS"
        tags["category"] = category

        tags["is_salary"] = tags["category"] == "Salary / Income"
        tags["is_emi"] = tags["category"] == "EMI / Loan"

        return tags
    except Exception as exc:
        logger.warning("parse_description failed for '%s': %s", description, exc)
        return _default_description_tags()


def _new_parse_debug(pages_processed: int) -> dict[str, Any]:
    return {
        "skip_log": [],
        "pages_processed": pages_processed,
        "rows_attempted": 0,
        "rows_skipped": 0,
    }


def _find_header_row(lines: list[str]) -> tuple[int | None, dict[str, int]]:
    for index, line in enumerate(lines):
        lower = line.lower()
        tokens = _split_row(line)
        keyword_hits = sum(1 for keyword in HEADER_KEYWORDS if keyword in lower)
        if keyword_hits < 2:
            continue

        column_map: dict[str, int] = {}
        for token_index, token in enumerate(tokens):
            token_lower = token.lower()
            if "date" in token_lower or "txn" in token_lower or "value" in token_lower:
                column_map.setdefault("date", token_index)
            elif any(marker in token_lower for marker in ("narration", "description", "particular", "detail", "remark")):
                column_map.setdefault("description", token_index)
            elif any(marker in token_lower for marker in ("debit", "dr", "withdrawal", "debit amount")):
                column_map.setdefault("debit", token_index)
            elif any(marker in token_lower for marker in ("credit", "cr", "deposit", "credit amount")):
                column_map.setdefault("credit", token_index)
            elif any(marker in token_lower for marker in ("balance", "closing", "running balance")):
                column_map.setdefault("balance", token_index)
            elif any(marker in token_lower for marker in ("ref", "chq", "cheque", "reference")):
                column_map.setdefault("ref_no", token_index)
            elif any(marker in token_lower for marker in ("amount", "txn amount")):
                column_map.setdefault("amount", token_index)

        if len(column_map) >= 2:
            return index, column_map

    return None, {}


def _longest_text_token(tokens: list[str], excluded_indexes: set[int] | None = None) -> str:
    excluded_indexes = excluded_indexes or set()
    candidates = [token.strip() for index, token in enumerate(tokens) if index not in excluded_indexes]
    candidates = [token for token in candidates if token and not DATE_RE.fullmatch(token)]
    if not candidates:
        return ""
    return max(candidates, key=len)


def _parse_row(
    line: str,
    column_map: dict[str, int],
    previous_balance: float | None,
) -> tuple[dict[str, Any] | None, float | None, str | None]:
    try:
        tokens = _split_row(line)
        if not tokens:
            return None, previous_balance, "empty row"

        row_text = " ".join(tokens)
        date_match = DATE_RE.search(row_text)
        if not date_match:
            return None, previous_balance, "no valid date"

        date_value = _normalize_date(date_match.group(0))
        if not date_value:
            return None, previous_balance, "invalid date"

        date_index = None
        for index, token in enumerate(tokens):
            if DATE_RE.search(token):
                date_index = index
                break
        if date_index is None:
            date_index = 0

        ref_no = ""
        debit = 0.0
        credit = 0.0

        if not column_map:
            tail = line[date_match.end() :].strip().lstrip("|-").strip()
            amount_matches = list(
                re.finditer(r"-?\d[\d,]*\.?\d*\s*(?:CR|DR)?", tail, flags=re.IGNORECASE)
            )
            candidates: list[tuple[int, int, str, float, str | None]] = []
            for match in amount_matches:
                token = match.group(0).strip()
                if DATE_RE.fullmatch(token):
                    continue
                value, suffix = clean_amount(token)
                if value:
                    candidates.append((match.start(), match.end(), token, value, suffix))

            if not candidates:
                return None, previous_balance, "no amount found"

            if len(candidates) >= 2:
                amount_start, amount_end, amount_token, amount_value, amount_suffix = candidates[-2]
                balance_start, balance_end, balance_token, balance_value, _ = candidates[-1]
                description = tail[:amount_start].strip()
            else:
                amount_start, amount_end, amount_token, amount_value, amount_suffix = candidates[-1]
                balance_start, balance_end, balance_token, balance_value, _ = candidates[-1]
                description = tail[:amount_start].strip()

            description = re.sub(r"\|", " ", description)
            description = re.sub(r"\s+", " ", description).strip()
            if not description:
                description = _longest_text_token(tokens, excluded_indexes={date_index})
                description = re.sub(r"\s+", " ", description).strip()

            if not description:
                return None, previous_balance, "empty description"

            if amount_suffix == "DR":
                debit = amount_value
            elif amount_suffix == "CR":
                credit = amount_value
            elif amount_value >= 0 and len(candidates) >= 2:
                credit = amount_value if balance_value >= amount_value else 0.0

            if not debit and not credit and previous_balance is not None:
                delta = balance_value - previous_balance
                if delta > 0:
                    credit = abs(delta)
                elif delta < 0:
                    debit = abs(delta)

            if not any([debit, credit, balance_value]):
                return None, previous_balance, "zero amount and balance"

            transaction = {
                "date": date_value,
                "description": description,
                "debit": float(debit or 0.0),
                "credit": float(credit or 0.0),
                "balance": float(balance_value or 0.0),
                "ref_no": ref_no,
                "tags": {},
            }
            next_previous_balance = float(balance_value) if balance_value else previous_balance
            return transaction, next_previous_balance, None

        balance_value = 0.0
        balance_found = False

        amount_token = ""
        amount_value = 0.0
        amount_suffix: str | None = None

        if "balance" in column_map and column_map["balance"] < len(tokens):
            balance_value, _ = clean_amount(tokens[column_map["balance"]])
            balance_found = balance_value != 0.0

        if "ref_no" in column_map and column_map["ref_no"] < len(tokens):
            ref_no = str(tokens[column_map["ref_no"]]).strip()

        if "debit" in column_map and column_map["debit"] < len(tokens):
            debit_value, debit_suffix = clean_amount(tokens[column_map["debit"]])
            if debit_value:
                debit = debit_value
                amount_token = tokens[column_map["debit"]]
                amount_value = debit_value
                amount_suffix = debit_suffix

        if not debit and "credit" in column_map and column_map["credit"] < len(tokens):
            credit_value, credit_suffix = clean_amount(tokens[column_map["credit"]])
            if credit_value:
                credit = credit_value
                amount_token = tokens[column_map["credit"]]
                amount_value = credit_value
                amount_suffix = credit_suffix

        if not balance_found and "amount" in column_map and column_map["amount"] < len(tokens):
            combined_value, combined_suffix = clean_amount(tokens[column_map["amount"]])
            if combined_value:
                amount_token = tokens[column_map["amount"]]
                amount_value = combined_value
                amount_suffix = combined_suffix
                if combined_suffix == "DR":
                    debit = combined_value
                else:
                    credit = combined_value

        numeric_candidates: list[tuple[int, str, float, str | None]] = []
        for index, token in enumerate(tokens[date_index + 1 :], start=date_index + 1):
            if DATE_RE.fullmatch(token):
                continue
            value, suffix = clean_amount(token)
            if value:
                numeric_candidates.append((index, token, value, suffix))

        if not balance_found and numeric_candidates:
            _, _, balance_value, _ = numeric_candidates[-1]
            balance_found = True

        if not amount_value and numeric_candidates:
            if len(numeric_candidates) >= 2:
                _, amount_token, amount_value, amount_suffix = numeric_candidates[-2]
            else:
                _, amount_token, amount_value, amount_suffix = numeric_candidates[-1]

            if amount_suffix == "DR":
                debit = amount_value
            elif amount_suffix == "CR":
                credit = amount_value

        if not balance_found and previous_balance is not None:
            balance_value = previous_balance + credit - debit
            balance_found = balance_value != 0.0

        description_index = column_map.get("description")
        excluded_indexes = {date_index}
        if description_index is not None:
            desc_tokens = tokens[description_index : max(description_index + 1, len(tokens))]
            description = " ".join(desc_tokens).strip()
        else:
            amount_indexes = {
                column_map.get("debit"),
                column_map.get("credit"),
                column_map.get("balance"),
                column_map.get("ref_no"),
                column_map.get("amount"),
            }
            amount_indexes = {idx for idx in amount_indexes if idx is not None}
            if amount_indexes:
                first_amount_index = min(amount_indexes)
                description_tokens = tokens[date_index + 1 : first_amount_index]
            else:
                if numeric_candidates:
                    description_tokens = tokens[date_index + 1 : numeric_candidates[-2][0] if len(numeric_candidates) >= 2 else numeric_candidates[-1][0]]
                else:
                    description_tokens = tokens[date_index + 1 :]
            description = " ".join(description_tokens).strip()

        description = re.sub(r"\|", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
        if not description:
            description = _longest_text_token(tokens, excluded_indexes={date_index})
            description = re.sub(r"\s+", " ", description).strip()

        if not description:
            return None, previous_balance, "empty description"

        if not debit and not credit and previous_balance is not None and balance_found:
            delta = balance_value - previous_balance
            if delta > 0:
                credit = abs(delta)
            elif delta < 0:
                debit = abs(delta)

        if not balance_found and previous_balance is not None:
            balance_value = previous_balance + credit - debit
            balance_found = balance_value != 0.0

        if not any([debit, credit, balance_value]):
            return None, previous_balance, "zero amount and balance"

        transaction = {
            "date": date_value,
            "description": description,
            "debit": float(debit or 0.0),
            "credit": float(credit or 0.0),
            "balance": float(balance_value or 0.0),
            "ref_no": ref_no,
            "tags": {},
        }
        next_previous_balance = float(balance_value) if balance_found else previous_balance
        return transaction, next_previous_balance, None

    except Exception as exc:
        logger.warning("Skipping bad OCR row: %s | error=%s", line, exc)
        return None, previous_balance, str(exc)


def _parse_page(
    page_text: str,
    previous_balance: float | None,
    skip_log: list[dict[str, str]],
    debug: dict[str, Any],
) -> tuple[list[dict[str, Any]], float | None]:
    if not _has_numeric_content(page_text):
        return [], previous_balance

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    lines = merge_transaction_lines(lines)
    header_index, column_map = _find_header_row(lines)
    if header_index is None:
        header_index = -1
        column_map = {}

    rows: list[dict[str, Any]] = []
    current_balance = previous_balance
    for line in lines[header_index + 1 :]:
        if not _has_numeric_content(line):
            continue
        if sum(1 for keyword in HEADER_KEYWORDS if keyword in line.lower()) >= 2:
            continue
        debug["rows_attempted"] += 1
        transaction, current_balance, reason = _parse_row(line, column_map, current_balance)
        if transaction is None:
            debug["rows_skipped"] += 1
            skip_log.append({"row": line, "reason": reason or "unknown"})
            continue
        rows.append(transaction)

    return rows, current_balance


def parse_bank_pdf(pdf_path: str) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    previous_balance: float | None = None

    try:
        pages = extract_text_by_page(pdf_path)
    except Exception as exc:
        logger.warning("Failed to extract OCR text from %s: %s", pdf_path, exc)
        return []

    global LAST_PARSE_DEBUG
    skip_log: list[dict[str, str]] = []
    debug = _new_parse_debug(len(pages))

    for page_text in pages:
        try:
            page_raw_text = page_text.get("raw_text", "") if isinstance(page_text, dict) else str(page_text or "")
            page_transactions, previous_balance = _parse_page(page_raw_text, previous_balance, skip_log, debug)
            transactions.extend(page_transactions)
        except Exception as exc:
            logger.warning("Skipping page in %s due to error: %s", pdf_path, exc)
            continue

    debug["skip_log"] = skip_log
    LAST_PARSE_DEBUG.clear()
    LAST_PARSE_DEBUG.update(debug)

    for transaction in transactions:
        try:
            transaction["tags"] = parse_description(str(transaction.get("description", "") or ""))
        except Exception as exc:
            logger.warning("Failed to tag transaction '%s': %s", transaction.get("description", ""), exc)
            transaction["tags"] = _default_description_tags()

    deduped: list[dict[str, Any]] = []
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


def to_standard_transactions(raw_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in raw_list or []:
        try:
            if not isinstance(item, dict):
                continue

            date_value = _normalize_date(item.get("date", ""))
            description = re.sub(r"\s+", " ", str(item.get("description", "") or "").replace("|", " ")).strip()
            debit_value, _ = clean_amount(item.get("debit", 0.0))
            credit_value, _ = clean_amount(item.get("credit", 0.0))
            balance_value, _ = clean_amount(item.get("balance", 0.0))
            ref_no = str(item.get("ref_no", "") or "").strip()
            tags = item.get("tags", {}) if isinstance(item.get("tags", {}), dict) else {}
            if not tags:
                tags = parse_description(description)

            if not date_value or not description:
                continue
            if not any([debit_value, credit_value, balance_value]):
                continue

            cleaned.append(
                {
                    "date": date_value,
                    "description": description,
                    "debit": float(debit_value or 0.0),
                    "credit": float(credit_value or 0.0),
                    "balance": float(balance_value or 0.0),
                    "ref_no": ref_no,
                    "tags": tags,
                }
            )
        except Exception as exc:
            logger.warning("Skipping invalid transaction during cleanup: %s", exc)
            continue

    return cleaned