from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..description_tagger import tag_description


class HDFCLayoutParser:
    """Parser for HDFC bank statement layouts with OCR-noise handling."""

    # Date regex accounting for O/0 OCR confusion: [0O][0-9O]\/[0-9O]{2}\/[0-9O]{2,4}
    DATE_RE = re.compile(r"^\s*([0O][0-9O]/[0-9O]{2}/[0-9O]{2,4})")
    AMOUNT_TOKEN_RE = re.compile(r"[0-9O,B,.,,]+")
    ACCOUNT_RE = re.compile(r"Account\s*No\.?[:\s]*([A-Za-z0-9O-]+)", re.IGNORECASE)

    @classmethod
    def detect(cls, pages: List[Dict[str, Any]]) -> bool:
        for p in pages:
            raw = str(p.get("raw_text", "") or "")
            if re.search(r"\bHDFC\b", raw, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _cleanup_token_for_date(cls, token: str) -> str:
        # Replace O with 0 in date-like tokens
        return token.replace("O", "0").replace("o", "0")

    @classmethod
    def _cleanup_token_for_amount(cls, token: str) -> str:
        # Replace OCR noise in amount tokens: O->0, B->8, remove non-numeric except . and , and -
        t = token.replace("O", "0").replace("o", "0").replace("B", "8")
        # keep digits, comma, dot, minus
        t = re.sub(r"[^0-9,\.\-]", "", t)
        return t

    @classmethod
    def _parse_amount(cls, token: str) -> Optional[float]:
        if not token:
            return None
        t = cls._cleanup_token_for_amount(token)
        t = t.replace(",", "")
        try:
            return float(t)
        except Exception:
            return None

    @classmethod
    def _clean_line(cls, line: str) -> str:
        # Replace pipe with tab for consistent splitting
        line = line.replace("|", "\t")
        # Normalize multiple spaces/tabs
        line = re.sub(r"[\t ]+", " ", line).strip()
        return line

    @classmethod
    def extract(cls, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        transactions: List[Dict[str, Any]] = []
        parse_warnings: List[str] = []
        account_number: Optional[str] = None
        partial = False

        prev_balance: Optional[float] = None
        last_tx: Optional[Dict[str, Any]] = None

        # Try account number on first two pages
        for p in pages[:2]:
            raw = str(p.get("raw_text") or "")
            # simple OCR cleanup for account extraction
            raw_clean = raw.replace("O", "0").replace("o", "0")
            m = cls.ACCOUNT_RE.search(raw_clean)
            if m:
                account_number = m.group(1).replace(" ", "").strip()
                break

        for p in pages:
            page_no = int(p.get("page_number", 0) or 0)
            raw = str(p.get("raw_text") or "")
            conf = float(p.get("confidence") or 0.0)
            low_conf = bool(p.get("low_confidence"))

            if low_conf:
                parse_warnings.append(f"Page {page_no}: low OCR confidence ({conf:.1f}%), results may be inaccurate")
                partial = True

            lines = [ln for ln in raw.splitlines()]
            for raw_ln in lines:
                ln = cls._clean_line(raw_ln)

                # attempt to find date at start (allow O/0 noise)
                mdate = cls.DATE_RE.match(ln)
                if mdate:
                    date_token = mdate.group(1)
                    # normalize date token O->0
                    date_token_clean = cls._cleanup_token_for_date(date_token)

                    rest = ln[mdate.end():].strip()

                    # Find amount-like tokens in the line
                    amt_tokens = re.findall(r"[0-9OBo,\.\-]+", ln)
                    numeric_tokens: List[float] = []
                    for t in amt_tokens:
                        val = cls._parse_amount(t)
                        if val is not None:
                            numeric_tokens.append(val)

                    balance = numeric_tokens[-1] if numeric_tokens else None

                    # The transaction amount is the numeric token before the balance (if present)
                    amount = None
                    if len(numeric_tokens) >= 2:
                        amount = numeric_tokens[-2]
                    elif len(numeric_tokens) == 1:
                        # only balance present; amount unknown
                        amount = None

                    # description: rest with trailing numeric tokens removed
                    desc = rest
                    for amt in re.findall(r"[0-9OBo,\.\-]+", rest):
                        desc = desc.rsplit(amt, 1)[0].strip()

                    if not desc:
                        desc = ""

                    # decide debit vs credit using previous balance when possible
                    debit = None
                    credit = None
                    if balance is not None:
                        if prev_balance is not None:
                            try:
                                if balance < prev_balance:
                                    # balance decreased -> debit
                                    debit = amount if amount is not None else round(prev_balance - balance, 2)
                                else:
                                    credit = amount if amount is not None else round(balance - prev_balance, 2)
                            except Exception:
                                # fallback: cannot decide
                                parse_warnings.append(f"Page {page_no}: could not determine debit/credit for line: '{raw_ln}'")
                                partial = True
                        else:
                            # no previous balance to compare against; cannot decide reliably
                            if amount is None:
                                parse_warnings.append(f"Page {page_no}: transaction amount missing on line: '{raw_ln}'")
                                partial = True

                    tx: Dict[str, Any] = {
                        "date": date_token_clean,
                        "description": desc,
                        "debit": float(debit) if debit is not None else None,
                        "credit": float(credit) if credit is not None else None,
                        "balance": float(balance) if balance is not None else None,
                        "source_page": page_no,
                    }

                    # add tags using description_tagger (best-effort)
                    try:
                        tx["tags"] = tag_description(tx["description"], bank="HDFC")
                    except Exception:
                        tx["tags"] = None

                    transactions.append(tx)
                    last_tx = tx
                    if balance is not None:
                        prev_balance = balance
                else:
                    # continuation line -> append to previous transaction description
                    if last_tx is not None:
                        cont = raw_ln.strip()
                        if cont:
                            last_tx["description"] = (last_tx.get("description", "") + " " + cont).strip()
                    else:
                        # orphan continuation
                        parse_warnings.append(f"Page {page_no}: orphan continuation line skipped: '{raw_ln}'")
                        partial = True

        # warnings: set partial True if any warnings
        if parse_warnings:
            partial = True

        return {
            "bank": "HDFC",
            "account_number": account_number,
            "transactions": transactions,
            "parse_warnings": parse_warnings,
            "partial": partial,
        }
