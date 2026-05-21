from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from ..description_tagger import tag_description


class SBILayoutParser:
    """Parser for SBI bank statement layouts.

    Expects pages as a list of dicts with at least `page_number`, `raw_text`, `confidence`, `low_confidence`.
    """

    # Allow OCR O/0 confusion in day/year positions
    DATE_RE = re.compile(r"^\s*([0-9O]{2}\s+[A-Za-z]{3}\s+[0-9O]{4})")
    AMT_RE = re.compile(r"[0-9OBo,]+\.?\d*")
    ACCOUNT_RE = re.compile(r"Account\s*(?:No|Number|No\.|A/c)[:\s]*([A-Za-z0-9-]+)", re.IGNORECASE)

    @classmethod
    def detect(cls, pages: List[Dict[str, Any]]) -> bool:
        """Return True if any page looks like an SBI statement."""
        for p in pages:
            raw = str(p.get("raw_text", "") or "")
            if re.search(r"STATE\s+BANK\s+OF\s+INDIA|\bSBI\b", raw, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _parse_amount(cls, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s.strip()
        if not s or s in ["-", "--"]:
            return None
        # cleanup common OCR noise: O->0, B->8
        s = s.replace("O", "0").replace("o", "0").replace("B", "8")
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return None

    @classmethod
    def _cleanup_date_token(cls, token: str) -> str:
        # token expected like '15 Jan 2024' or with OCR '1O Jan 2O24'
        parts = token.split()
        if len(parts) >= 3:
            day = parts[0].replace("O", "0").replace("o", "0")
            month = parts[1]
            year = parts[2].replace("O", "0").replace("o", "0")
            return f"{day} {month} {year}"
        return token.replace("O", "0").replace("o", "0")

    @classmethod
    def _clean_line(cls, line: str) -> str:
        # replace pipe with space delimiter and normalize whitespace
        ln = line.replace("|", " ")
        ln = re.sub(r"[\t ]+", " ", ln).strip()
        return ln

    @classmethod
    def extract(cls, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse transactions from SBI-styled statement pages.

        Returns dict with keys: bank, account_number, transactions, parse_warnings, partial
        """
        transactions: List[Dict[str, Any]] = []
        parse_warnings: List[str] = []
        account_number: Optional[str] = None
        partial = False

        last_tx: Optional[Dict[str, Any]] = None

        for p in pages:
            page_no = int(p.get("page_number", 0) or 0)
            raw = str(p.get("raw_text") or "")
            conf = float(p.get("confidence") or 0.0)
            low_conf = bool(p.get("low_confidence"))

            if low_conf:
                parse_warnings.append(f"Page {page_no}: low OCR confidence ({conf:.1f}%), results may be inaccurate")
                partial = True

            if account_number is None:
                raw_for_acc = raw.replace("O", "0").replace("o", "0")
                macc = cls.ACCOUNT_RE.search(raw_for_acc)
                if macc:
                    account_number = macc.group(1).replace(" ", "").strip()

            lines = [ln.strip() for ln in raw.splitlines() if ln and ln.strip()]

            # Detect SBI format variant for this page by inspecting header row
            sbi_format = "B"  # default to compact
            header_found = False
            ref_found = False
            for ln in lines[:10]:
                low = ln.lower()
                if "ref" in low or "cheque" in low or "chq" in low:
                    ref_found = True
                if "description" in low or "txn" in low or "date" in low or "transaction" in low:
                    header_found = True

            if ref_found:
                sbi_format = "A"
            elif header_found:
                sbi_format = "B"
            else:
                # ambiguous -> default to B and warn
                parse_warnings.append(f"Page {page_no}: defaulted to Format B")
                sbi_format = "B"


            for ln in lines:
                ln_clean = cls._clean_line(ln)
                if ln_clean.lower().startswith("opening balance") or ln_clean.lower().startswith("closing balance"):
                    continue

                date_match = cls.DATE_RE.match(ln_clean)
                if date_match:
                    raw_date = date_match.group(1)
                    date_str = cls._cleanup_date_token(raw_date)
                    rest = ln_clean[date_match.end():].strip()

                    amt_tokens = re.findall(r"[0-9OBo,]+\.?\d*", ln_clean)

                    debit = None
                    credit = None
                    balance = None
                    ref_no = None

                    if amt_tokens:
                        parsed_amts = [cls._parse_amount(t) for t in amt_tokens if cls._parse_amount(t) is not None]
                        if parsed_amts:
                            # For both formats, balance is last
                            if len(parsed_amts) >= 1:
                                balance = parsed_amts[-1]
                            # Format A expected to have Debit, Credit before balance
                            if sbi_format == "A":
                                if len(parsed_amts) >= 3:
                                    debit = parsed_amts[-3]
                                    credit = parsed_amts[-2]
                                elif len(parsed_amts) == 2:
                                    debit = parsed_amts[-2]
                            else:
                                # Format B: may have Debit, Credit before balance or only Debit/Balance
                                if len(parsed_amts) >= 3:
                                    debit = parsed_amts[-3]
                                    credit = parsed_amts[-2]
                                elif len(parsed_amts) == 2:
                                    # ambiguous: treat first as debit
                                    debit = parsed_amts[0]

                    # Remove amount substrings from rest to isolate description and possible ref
                    desc = rest
                    for amt in re.findall(r"[0-9OBo,]+\.?\d*", rest):
                        desc = desc.rsplit(amt, 1)[0].strip()

                    # If Format A, try to extract ref_no as the first token of desc if it looks like a reference
                    if sbi_format == "A" and desc:
                        parts = desc.split()
                        if parts:
                            candidate = parts[0]
                            if re.match(r"^[A-Za-z0-9/.-]{2,}$", candidate):
                                ref_no = candidate
                                # remove candidate from description
                                desc = " ".join(parts[1:]).strip()

                    if not desc:
                        desc = ""

                    tx = {
                        "date": date_str,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": balance,
                        "ref_no": ref_no,
                        "source_page": page_no,
                        "sbi_format_variant": sbi_format,
                    }

                    # add parsed tags for description
                    try:
                        tx["tags"] = tag_description(tx["description"], bank="SBI")
                    except Exception:
                        tx["tags"] = None

                    if tx["description"] == "" and tx["debit"] is None and tx["credit"] is None and tx["balance"] is None:
                        parse_warnings.append(f"Page {page_no}: could not parse transaction line: '{ln}'")
                        last_tx = None
                        partial = True
                        continue

                    transactions.append(tx)
                    last_tx = tx
                else:
                    # continuation line
                    if last_tx is not None:
                        last_tx["description"] = (last_tx.get("description", "") + " " + ln).strip()
                    else:
                        parse_warnings.append(f"Page {page_no}: orphan continuation line skipped: '{ln}'")
                        partial = True

        result = {
            "bank": "SBI",
            "account_number": account_number,
            "transactions": transactions,
            "parse_warnings": parse_warnings,
            "partial": partial,
        }

        return result
