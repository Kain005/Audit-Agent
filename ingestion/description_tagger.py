from __future__ import annotations

import re
from typing import Dict, Optional


PAYMENT_MODE_KEYWORDS = [
    (r"\bUPI\b", "UPI"),
    (r"\bNEFT\b", "NEFT"),
    (r"\bRTGS\b", "RTGS"),
    (r"\bIMPS\b", "IMPS"),
    (r"\bATM\b", "ATM"),
    (r"\bCASH\b", "ATM"),
    (r"\bCLG\b|\bCHEQUE\b|\bCHQ\b", "CLG"),
    (r"\bPOS\b|\bPURCHASE\b", "POS"),
    (r"\bEMI\b", "EMI"),
    (r"\bSI/|\bSTANDING\b", "SI"),
    (r"\bACH\b|AUTOMATED\s+CLEARING", "ACH"),
]

CATEGORY_KEYWORDS_PRIORITY = [
    ("salary", [r"\bSALARY\b", r"\bSAL/\b", r"\bPAYROLL\b"]),
    ("refund", [r"\bREFUND\b", r"\bREVERSAL\b", r"\bREV/\b"]),
    ("emi", [r"\bEMI\b", r"\bLOAN\b", r"\bLOANEMI\b", r"\bACH\b"]),
    ("utilities", [r"\bELECTRICITY\b", r"\bWATER\b", r"\bGAS\b", r"\bBESCOM\b", r"\bMSEDCL\b", r"\bBSES\b", r"\bTATA POWER\b"]),
    ("food_delivery", [r"\bSWIGGY\b", r"\bZOMATO\b", r"\bDUNZO\b"]),
    ("ecommerce", [r"\bAMAZON\b", r"\bFLIPKART\b", r"\bMYNTRA\b", r"\bMEESHO\b", r"\bSNAPDEAL\b"]),
    ("travel", [r"\bIRCTC\b", r"\bMAKEMYTRIP\b", r"\bREDBUS\b", r"\bUBER\b", r"\bOLA\b", r"\bRAPIDO\b"]),
    ("fuel", [r"\bPETROL\b", r"\bDIESEL\b", r"\bHPCL\b", r"\bBPCL\b", r"\bIOCL\b", r"\bINDIANOIL\b"]),
    ("insurance", [r"\bLIC\b", r"\bINSURANCE\b", r"\bINSUR\b"]),
    ("tax", [r"\bTDS\b", r"\bGST\b", r"\bINCOMETAX\b", r"\bINCOME\s+TAX\b"]),
    ("cash", [r"\bATM\b", r"\bCASH WITHDRAWAL\b", r"\bCASH DEPOSIT\b"]),
    ("internal_transfer", [r"\bSELF\b", r"\bOWN ACCOUNT\b", r"\bSWEEP\b"]),
]


def _first_regex_match(patterns, text: str) -> Optional[str]:
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None


def _extract_transaction_id(text: str) -> Optional[str]:
    # Look for UTR, REF, RRN, or 12+ digit numbers
    patterns = [r"\bUTR[:\-\s]*([A-Za-z0-9-]+)", r"\bREF[:\-\s]*([A-Za-z0-9-]+)", r"\bRRN[:\-\s]*([A-Za-z0-9-]+)", r"\b(\d{12,})\b"]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: look for tokens like UPI12345
    m2 = re.search(r"\b(UTR[A-Za-z0-9-]+|RRN[A-Za-z0-9-]+|REF[A-Za-z0-9-]+)\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None


def _fallback_receiver(text: str, exclude_keywords: Optional[set] = None) -> Optional[str]:
    if not text:
        return None
    words = re.findall(r"[A-Za-z&().\-]{4,}", text)
    if not words:
        return None
    exclude_keywords = exclude_keywords or set()
    # pick the longest sensible token that's not a keyword
    candidates = [w for w in words if w.upper() not in exclude_keywords]
    if not candidates:
        return None
    return max(candidates, key=len)


def tag_description(description: str, bank: str = "unknown") -> Dict[str, Optional[object]]:
    try:
        desc = (description or "").strip()
        text = desc.upper()

        # initialize
        payment_mode = None
        direction = None
        receiver = None
        transaction_id = None
        category = None
        is_merchant = False
        is_salary = False
        is_cash = False
        is_emi = False
        is_refund = False
        is_internal = False

        # payment mode detection
        for pat, mode in PAYMENT_MODE_KEYWORDS:
            if re.search(pat, text, re.IGNORECASE):
                payment_mode = mode
                break

        # direction detection
        if re.search(r"\bDR\b|\bDEBIT\b", text):
            direction = "DR"
        elif re.search(r"\bCR\b|\bCREDIT\b", text):
            direction = "CR"

        # category detection (priority)
        for cat_name, pats in CATEGORY_KEYWORDS_PRIORITY:
            for pat in pats:
                if re.search(pat, text, re.IGNORECASE):
                    category = cat_name
                    break
            if category:
                break

        # flags from category
        if category == "salary":
            is_salary = True
        if category == "cash":
            is_cash = True
        if category == "emi":
            is_emi = True
        if category == "refund":
            is_refund = True
        if category == "internal_transfer":
            is_internal = True

        # is_merchant heuristic
        if category in {"food_delivery", "ecommerce", "travel", "fuel"} or payment_mode == "POS":
            is_merchant = True

        # transaction id
        transaction_id = _extract_transaction_id(text)

        # receiver extraction logic
        try:
            if payment_mode == "UPI":
                parts = desc.split("/")
                # take index 3 if present, else index 2
                if len(parts) >= 4:
                    receiver = parts[3].strip()
                elif len(parts) >= 3:
                    receiver = parts[2].strip()
            elif payment_mode in {"NEFT", "RTGS", "IMPS"}:
                parts = desc.split("/")
                if len(parts) >= 3:
                    receiver = parts[2].strip()
            elif payment_mode == "POS":
                # POS/MERCHANT NAME/CITY — take everything after first "/"
                if "/" in desc:
                    after = desc.split("/", 1)[1]
                    # strip trailing numbers (dates/ref) heuristically
                    receiver = re.sub(r"\b\d{2,}\b.*$", "", after).strip()
                else:
                    receiver = None
            elif payment_mode == "ATM":
                receiver = None
            else:
                # generic fallback: try to pull from slash-separated tokens
                parts = desc.split("/")
                if len(parts) >= 3:
                    # take middle part that looks like a name
                    candidate = parts[1].strip()
                    if not re.fullmatch(r"\d+", candidate):
                        receiver = candidate

            # final fallback receiver
            if not receiver:
                exclude = {k for k, _ in PAYMENT_MODE_KEYWORDS}
                # include common tokens to exclude
                exclude.update({"DR", "CR", "DEBIT", "CREDIT", "REF", "UTR", "RRN"})
                receiver = _fallback_receiver(desc, exclude_keywords=exclude)
        except Exception:
            receiver = None

        return {
            "payment_mode": payment_mode,
            "direction": direction,
            "receiver": receiver,
            "transaction_id": transaction_id,
            "category": category or "other",
            "is_merchant": bool(is_merchant),
            "is_salary": bool(is_salary),
            "is_cash": bool(is_cash),
            "is_emi": bool(is_emi),
            "is_refund": bool(is_refund),
            "is_internal": bool(is_internal),
        }
    except Exception:
        return {
            "payment_mode": None,
            "direction": None,
            "receiver": None,
            "transaction_id": None,
            "category": None,
            "is_merchant": False,
            "is_salary": False,
            "is_cash": False,
            "is_emi": False,
            "is_refund": False,
            "is_internal": False,
        }
