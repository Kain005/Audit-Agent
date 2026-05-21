"""Robust parsing normalizers for Indian financial amounts and dates."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_amount(value: Any) -> float:
    """
    Parse various Indian and international amount formats to a clean float.
    
    Handles:
    - Indian format: 1,00,000.00 (1 lakh)
    - European format: 1.00.000 (1 million)
    - Brackets for negative: (50,000) → -50000
    - Dr/Cr suffixes: 50,000 Dr → -50000, 50,000 Cr → 50000
    - Currency symbols: ₹, Rs., INR
    - Indian words: lakh (100k), crore (10M)
    - Already numeric values
    
    Args:
        value: Any value (string, float, int, None, NaN)
    
    Returns:
        float: Cleaned amount (0.0 for empty/None/invalid)
    """
    # Preserve real float values exactly as-is.
    if isinstance(value, float):
        if pd.isna(value):
            return 0.0
        return value

    # Handle None, NaN-like, and empty inputs.
    if value is None or pd.isna(value):
        return 0.0
    
    text = str(value).strip()
    if not text or text.lower() in {"", "nan", "none", "-"}:
        return 0.0
    
    # Detect if negative from brackets or suffix
    is_negative = False
    
    # Check for brackets (Indian accounting format)
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]
    
    # Check for Dr/Cr suffix
    text_lower = text.lower()
    if text_lower.endswith(" dr"):
        is_negative = True
        text = text[:-3]
    elif text_lower.endswith("dr"):
        is_negative = True
        text = text[:-2]
    elif text_lower.endswith(" cr"):
        is_negative = False
        text = text[:-3]
    elif text_lower.endswith("cr"):
        is_negative = False
        text = text[:-2]

    text = text.strip()
    
    # Remove currency symbols
    text = text.replace("₹", "")
    text = re.sub(r"\b(?:rs\.?|inr)\b", "", text, flags=re.IGNORECASE)
    text = text.replace("/-", "")
    text = text.replace("/", "")
    text = text.strip()
    
    # Handle Indian words
    text_lower = text.lower()
    if "crore" in text_lower:
        text = text_lower.replace("crore", "").strip()
        try:
            num = float(text.replace(",", "")) * 10_000_000
            return -num if is_negative else num
        except (ValueError, TypeError):
            return 0.0
    
    if "lakh" in text_lower:
        text = text_lower.replace("lakh", "").strip()
        try:
            num = float(text.replace(",", "")) * 100_000
            return -num if is_negative else num
        except (ValueError, TypeError):
            return 0.0
    
    # Handle Indian-style separators: 1,00,000 or 1.00.000
    # Count separators to detect format
    comma_count = text.count(",")
    dot_count = text.count(".")
    
    if comma_count >= 2:
        # Indian format: 1,00,000.00 → remove commas, keep last dot
        text = text.replace(",", "")
    elif dot_count >= 2 and comma_count == 0:
        # European format: 1.000.000 → remove dots, add commas
        # But we need to be careful: last dot might be decimal
        parts = text.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1]) == 2:  # Last part is 2 digits = decimal
            text = parts[0].replace(".", "") + "." + parts[1]
        else:  # All dots are thousand separators
            text = text.replace(".", "")
    
    # Convert trailing minus notation (e.g. 50000-) to leading minus.
    if text.endswith("-") and text.count("-") == 1:
        text = f"-{text[:-1]}"

    # Clean remaining non-numeric characters (except minus and decimal)
    text = re.sub(r"[^\d.\-]", "", text)
    
    # Parse to float
    try:
        num = float(text) if text and text not in {"", ".", "-"} else 0.0
        return -num if is_negative else num
    except (ValueError, TypeError):
        return 0.0


def clean_date(value: Any) -> str | None:
    """
    Parse various date formats to ISO format (YYYY-MM-DD).
    
    Handles:
    - Indian format: 15/03/2024 (DD/MM/YYYY)
    - Dashes: 15-03-2024 (DD-MM-YYYY)
    - Words: 15 Mar 2024, 15 March 2024, Mar 15, 2024
    - ISO: 2024-03-15 (already formatted)
    - Short year: 15/3/24
    - All variants with dayfirst=True
    
    Args:
        value: Any date-like value
    
    Returns:
        str: ISO date format (YYYY-MM-DD) or None if unparseable
    """
    # Handle None, NaN, empty
    if value is None or pd.isna(value):
        return None
    
    text = str(value).strip()
    if not text or text.lower() in {"", "nan", "none"}:
        return None
    
    # Fast-path for strict ISO date format.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
            return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
        except Exception:
            return None

    try:
        # Use dayfirst=True for all non-ISO inputs.
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        pass
    
    # Fallback: manual parsing for edge cases
    # Remove extra spaces, normalize
    text = re.sub(r"\s+", " ", text).strip()
    
    # Try common Indian formats manually
    patterns = [
        (r"(\d{1,2})[/\-.\s]+(\d{1,2})[/\-.\s]+(\d{4})", "%d/%m/%Y"),  # DD/MM/YYYY or DD-MM-YYYY
        (r"(\d{4})[/\-.\s]+(\d{1,2})[/\-.\s]+(\d{1,2})", "%Y/%m/%d"),   # YYYY/MM/DD
        (r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", "%d %B %Y"),         # DD Mon YYYY
        (r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", "%B %d %Y"),       # Mon DD, YYYY
    ]
    
    for pattern, date_format in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                date_str = " ".join(match.groups())
                parsed = pd.to_datetime(date_str, format=date_format, dayfirst=True, errors="coerce")
                if not pd.isna(parsed):
                    return parsed.strftime("%Y-%m-%d")
            except Exception:
                continue
    
    # If all else fails, return None
    return None
