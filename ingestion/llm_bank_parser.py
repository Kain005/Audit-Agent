"""Generic LLM-based bank statement parser for image PDFs.

Ollama replaced with Gemini 3.1 Flash Lite.
Architecture: OCR all pages locally -> single Gemini API call for full document.
Fallback: positional heuristic parser when GEMINI_API_KEY is not set.
"""
from __future__ import annotations
import json
import logging
import os
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite"

BANK_PROMPT = """You are a bank statement data extractor.

Extract every transaction from the bank statement text below.
Return ONLY a JSON array. No markdown, no explanation, no extra keys.
Each element must have exactly these keys:
  date        — transaction date string as it appears in the source
  description — narration / payee / remarks
  debit       — amount debited, numeric, 0.0 if none
  credit      — amount credited, numeric, 0.0 if none
  balance     — closing balance after transaction, numeric, 0.0 if not shown

Rules:
- Include every transaction row. Do not skip any.
- If a row has no date but has an amount, inherit the date from the previous row.
- Remove comma separators from numbers (1,234.56 -> 1234.56).
- Use empty string for description only if truly absent.
- Output nothing except the JSON array.

Bank statement text:
{raw_text}"""


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, api_key: str) -> str:
    from google import genai  # noqa: PLC0415
    import httpx

    print(f"[Gemini] Sending request to model: {GEMINI_MODEL}")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0},
        )
        print(f"[Gemini] Response received — {len(response.text.strip())} chars")
        return response.text.strip()
    except Exception as exc:
        status = None
        if hasattr(exc, "status_code"):
            status = exc.status_code
        elif hasattr(exc, "code"):
            status = exc.code
        elif hasattr(exc, "response") and hasattr(exc.response, "status_code"):
            status = exc.response.status_code
        if status:
            print(f"[Gemini] HTTP Error {status} — {exc}")
        else:
            print(f"[Gemini] Error — {type(exc).__name__}: {exc}")
        raise


# ---------------------------------------------------------------------------
# JSON cleaning + row parsing  (unchanged from v3)
# ---------------------------------------------------------------------------

def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_llm_rows(raw_response: str, bank_name: str) -> list[dict]:
    cleaned = _clean_json(raw_response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not array_match:
            logger.warning("LLM bank parser: no JSON array found in response")
            return []
        try:
            parsed = json.loads(array_match.group(0))
        except json.JSONDecodeError:
            logger.warning("LLM bank parser: JSON array parse failed after extraction")
            return []

    if not isinstance(parsed, list):
        return []

    rows = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        date_raw = str(item.get("date", "") or "")
        description = str(item.get("description", "") or "").strip()
        if not date_raw.strip() and not description:
            continue
        try:
            debit = float(str(item.get("debit", 0) or 0).replace(",", ""))
        except (ValueError, TypeError):
            debit = 0.0
        try:
            credit = float(str(item.get("credit", 0) or 0).replace(",", ""))
        except (ValueError, TypeError):
            credit = 0.0
        try:
            balance = float(str(item.get("balance", 0) or 0).replace(",", ""))
        except (ValueError, TypeError):
            balance = 0.0
        rows.append({
            "date": pd.to_datetime(date_raw, dayfirst=True, errors="coerce"),
            "description": description,
            "debit": debit,
            "credit": credit,
            "balance": balance,
            "transaction_id": "",
            "bank_name": bank_name,
        })
    return rows


# ---------------------------------------------------------------------------
# OCR helpers  (unchanged from v3)
# ---------------------------------------------------------------------------

def _ocr_page_to_text(page_bgr, ocr_engine) -> str:
    results = ocr_engine.readtext(page_bgr)
    tokens = []
    current_y = None
    line_tokens = []
    for item in sorted(results, key=lambda x: min(p[1] for p in x[0])):
        try:
            top_y = min(p[1] for p in item[0])
            text = str(item[1] or "").strip()
            if not text:
                continue
            if current_y is None:
                current_y = top_y
            if abs(top_y - current_y) > 12 and line_tokens:
                tokens.append(" ".join(line_tokens))
                line_tokens = []
                current_y = top_y
            line_tokens.append(text)
        except Exception:
            continue
    if line_tokens:
        tokens.append(" ".join(line_tokens))
    return "\n".join(tokens)


# ---------------------------------------------------------------------------
# Positional heuristic fallback (no API key)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\b")


def _heuristic_parse(full_text: str, bank_name: str) -> list[dict]:
    """Best-effort positional parser used when Gemini key is absent."""
    rows = []
    last_date = None
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        date_match = _DATE_RE.search(line)
        if date_match:
            last_date = date_match.group(0)
        if not last_date:
            continue
        amounts = [float(a.replace(",", "")) for a in _AMOUNT_RE.findall(line)]
        if not amounts:
            continue
        # Heuristic: last amount = balance, second-last = debit or credit
        balance = amounts[-1] if len(amounts) >= 1 else 0.0
        mid = amounts[-2] if len(amounts) >= 2 else 0.0
        desc_part = _DATE_RE.sub("", line)
        desc_part = _AMOUNT_RE.sub("", desc_part).strip(" |-_/")
        rows.append({
            "date": pd.to_datetime(last_date, dayfirst=True, errors="coerce"),
            "description": desc_part,
            "debit": mid,
            "credit": 0.0,
            "balance": balance,
            "transaction_id": "",
            "bank_name": bank_name,
        })
    return rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_image_pdf_with_llm(
    file_path: str,
    bank_name: str = "generic",
) -> pd.DataFrame:
    import pypdfium2 as pdfium  # noqa: PLC0415
    from ingestion.pdf_parser import _render_page_bgr, detect_pdf_type_and_scale, get_ocr_engine  # noqa: PLC0415

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    _, render_scale = detect_pdf_type_and_scale(file_path)
    render_scale = min(render_scale, 2)

    ocr = get_ocr_engine()
    pdf = pdfium.PdfDocument(file_path)
    page_texts: list[str] = []

    # --- Phase 1: OCR every page locally ---
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bgr = _render_page_bgr(page, render_scale)
            page_text = _ocr_page_to_text(bgr, ocr)
            if page_text.strip():
                page_texts.append(f"--- PAGE {index + 1} ---\n{page_text}")
                logger.info("LLM bank parser: OCR page %d — %d chars", index + 1, len(page_text))
            else:
                logger.info("LLM bank parser: page %d empty OCR, skipping", index + 1)
    finally:
        pdf.close()

    if not page_texts:
        logger.warning("LLM bank parser: no OCR text extracted from %s", file_path)
        return pd.DataFrame()

    full_text = "\n\n".join(page_texts)
    logger.info("LLM bank parser: total OCR text length %d chars across %d pages", len(full_text), len(page_texts))

    # --- Phase 2: single LLM call or heuristic fallback ---
    if not api_key:
        logger.warning(
            "LLM bank parser: GEMINI_API_KEY not set — falling back to positional heuristic. "
            "Set the env variable for full accuracy."
        )
        all_rows = _heuristic_parse(full_text, bank_name)
    else:
        prompt = BANK_PROMPT.format(raw_text=full_text)
        logger.info("LLM bank parser: sending %d-char prompt to Gemini (%s)", len(prompt), GEMINI_MODEL)
        try:
            raw_response = _call_gemini(prompt, api_key)
            logger.info("LLM bank parser: Gemini response length %d chars", len(raw_response))
            all_rows = _parse_llm_rows(raw_response, bank_name)
            logger.info("LLM bank parser: parsed %d rows from Gemini response", len(all_rows))
        except Exception as exc:
            print(f"[Gemini] Call failed — falling back to heuristic. Reason: {exc}")
            logger.error("LLM bank parser: Gemini call failed: %s — falling back to heuristic", exc)
            all_rows = _heuristic_parse(full_text, bank_name)

    if not all_rows:
        logger.warning("LLM bank parser: zero rows extracted from %s", file_path)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["date"])
    df = df[df["date"].notna()]
    df = df.drop_duplicates(subset=["date", "description", "debit", "credit", "balance"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("LLM bank parser: final DataFrame %d rows for %s", len(df), file_path)
    return df