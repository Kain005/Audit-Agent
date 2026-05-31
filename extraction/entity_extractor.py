"""Hybrid entity extraction for invoices and financial documents."""

from __future__ import annotations

import json
import logging
import os
from pydoc import text
import re
from pathlib import Path
from typing import Any

import dateparser
import requests
import spacy
from dateparser.search import search_dates

from .models import InvoiceEntities, LineItem

LOGGER = logging.getLogger(__name__)

GST_REGEX = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}")
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
IFSC_REGEX = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")

# Common account numbers are usually 9 to 18 digits.
BANK_ACCOUNT_REGEX = re.compile(r"\b\d{9,18}\b")

AMOUNT_REGEX = re.compile(
    r"(?:₹|Rs\.?|INR)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(/-)?",
    flags=re.IGNORECASE,
)

LAKH_CRORE_REGEX = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*(lakh|lakhs|crore|crores)\b",
    flags=re.IGNORECASE,
)

DATE_TOKEN_REGEX = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b"
)

INDIAN_STATE_CODES = {
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "96",
    "97",
    "99",
}


class EntityExtractor:
    """Combines regex, spaCy, and LLM extraction for invoice entities."""

    def __init__(self) -> None:
        root_path = Path(__file__).parent.parent
        custom_model_path = root_path / "models" / "finance_ner_model"

        if custom_model_path.exists():
            try:
                self.nlp = spacy.load(str(custom_model_path))
                LOGGER.info("Using custom spaCy model at %s", custom_model_path)
            except Exception as exc:
                LOGGER.warning("Failed to load custom model (%s). Falling back: %s", custom_model_path, exc)
                self.nlp = self._load_default_spacy_model()
        else:
            self.nlp = self._load_default_spacy_model()

        self.ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    def _load_default_spacy_model(self) -> Any:
        try:
            model = spacy.load("en_core_web_sm")
            LOGGER.info("Using spaCy model: en_core_web_sm")
            return model
        except Exception as exc:
            LOGGER.warning("Failed to load en_core_web_sm. Using blank 'en' model: %s", exc)
            model = spacy.blank("en")
            LOGGER.info("Using spaCy model: blank_en")
            return model

    def extract_gst_number(self, text: str) -> list[str]:
        """Extract GSTIN values with de-duplication."""
        matches = [m.group(0) for m in GST_REGEX.finditer((text or "").upper())]
        unique = []
        for gst in matches:
            if self.validate_gst_number(gst) and gst not in unique:
                unique.append(gst)
        return unique

    def extract_pan_number(self, text: str) -> list[str]:
        """Extract PAN values with de-duplication."""
        matches = [m.group(0) for m in PAN_REGEX.finditer((text or "").upper())]
        return list(dict.fromkeys(matches))

    def extract_bank_account(self, text: str) -> list[str]:
        """Extract likely bank account numbers."""
        matches = [m.group(0) for m in BANK_ACCOUNT_REGEX.finditer(text or "")]
        return list(dict.fromkeys(matches))

    def extract_ifsc(self, text: str) -> list[str]:
        """Extract IFSC codes with de-duplication."""
        matches = [m.group(0) for m in IFSC_REGEX.finditer((text or "").upper())]
        return list(dict.fromkeys(matches))

    def extract_amounts(self, text: str) -> list[float]:
        """Extract rupee amounts including lakh/crore text formats."""
        source = text or ""
        amounts: list[float] = []

        for match in AMOUNT_REGEX.finditer(source):
            number_text = match.group(1)
            normalized = number_text.replace(",", "")
            try:
                amounts.append(float(normalized))
            except ValueError:
                continue

        for match in LAKH_CRORE_REGEX.finditer(source):
            base = float(match.group(1))
            unit = match.group(2).lower()
            if unit.startswith("lakh"):
                amounts.append(base * 100000)
            elif unit.startswith("crore"):
                amounts.append(base * 10000000)

        deduped: list[float] = []
        seen: set[float] = set()
        for value in amounts:
            rounded = round(value, 2)
            if rounded not in seen:
                seen.add(rounded)
                deduped.append(rounded)
        return deduped

    def extract_dates(self, text: str) -> list[str]:
        """Extract and normalize dates to YYYY-MM-DD."""
        source = text or ""
        normalized_dates: list[str] = []

        token_candidates = [m.group(0) for m in DATE_TOKEN_REGEX.finditer(source)]
        if token_candidates:
            for candidate in token_candidates:
                parsed = dateparser.parse(
                    candidate,
                    settings={"DATE_ORDER": "DMY", "PREFER_LOCALE_DATE_ORDER": False, "STRICT_PARSING": False},
                )
                if parsed:
                    normalized = parsed.strftime("%Y-%m-%d")
                    if normalized not in normalized_dates:
                        normalized_dates.append(normalized)

        searched = search_dates(
            source,
            settings={"DATE_ORDER": "DMY", "PREFER_LOCALE_DATE_ORDER": False, "STRICT_PARSING": False},
        )
        if searched:
            for _, parsed in searched:
                normalized = parsed.strftime("%Y-%m-%d")
                if normalized not in normalized_dates:
                    normalized_dates.append(normalized)

        return normalized_dates

    def validate_gst_number(self, gst: str) -> bool:
        """Validate GSTIN by format and Indian state code."""
        candidate = (gst or "").strip().upper()
        if not GST_REGEX.fullmatch(candidate):
            return False
        return candidate[:2] in INDIAN_STATE_CODES

    def extract_with_spacy(self, text: str) -> dict[str, list[str]]:
        """Run NER and return entities grouped by label."""
        doc = self.nlp(text or "")
        grouped: dict[str, list[str]] = {}
        for ent in doc.ents:
            grouped.setdefault(ent.label_, [])
            if ent.text not in grouped[ent.label_]:
                grouped[ent.label_].append(ent.text)
        return grouped

    def extract_with_llm(self, text: str, doc_type: str) -> dict[str, Any]:
        print(f"DEBUG extract_with_llm ENTRY", flush=True)
        print(f"DEBUG extract_with_llm ENTRY doc_type={doc_type} text_len={len(text or '')}", flush=True)
        """Call Ollama for context-aware extraction. Returns empty dict on failure."""
        truncated_text = text[:1500] if text else ""  # ~375 tokens, enough for header fields
        prompt = (
            "You are an information extraction assistant. "
            f"Document type: {doc_type}. "
            "Extract fields from the text and return ONLY strict JSON with keys: "
            "vendor_name (string or null), payment_terms (string or null), "
            "line_item_descriptions (array of strings).\n\n"
            "Text:\n"
            f"{truncated_text}"
        )

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                f"{self.ollama_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            llm_text = str(body.get("response", "")).strip()

            if not llm_text:
                return {}

            try:
                parsed = json.loads(llm_text)
                print(f"DEBUG LLM extraction result: {parsed}", flush=True)


                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                json_text = self._extract_json_object(llm_text)
                if not json_text:
                    return {}
                parsed = json.loads(json_text)
                print(f"DEBUG LLM extraction result: {parsed}", flush=True)  # ADD HERE

                return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            print(f"DEBUG Ollama extraction failed, continuing without LLM: {exc}", flush=True)
            return {}

    def extract_invoice_entities(self, text: str) -> InvoiceEntities:
        print(f"DEBUG extract_invoice_entities called: text_len={len(text or '')}", flush=True)
        """Run all extractors and merge output into typed invoice entities."""
        source = text or ""

        gst_numbers = self.extract_gst_number(source)
        pan_numbers = self.extract_pan_number(source)
        account_numbers = self.extract_bank_account(source)
        ifsc_codes = self.extract_ifsc(source)
        amounts = self.extract_amounts(source)
        dates = self.extract_dates(source)

        spacy_entities = self.extract_with_spacy(source)
        llm_entities = self.extract_with_llm(source, doc_type="invoice")

        vendor_name = self._first_non_empty(
            llm_entities.get("vendor_name"),
            self._first_from_labels(spacy_entities, ["ORG", "VENDOR", "COMPANY"]),
        )

        vendor_gst = gst_numbers[0] if gst_numbers else None
        vendor_address = self._first_from_labels(spacy_entities, ["ADDRESS", "LOC", "GPE"])

        invoice_number = self._extract_invoice_number(source)
        invoice_date = dates[0] if len(dates) >= 1 else None
        due_date = dates[1] if len(dates) >= 2 else None

        line_item_descriptions = llm_entities.get("line_item_descriptions")
        line_items = self._build_line_items(line_item_descriptions)

        subtotal, gst_amount, total_amount = self._infer_totals(amounts)
        payment_terms = self._first_non_empty(llm_entities.get("payment_terms"))

        # Preserve extra structured artifacts in raw text when patterns are found.
        if pan_numbers or account_numbers or ifsc_codes:
            augmented_text = (
                f"{source}\n\n"
                f"[EXTRACTED_META] PAN={pan_numbers}; ACCOUNT={account_numbers}; IFSC={ifsc_codes}"
            )
        else:
            augmented_text = source

        return InvoiceEntities(
            vendor_name=vendor_name,
            vendor_gst=vendor_gst,
            vendor_address=vendor_address,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            due_date=due_date,
            line_items=line_items,
            subtotal=subtotal,
            gst_amount=gst_amount,
            total_amount=total_amount,
            payment_terms=payment_terms,
            currency="INR",
            raw_text=augmented_text,
        )

    def _extract_json_object(self, value: str) -> str | None:
        start = value.find("{")
        end = value.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return value[start : end + 1]

    def _first_non_empty(self, *values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _first_from_labels(self, entities: dict[str, list[str]], labels: list[str]) -> str | None:
        for label in labels:
            candidates = entities.get(label, [])
            if candidates:
                candidate = str(candidates[0]).strip()
                if candidate:
                    return candidate
        return None

    def _extract_invoice_number(self, text: str) -> str | None:
        patterns = [
            r"(?i)invoice\s*(?:no|number|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
            r"(?i)bill\s*(?:no|number|#)\s*[:\-]?\s*([A-Z0-9\-/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _build_line_items(self, descriptions: Any) -> list[LineItem]:
        if not isinstance(descriptions, list):
            return []

        line_items: list[LineItem] = []
        for description in descriptions:
            if not isinstance(description, str) or not description.strip():
                continue
            line_items.append(
                LineItem(
                    description=description.strip(),
                    quantity=None,
                    unit_price=None,
                    amount=0.0,
                    gst_rate=None,
                )
            )
        return line_items

    def _infer_totals(self, amounts: list[float]) -> tuple[float | None, float | None, float | None]:
        if not amounts:
            return (None, None, None)

        sorted_amounts = sorted(amounts)
        total_amount = sorted_amounts[-1]
        subtotal = sorted_amounts[-2] if len(sorted_amounts) >= 2 else None

        gst_amount = None
        if subtotal is not None:
            diff = round(total_amount - subtotal, 2)
            if diff > 0:
                gst_amount = diff

        return (subtotal, gst_amount, total_amount)
