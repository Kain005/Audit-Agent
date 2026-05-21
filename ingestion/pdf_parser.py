"""PDF parsing utilities for text extraction and structured invoice extraction."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
from pdf2image import convert_from_path
from PIL import Image, ImageOps
import easyocr
import pytesseract

from .parsers.normalizers import clean_amount, clean_date
from .table_extractor import extract_table_cells


logger = logging.getLogger(__name__)

DATE_ONLY_RE = re.compile(r"^\s*\d{2}[-/]\d{2}[-/]\d{2,4}\s*$")
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = easyocr.Reader(["en"], gpu=False)
    return _ocr_engine


def _render_page_bgr(pdfium_page: Any) -> np.ndarray:
    bitmap = pdfium_page.render(scale=2)
    rgba_array = np.array(bitmap.to_pil().convert("RGBA"))
    return cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)


def _read_easyocr_lines(result: Any) -> tuple[str, float]:
    items: list[tuple[float, float, str, float]] = []

    for item in result or []:
        try:
            box, text, score = item[0], item[1], item[2]
        except Exception:
            continue
        cleaned_text = str(text or "").strip()
        if not cleaned_text:
            continue
        try:
            x_coord = float(box[0][0])
            y_coord = float(box[0][1])
        except Exception:
            x_coord = 0.0
            y_coord = 0.0
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        items.append((y_coord, x_coord, cleaned_text, score_value))

    if not items:
        return "", 0.0

    items.sort(key=lambda item: (item[0], item[1]))

    lines: list[list[tuple[float, str, float]]] = []
    current_line: list[tuple[float, str, float]] = []
    current_y: float | None = None

    for y_coord, x_coord, text, score in items:
        if current_y is None or abs(y_coord - current_y) <= 15:
            current_line.append((x_coord, text, score))
            current_y = y_coord if current_y is None else current_y
        else:
            if current_line:
                current_line.sort(key=lambda item: item[0])
                lines.append(current_line)
            current_line = [(x_coord, text, score)]
            current_y = y_coord

    if current_line:
        current_line.sort(key=lambda item: item[0])
        lines.append(current_line)

    raw_text = "\n".join(" ".join(text for _, text, _ in line) for line in lines).strip()
    scores = [score for line in lines for _, _, score in line]
    confidence = sum(scores) / len(scores) if scores else 0.0
    return raw_text, confidence

def _ocr_page_image(image: Any, config: str) -> str:
    return (pytesseract.image_to_string(image, lang="eng", config=config) or "").strip()


def _ocr_page_confidence(image: Any, config: str) -> float:
    try:
        data = pytesseract.image_to_data(image, lang="eng", config=config, output_type=pytesseract.Output.DICT)
        confidences = []
        for value in data.get("conf", []):
            try:
                confidence = float(value)
            except (TypeError, ValueError):
                continue
            if confidence >= 0:
                confidences.append(confidence)
        if not confidences:
            return 0.0
        return round(sum(confidences) / len(confidences), 2)
    except Exception:
        return 0.0


def _date_only_line_count(text: str) -> int:
    return sum(1 for line in str(text or "").splitlines() if DATE_ONLY_RE.fullmatch(line.strip()))


def _prepare_ocr_image(page: Any, scale: int) -> Any:
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("L")
    return ImageOps.autocontrast(image.point(lambda pixel: 255 if pixel > 180 else 0))


def extract_text_by_page(file_path: str) -> list[dict[str, Any]]:
    pages_output: list[dict[str, Any]] = []

    with pdfplumber.open(str(Path(file_path))) as pdf:
        if len(pdf.pages) > 10:
            raise ValueError("PDF exceeds 10 pages")

        pdfium_pdf = pdfium.PdfDocument(str(Path(file_path)))
        try:
            for index, page in enumerate(pdf.pages):
                page_number = index + 1
                page_text = (page.extract_text() or "").strip()

                if page_text and len(page_text) > 50:
                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": page_text,
                        "confidence": 100,
                        "low_confidence": False,
                        "source": "pdfplumber",
                    })
                    continue

                try:
                    pdfium_page = pdfium_pdf[index]
                    bgr_array = _render_page_bgr(pdfium_page)

                    ocr = get_ocr_engine()
                    table_rows = extract_table_cells(bgr_array)

                    if table_rows:
                        row_texts: list[str] = []
                        confidences: list[float] = []
                        for row in table_rows:
                            cell_texts: list[str] = []
                            for crop in row:
                                cell_result = ocr.readtext(crop)
                                cell_text, cell_confidence = _read_easyocr_lines(cell_result)
                                if cell_text:
                                    cell_texts.append(cell_text)
                                if cell_confidence > 0:
                                    confidences.append(cell_confidence)
                            if cell_texts:
                                row_texts.append("\t".join(cell_texts))

                        raw_text = "\n".join(row_texts).strip()
                        mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
                    else:
                        result = ocr.readtext(bgr_array)
                        raw_text, mean_confidence = _read_easyocr_lines(result)

                    if not raw_text:
                        pages_output.append({
                            "page_number": page_number,
                            "raw_text": "",
                            "confidence": 0.0,
                            "low_confidence": True,
                            "source": "easyocr",
                        })
                        continue

                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": raw_text,
                        "confidence": mean_confidence,
                        "low_confidence": mean_confidence < 0.7,
                        "source": "easyocr",
                    })

                except Exception as exc:
                    logger.warning("EasyOCR failed for page %s in %s: %s", page_number, file_path, exc)
                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": "",
                        "confidence": 0.0,
                        "low_confidence": True,
                        "source": "easyocr",
                    })
        finally:
            pdfium_pdf.close()

    return pages_output


class PDFParser:
    """PDF parser for extracting text, tables, and invoice fields."""

    def __init__(self) -> None:
        self.tesseract_path = os.environ.get(
            "TESSERACT_PATH",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        except Exception:
            # OCR remains optional; text PDFs will still parse.
            pass

    def parse(self, file_path: str) -> dict[str, Any]:
        """Extract text and tabular data from PDF, using OCR when needed."""
        path = Path(file_path)
        all_text: list[str] = []
        all_tables: list[pd.DataFrame] = []

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    page_text = self._ocr_page(page)
                if page_text:
                    all_text.append(page_text)

                page_tables = page.extract_tables() or []
                for table in page_tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(cell).strip() if cell is not None else "" for cell in table[0]]
                    rows = table[1:]
                    all_tables.append(pd.DataFrame(rows, columns=header))

        data = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
        raw_text = "\n".join(all_text).strip()

        if len(raw_text) < 50:
            try:
                pages = convert_from_path(
                    str(path),
                    poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin",
                )
                ocr_text = "\n".join(
                    (pytesseract.image_to_string(page) or "").strip()
                    for page in pages
                ).strip()
                if ocr_text:
                    raw_text = ocr_text
            except Exception:
                pass

        return {"data": data, "raw_text": raw_text}

    def _ocr_page(self, page: Any) -> str:
        try:
            import pytesseract

            page_image = page.to_image(resolution=300).original
            return (pytesseract.image_to_string(page_image, lang="eng") or "").strip()
        except Exception:
            return ""

    def extract_invoice_data(self, file_path: str) -> dict[str, Any]:
        """Extract structured fields from Indian invoice PDFs."""
        parsed = self.parse(file_path)
        raw_text = str(parsed.get("raw_text") or "")
        table_df = parsed.get("data", pd.DataFrame())
        lines = [line.strip() for line in raw_text.splitlines() if line and line.strip()]

        invoice_number = self._extract_invoice_number(raw_text)
        invoice_date = self._extract_invoice_date(raw_text)
        vendor_name = self._extract_vendor_name(raw_text, lines)
        total_amount = self._extract_total_amount(raw_text)
        vendor_gst = self._extract_gst_number(raw_text)
        gst_amount = self._extract_gst_amount(raw_text)
        line_items = self._extract_line_items(table_df)

        confidence = 0.0
        if invoice_number:
            confidence += 0.2
        if total_amount is not None:
            confidence += 0.2
        if vendor_name:
            confidence += 0.2
        if invoice_date:
            confidence += 0.2
        if vendor_gst:
            confidence += 0.2

        warnings: list[str] = []
        if confidence < 0.4:
            warnings.append("Low confidence — manual review recommended")
        if not vendor_gst:
            warnings.append("No GST number found")

        return {
            "invoice_number": invoice_number,
            "vendor_name": vendor_name,
            "vendor_gst": vendor_gst,
            "invoice_date": invoice_date,
            "total_amount": total_amount,
            "gst_amount": gst_amount,
            "line_items": line_items,
            "raw_text": raw_text,
            "confidence": round(confidence, 2),
            "warnings": warnings,
        }

    def _extract_invoice_number(self, text: str) -> str | None:
        patterns = [
            r"Invoice No[.:]?\s*([A-Z0-9/-]+)",
            r"Invoice #\s*([A-Z0-9/-]+)",
            r"Bill No[.:]?\s*([A-Z0-9/-]+)",
            r"INV[-/]([A-Z0-9-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1).strip()
            if pattern.startswith(r"INV[-/]"):
                return f"INV-{value}"
            return value
        return None

    def _extract_invoice_date(self, text: str) -> str | None:
        date_near_patterns = [
            r"Invoice Date[\s:.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"Date[\s:.-]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"Invoice Date[\s:.-]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"Date[\s:.-]*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"Invoice Date[\s:.-]*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})",
            r"Date[\s:.-]*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})",
        ]

        for pattern in date_near_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            cleaned = clean_date(match.group(1).strip())
            if cleaned:
                return cleaned
        return None

    def _extract_vendor_name(self, text: str, lines: list[str]) -> str | None:
        for pattern in [r"From:\s*(.+)", r"Vendor:\s*(.+)"]:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    return candidate

        for line in lines:
            if line:
                return line
        return None

    def _extract_total_amount(self, text: str) -> float | None:
        amount_patterns = [
            r"Total[:\s]+(?:\u20B9\s*)?([\d,]+\.?\d*)",
            r"Grand Total[:\s]+(?:\u20B9\s*)?([\d,]+\.?\d*)",
            r"Amount Due[:\s]+(?:\u20B9\s*)?([\d,]+\.?\d*)",
            r"Net Payable[:\s]+(?:\u20B9\s*)?([\d,]+\.?\d*)",
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = clean_amount(match.group(1).strip())
            if value != 0.0:
                return value
        return None

    def _extract_gst_number(self, text: str) -> str | None:
        pattern = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}"
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def _extract_gst_amount(self, text: str) -> float | None:
        gst_components = 0.0
        patterns = [
            r"CGST[^\n\r]*?(?:\u20B9\s*)?([\d,]+\.?\d*)",
            r"SGST[^\n\r]*?(?:\u20B9\s*)?([\d,]+\.?\d*)",
            r"IGST[^\n\r]*?(?:\u20B9\s*)?([\d,]+\.?\d*)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                gst_components += clean_amount(match.group(1).strip())

        if gst_components == 0.0:
            return None
        return gst_components

    def _extract_line_items(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        if data.empty:
            return []

        records = data.fillna("").to_dict(orient="records")
        line_items: list[dict[str, Any]] = []
        for record in records:
            clean_record = {str(k).strip(): v for k, v in record.items()}
            if any(str(v).strip() for v in clean_record.values()):
                line_items.append(clean_record)
        return line_items
