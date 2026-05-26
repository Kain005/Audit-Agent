"""PDF parsing utilities for text extraction and structured invoice extraction."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from datetime import datetime
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
from .table_extractor import extract_table_cells, segment_rows_by_density


logger = logging.getLogger(__name__)

DATE_ONLY_RE = re.compile(r"^\s*\d{2}[-/]\d{2}[-/]\d{2,4}\s*$")
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = easyocr.Reader(["en"], gpu=False)
    return _ocr_engine


def detect_pdf_type_and_scale(file_path: str) -> tuple[str, int]:
    """Classify a PDF as native text or image-based before rendering."""
    total_chars = 0
    with pdfplumber.open(str(Path(file_path))) as pdf:
        for page in pdf.pages[:3]:
            total_chars += len((page.extract_text() or "").strip())

    if total_chars > 100:
        return "native_text", 2
    return "image_pdf", 4


def extract_image_pdf_full(file_path: str) -> list[dict[str, Any]]:
    """Single OCR pass per page — returns raw_text, confidence, AND dataframe."""
    pages_output: list[dict[str, Any]] = []
    _, render_scale = detect_pdf_type_and_scale(file_path)
    render_scale = min(render_scale, 2)

    with pdfplumber.open(str(Path(file_path))) as pdf:
        pdfium_pdf = pdfium.PdfDocument(str(Path(file_path)))
        try:
            for index, _page in enumerate(pdf.pages):
                page_number = index + 1
                try:
                    pdfium_page = pdfium_pdf[index]
                    bgr_array = _render_page_bgr(pdfium_page, render_scale)
                    page_width = int(bgr_array.shape[1])
                    density_strips = segment_rows_by_density(bgr_array)

                    if not density_strips:
                        pages_output.append({
                            "page_number": page_number,
                            "raw_text": "",
                            "confidence": 0.0,
                            "low_confidence": True,
                            "dataframe": None,
                        })
                        continue

                    ocr = get_ocr_engine()

                    strip_ocr: list[list] = []
                    for strip in density_strips:
                        if strip.shape[0] > 200 and len(strip_ocr) < 8:
                            strip_ocr.append([])
                        else:
                            strip_ocr.append(ocr.readtext(strip))

                    all_texts = []
                    all_confidences = []
                    for ocr_result in strip_ocr:
                        for item in ocr_result:
                            try:
                                text = str(item[1] or "").strip()
                                conf = float(item[2]) if len(item) > 2 else 0.8
                                if text:
                                    all_texts.append(text)
                                    all_confidences.append(conf)
                            except Exception:
                                continue
                    raw_text = " ".join(all_texts)
                    mean_conf = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

                    header_index = None
                    column_ranges = []
                    for strip_index, ocr_result in enumerate(strip_ocr):
                        if strip_index > 12:
                            break
                        candidate_ranges = build_column_ranges_from_ocr(ocr_result, page_width)
                        if len(candidate_ranges) >= 3:
                            header_index = strip_index
                            column_ranges = candidate_ranges
                            break

                    dataframe = None
                    if header_index is not None:
                        row_dicts = []
                        for ocr_result in strip_ocr[header_index + 1:]:
                            date_matches = re.findall(
                                r"\d{2}-\d{2}-\d{2,4}",
                                " ".join(str(item[1] or "").strip() for item in ocr_result or []),
                            )

                            if len(date_matches) >= 4:
                                grouped_items: list[list] = []
                                sorted_items = []
                                for item in ocr_result or []:
                                    try:
                                        box = item[0]
                                        top_y = float(min(point[1] for point in box))
                                    except Exception:
                                        continue
                                    sorted_items.append((top_y, item))

                                sorted_items.sort(key=lambda item: item[0])

                                current_group: list = []
                                previous_y: float | None = None
                                for top_y, item in sorted_items:
                                    if previous_y is not None and top_y - previous_y > 15 and current_group:
                                        grouped_items.append(current_group)
                                        current_group = []
                                    current_group.append(item)
                                    previous_y = top_y

                                if current_group:
                                    grouped_items.append(current_group)

                                for grouped_ocr_result in grouped_items:
                                    row_dicts.append(_map_row_to_columns_from_ocr(grouped_ocr_result, column_ranges, page_width))
                            else:
                                row_dicts.append(_map_row_to_columns_from_ocr(ocr_result, column_ranges, page_width))
                        dataframe = pd.DataFrame(row_dicts) if row_dicts else None

                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": raw_text,
                        "confidence": mean_conf,
                        "low_confidence": mean_conf < 0.7,
                        "dataframe": dataframe,
                    })
                except Exception as exc:
                    logger.warning("extract_image_pdf_full failed page %s: %s", page_number, exc)
                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": "",
                        "confidence": 0.0,
                        "low_confidence": True,
                        "dataframe": None,
                    })
        finally:
            pdfium_pdf.close()

    return pages_output


def build_column_ranges(header_strip: np.ndarray, page_width: int) -> list[dict[str, int | str]]:
    """Detect column ranges from a header strip using EasyOCR header keywords."""
    if header_strip is None or getattr(header_strip, "size", 0) == 0 or page_width <= 0:
        return []

    keyword_map = {
        "date": "date",
        "dale": "date",
        "txn dale": "date",
        "value dale": "date",
        "value date": "value date",
        "narration": "narration",
        "narralion": "narration",
        "particulars": "particulars",
        "parliculars": "particulars",
        "description": "description",
        "descriplion": "description",
        "descripton": "description",
        "withdrawal": "withdrawal",
        "withdrawl": "withdrawl",
        "debit": "debit",
        "debil": "debit",
        "deposit": "deposit",
        "credit": "credit",
        "credil": "credit",
        "balance": "balance",
        "balonce": "balance",
        "balancc": "balance",
        "dr": "dr",
        "cr": "cr",
        "ref": "ref",
        "rel cheque": "ref",
        "rel": "ref",
        "chq": "chq",
        "cheque": "cheque",
    }

    def _char_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        left_length = len(left)
        right_length = len(right)
        if left_length != right_length:
            return abs(left_length - right_length) + sum(
                1 for index in range(min(left_length, right_length)) if left[index] != right[index]
            )
        return sum(1 for index in range(left_length) if left[index] != right[index])

    def _resolve_keyword(text: str) -> str | None:
        if text in keyword_map:
            return keyword_map[text]

        closest_name = None
        closest_distance = 3
        for keyword, canonical_name in keyword_map.items():
            distance = _char_distance(keyword, text)
            if distance <= 2 and distance < closest_distance:
                closest_name = canonical_name
                closest_distance = distance
        return closest_name

    ocr = get_ocr_engine()
    result = ocr.readtext(header_strip)

    detected_words: list[dict[str, int | str]] = []
    for item in result or []:
        try:
            box, text = item[0], str(item[1] or "").strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            left_x = int(round(min(point[0] for point in box)))
            right_x = int(round(max(point[0] for point in box)))
        except Exception:
            continue
        normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
        matched_keyword = _resolve_keyword(normalized_text)
        if matched_keyword:
            detected_words.append({"text": matched_keyword, "x_start": left_x, "x_end": right_x})

    detected_words.sort(key=lambda item: int(item["x_start"]))

    if len(detected_words) < 3:
        return []

    column_ranges: list[dict[str, int | str]] = []
    for index, word in enumerate(detected_words):
        next_x = int(detected_words[index + 1]["x_start"]) if index + 1 < len(detected_words) else page_width
        column_ranges.append({
            "name": str(word["text"]),
            "x_start": int(word["x_start"]),
            "x_end": int(next_x),
        })

    if len(column_ranges) < 3:
        return []

    return column_ranges


def map_row_to_columns(
    row_strip: np.ndarray,
    column_ranges: list[dict[str, int | str]],
    page_width: int,
) -> dict[str, str]:
    """Map OCR words in a row strip to detected column ranges."""
    if row_strip is None or getattr(row_strip, "size", 0) == 0 or not column_ranges or page_width <= 0:
        return {str(column.get("name", "")): "" for column in column_ranges}

    ocr = get_ocr_engine()
    result = ocr.readtext(row_strip)

    normalized_columns: list[dict[str, int | str | float]] = []
    for column in column_ranges:
        name = str(column.get("name", ""))
        x_start = int(column.get("x_start", 0) or 0)
        x_end = int(column.get("x_end", page_width) or page_width)
        midpoint = (x_start + x_end) / 2.0
        normalized_columns.append({"name": name, "x_start": x_start, "x_end": x_end, "midpoint": midpoint})

    if not normalized_columns:
        return {}

    assigned_words: dict[str, list[tuple[float, str]]] = {str(column["name"]): [] for column in normalized_columns}

    detected_words: list[tuple[float, float, str]] = []
    for item in result or []:
        try:
            box, text = item[0], str(item[1] or "").strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            left_x = float(min(point[0] for point in box))
            right_x = float(max(point[0] for point in box))
            center_x = (left_x + right_x) / 2.0
        except Exception:
            continue
        detected_words.append((left_x, center_x, text))

    detected_words.sort(key=lambda item: item[0])

    for _, center_x, text in detected_words:
        matched_index: int | None = None
        for index, column in enumerate(normalized_columns):
            x_start = float(column["x_start"])
            x_end = float(column["x_end"])
            if x_start <= center_x < x_end:
                matched_index = index
                break

        if matched_index is None:
            nearest_index = 0
            nearest_distance = abs(center_x - float(normalized_columns[0]["midpoint"]))
            for index, column in enumerate(normalized_columns[1:], start=1):
                distance = abs(center_x - float(column["midpoint"]))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = index
            matched_index = nearest_index

        column_name = str(normalized_columns[matched_index]["name"])
        assigned_words[column_name].append((center_x, text))

    return {
        str(column["name"]): " ".join(text for _, text in sorted(assigned_words[str(column["name"])] , key=lambda item: item[0])).strip()
        for column in normalized_columns
    }


def _serialize_row_dicts_as_tab_text(row_dicts: list[dict[str, str]], column_ranges: list[dict[str, int | str]]) -> str:
    column_names = [str(column.get("name", "")).strip() for column in column_ranges if str(column.get("name", "")).strip()]
    if not column_names:
        return ""

    lines = ["\t".join(column_names)]
    for row in row_dicts:
        lines.append("\t".join(str(row.get(column_name, "") or "").strip() for column_name in column_names))
    return "\n".join(lines).strip()


def validate_column_consistency(
    row_dicts: list[dict[str, str]],
    column_ranges: list[dict[str, int | str]],
) -> dict[str, float]:
    """Validate mapped rows for date, numeric, and balance consistency."""
    column_names = [str(column.get("name", "")).strip().lower() for column in column_ranges]

    date_values: list[str] = []
    numeric_values: list[str] = []
    balance_values: list[float] = []
    debit_column_names = {"debit", "withdrawal", "withdrawl"}
    credit_column_names = {"credit", "deposit"}
    numeric_column_names = debit_column_names | credit_column_names | {"balance"}

    for row in row_dicts:
        for column in column_names:
            value = str(row.get(column, "") or "").strip()
            if not value:
                continue
            if column == "date":
                date_values.append(value)
            if column in numeric_column_names:
                numeric_values.append(value)
                if column == "balance":
                    cleaned_balance = re.sub(r"[,$₹$€£]", "", value)
                    cleaned_balance = re.sub(r"\s+", "", cleaned_balance)
                    try:
                        balance_values.append(float(cleaned_balance))
                    except (TypeError, ValueError):
                        continue

    date_formats = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d %b %Y", "%Y-%m-%d"]
    successful_dates = 0
    for value in date_values:
        parsed = False
        for date_format in date_formats:
            try:
                datetime.strptime(value, date_format)
                parsed = True
                break
            except ValueError:
                continue
        if parsed:
            successful_dates += 1

    date_confidence = successful_dates / len(date_values) if date_values else 0.0

    successful_numeric = 0
    parsed_balances: list[float] = []
    for row in row_dicts:
        for column in column_names:
            if column not in numeric_column_names:
                continue
            value = str(row.get(column, "") or "").strip()
            if not value:
                continue
            cleaned_value = re.sub(r"[,$₹$€£]", "", value)
            cleaned_value = re.sub(r"\s+", "", cleaned_value)
            try:
                numeric_value = float(cleaned_value)
            except (TypeError, ValueError):
                continue
            successful_numeric += 1
            if column == "balance":
                parsed_balances.append(numeric_value)

    numeric_confidence = successful_numeric / len(numeric_values) if numeric_values else 0.0

    balance_confidence = 0.0
    if numeric_confidence > 0.7 and parsed_balances and len(parsed_balances) > 1:
        balance_column_name = next((name for name in column_names if name == "balance"), None)
        if balance_column_name is not None:
            checked_rows = 0
            reconciling_rows = 0
            mean_balance = float(np.mean(parsed_balances)) if parsed_balances else 0.0
            tolerance = abs(mean_balance) * 0.01
            previous_balance: float | None = None

            for row in row_dicts:
                balance_value = str(row.get(balance_column_name, "") or "").strip()
                if not balance_value:
                    continue

                cleaned_balance = re.sub(r"[,$₹$€£]", "", balance_value)
                cleaned_balance = re.sub(r"\s+", "", cleaned_balance)
                try:
                    current_balance = float(cleaned_balance)
                except (TypeError, ValueError):
                    continue

                debit_value = 0.0
                credit_value = 0.0
                for debit_column in debit_column_names:
                    cleaned_debit = re.sub(r"[,$₹$€£]", "", str(row.get(debit_column, "") or ""))
                    cleaned_debit = re.sub(r"\s+", "", cleaned_debit)
                    try:
                        debit_value += float(cleaned_debit) if cleaned_debit else 0.0
                    except (TypeError, ValueError):
                        continue
                for credit_column in credit_column_names:
                    cleaned_credit = re.sub(r"[,$₹$€£]", "", str(row.get(credit_column, "") or ""))
                    cleaned_credit = re.sub(r"\s+", "", cleaned_credit)
                    try:
                        credit_value += float(cleaned_credit) if cleaned_credit else 0.0
                    except (TypeError, ValueError):
                        continue

                checked_rows += 1
                if previous_balance is None:
                    previous_balance = current_balance
                    continue

                expected_balance = previous_balance - debit_value + credit_value
                if abs(current_balance - expected_balance) <= tolerance:
                    reconciling_rows += 1
                previous_balance = current_balance

            balance_confidence = reconciling_rows / checked_rows if checked_rows else 0.0

    overall_confidence = (
        date_confidence * 0.25
        + numeric_confidence * 0.25
        + balance_confidence * 0.5
    )

    return {
        "date_confidence": date_confidence,
        "numeric_confidence": numeric_confidence,
        "balance_confidence": balance_confidence,
        "overall_confidence": overall_confidence,
    }


def _render_page_bgr(pdfium_page: Any, render_scale: int) -> np.ndarray:
    bitmap = pdfium_page.render(scale=render_scale)
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
    pdf_type, render_scale = detect_pdf_type_and_scale(file_path)

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
                        "pdf_type": pdf_type,
                        "render_scale": render_scale,
                    })
                    continue

                try:
                    pdfium_page = pdfium_pdf[index]
                    bgr_array = _render_page_bgr(pdfium_page, render_scale)

                    ocr = get_ocr_engine()
                    table_rows = extract_table_cells(bgr_array)
                    raw_text = ""
                    mean_confidence = 0.0
                    confidence_validated = False
                    density_fallback_used = False

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
                    if raw_text and pdf_type == "image_pdf" and len(raw_text) < 100:
                        raw_text = ""
                        mean_confidence = 0.0

                    if not raw_text and pdf_type == "image_pdf":
                        density_strips = segment_rows_by_density(bgr_array)
                        if density_strips:
                            strip_texts: list[str] = []
                            strip_confidences: list[float] = []
                            for strip in density_strips:
                                cell_result = ocr.readtext(strip)
                                strip_text, strip_confidence = _read_easyocr_lines(cell_result)
                                if strip_text:
                                    strip_texts.append(strip_text)
                                if strip_confidence > 0:
                                    strip_confidences.append(strip_confidence)

                            raw_text = "\n".join(strip_texts).strip()
                            mean_confidence = round(sum(strip_confidences) / len(strip_confidences), 4) if strip_confidences else 0.0
                            density_fallback_used = True

                    if not raw_text:
                        result = ocr.readtext(bgr_array)
                        raw_text, mean_confidence = _read_easyocr_lines(result)

                    low_confidence = True if density_fallback_used else mean_confidence < 0.7
                    if not raw_text:
                        pages_output.append({
                            "page_number": page_number,
                            "raw_text": "",
                            "confidence": 0.0,
                            "low_confidence": True,
                            "source": "easyocr",
                            "pdf_type": pdf_type,
                            "render_scale": render_scale,
                        })
                        continue

                    page_output = {
                        "page_number": page_number,
                        "raw_text": raw_text,
                        "confidence": mean_confidence,
                        "low_confidence": low_confidence,
                        "source": "easyocr",
                        "pdf_type": pdf_type,
                        "render_scale": render_scale,
                    }
                    if confidence_validated:
                        page_output["confidence_validated"] = True
                    pages_output.append(page_output)

                except Exception as exc:
                    logger.warning("EasyOCR failed for page %s in %s: %s", page_number, file_path, exc)
                    pages_output.append({
                        "page_number": page_number,
                        "raw_text": "",
                        "confidence": 0.0,
                        "low_confidence": True,
                        "source": "easyocr",
                        "pdf_type": pdf_type,
                        "render_scale": render_scale,
                    })
        finally:
            pdfium_pdf.close()

    return pages_output


def _map_row_to_columns_from_ocr(
    ocr_result: list,
    column_ranges: list[dict[str, int | str]],
    page_width: int,
) -> dict[str, str]:
    """Map pre-computed OCR result to column ranges without re-running OCR."""
    if not ocr_result or not column_ranges or page_width <= 0:
        return {str(col.get("name", "")): "" for col in column_ranges}

    normalized_columns = []
    for col in column_ranges:
        name = str(col.get("name", ""))
        x_start = int(col.get("x_start", 0) or 0)
        x_end = int(col.get("x_end", page_width) or page_width)
        midpoint = (x_start + x_end) / 2.0
        normalized_columns.append({"name": name, "x_start": x_start, "x_end": x_end, "midpoint": midpoint})

    assigned_words: dict[str, list[tuple[float, str]]] = {str(col["name"]): [] for col in normalized_columns}

    detected_words = []
    for item in ocr_result or []:
        try:
            box, text = item[0], str(item[1] or "").strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            left_x = float(min(p[0] for p in box))
            right_x = float(max(p[0] for p in box))
            center_x = (left_x + right_x) / 2.0
        except Exception:
            continue
        detected_words.append((left_x, center_x, text))

    detected_words.sort(key=lambda item: item[0])

    for _, center_x, text in detected_words:
        matched_index = None
        for idx, col in enumerate(normalized_columns):
            if float(col["x_start"]) <= center_x < float(col["x_end"]):
                matched_index = idx
                break
        if matched_index is None:
            nearest_index = 0
            nearest_dist = abs(center_x - float(normalized_columns[0]["midpoint"]))
            for idx, col in enumerate(normalized_columns[1:], start=1):
                d = abs(center_x - float(col["midpoint"]))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_index = idx
            matched_index = nearest_index
        col_name = str(normalized_columns[matched_index]["name"])
        assigned_words[col_name].append((center_x, text))

    return {
        str(col["name"]): " ".join(t for _, t in sorted(assigned_words[str(col["name"])], key=lambda x: x[0])).strip()
        for col in normalized_columns
    }

def build_column_ranges_from_ocr(
    ocr_result: list,
    page_width: int,
) -> list[dict[str, int | str]]:
    """Build column ranges from pre-computed OCR result."""
    if not ocr_result or page_width <= 0:
        return []

    keyword_map = {
        "date": "date", "txn date": "date", "transaction date": "date",
        "value date": "date", "posting date": "date",
        "dale": "date", "txn dale": "date", "value dale": "date",
        "narration": "description", "particulars": "description",
        "description": "description", "details": "description",
        "remarks": "description", "txn remarks": "description",
        "descriplion": "description", "descripton": "description",
        "narralion": "description", "parliculars": "description",
        "withdrawal": "debit", "withdrawl": "debit",
        "debit": "debit", "debit amount": "debit", "dr": "debit",
        "debil": "debit",
        "deposit": "credit", "credit": "credit",
        "credit amount": "credit", "cr": "credit", "credil": "credit",
        "balance": "balance", "closing": "balance", "running": "balance",
        "balonce": "balance", "balancc": "balance",
        "ref": "ref", "chq": "ref", "cheque": "ref",
        "reference": "ref", "chq/ref": "ref", "ref no": "ref",
        "cheque no": "ref", "rel cheque": "ref", "rel": "ref",
        "amount": "amount", "txn amount": "amount",
    }

    def _edit_distance(a: str, b: str) -> int:
        if len(a) != len(b):
            return abs(len(a) - len(b)) + sum(c1 != c2 for c1, c2 in zip(a, b))
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    detected_words = []
    for item in ocr_result or []:
        try:
            box, text = item[0], str(item[1] or "").strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            left_x = int(round(min(p[0] for p in box)))
            right_x = int(round(max(p[0] for p in box)))
        except Exception:
            continue
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        matched = keyword_map.get(normalized)
        if matched is None:
            for key in keyword_map:
                if _edit_distance(normalized, key) <= 2:
                    matched = keyword_map[key]
                    break
        if matched:
            detected_words.append({"text": matched, "x_start": left_x, "x_end": right_x})

    detected_words.sort(key=lambda item: int(item["x_start"]))
    if len(detected_words) < 3:
        return []

    column_ranges = []
    for index, word in enumerate(detected_words):
        next_x = int(detected_words[index + 1]["x_start"]) if index + 1 < len(detected_words) else page_width
        column_ranges.append({"name": str(word["text"]), "x_start": int(word["x_start"]), "x_end": int(next_x)})

    return column_ranges if len(column_ranges) >= 3 else []


def _map_data_region_to_rows(
    ocr_result: list,
    column_ranges: list[dict[str, int | str]],
    page_width: int,
    data_strips: list,
    y_offset: int,
) -> list[dict[str, str]]:
    """Map a single OCR pass over the full data region into per-row dicts."""
    if not ocr_result or not column_ranges:
        return []

    # Build row boundaries from strip heights
    row_boundaries: list[tuple[int, int]] = []
    y = 0
    for strip in data_strips:
        h = strip.shape[0]
        row_boundaries.append((y, y + h))
        y += h

    if not row_boundaries:
        return []

    # Assign each OCR word to a row and column
    col_names = [str(c["name"]) for c in column_ranges]
    rows: list[dict[str, list[tuple[float, str]]]] = [
        {name: [] for name in col_names} for _ in row_boundaries
    ]

    normalized_columns = []
    for col in column_ranges:
        x_start = int(col.get("x_start", 0) or 0)
        x_end = int(col.get("x_end", page_width) or page_width)
        normalized_columns.append({
            "name": str(col["name"]),
            "x_start": x_start,
            "x_end": x_end,
            "midpoint": (x_start + x_end) / 2.0,
        })

    for item in ocr_result or []:
        try:
            box, text = item[0], str(item[1] or "").strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            left_x = float(min(p[0] for p in box))
            right_x = float(max(p[0] for p in box))
            top_y = float(min(p[1] for p in box))
            center_x = (left_x + right_x) / 2.0
            center_y = top_y
        except Exception:
            continue

        # Find row
        row_index = None
        for ri, (ry_start, ry_end) in enumerate(row_boundaries):
            if ry_start <= center_y < ry_end:
                row_index = ri
                break
        if row_index is None:
            # Assign to nearest row
            row_index = min(
                range(len(row_boundaries)),
                key=lambda i: abs(center_y - (row_boundaries[i][0] + row_boundaries[i][1]) / 2)
            )

        # Find column
        col_index = None
        for ci, col in enumerate(normalized_columns):
            if col["x_start"] <= center_x < col["x_end"]:
                col_index = ci
                break
        if col_index is None:
            col_index = min(
                range(len(normalized_columns)),
                key=lambda i: abs(center_x - normalized_columns[i]["midpoint"])
            )

        col_name = normalized_columns[col_index]["name"]
        rows[row_index][col_name].append((center_x, text))

    return [
        {
            name: " ".join(t for _, t in sorted(rows[ri][name], key=lambda x: x[0])).strip()
            for name in col_names
        }
        for ri in range(len(row_boundaries))
    ]



def extract_image_pdf_as_dataframe(file_path: str) -> list[dict[str, Any] | None]:
    """Extract image-PDF pages as column-mapped DataFrames using density strips."""
    pages_output: list[dict[str, Any] | None] = []
    _, render_scale = detect_pdf_type_and_scale(file_path)
    render_scale = min(render_scale, 2)


    with pdfplumber.open(str(Path(file_path))) as pdf:
        pdfium_pdf = pdfium.PdfDocument(str(Path(file_path)))
        try:
            for index, _page in enumerate(pdf.pages):
                page_number = index + 1
                try:
                    pdfium_page = pdfium_pdf[index]
                    bgr_array = _render_page_bgr(pdfium_page, render_scale)
                    density_strips = segment_rows_by_density(bgr_array)
                    if not density_strips:
                        pages_output.append(None)
                        continue

                    page_width = int(bgr_array.shape[1])
                    ocr = get_ocr_engine()
                    header_index = None
                    column_ranges = []
                    cached_ocr: list[list] = []

                    from concurrent.futures import ThreadPoolExecutor as _TPE

                    def _ocr_strip(args):
                        idx, strip, skip = args
                        if skip:
                            return idx, []
                        return idx, ocr.readtext(strip)

                    skip_flags = [
                        (header_index is None and s.shape[0] > 200)
                        for s in density_strips
                    ]
                    # We don't know header_index yet so conservatively skip strips >200px in first 12
                    skip_flags = [
                        (i <= 12 and density_strips[i].shape[0] > 200)
                        for i in range(len(density_strips))
                    ]

                    with _TPE(max_workers=4) as pool:
                        results = list(pool.map(
                            _ocr_strip,
                            [(i, density_strips[i], skip_flags[i]) for i in range(len(density_strips))]
                        ))

                    results.sort(key=lambda x: x[0])
                    cached_ocr = [r[1] for r in results]

                    for strip_index, ocr_result in enumerate(cached_ocr):
                        if header_index is None and strip_index <= 12:
                            candidate_ranges = build_column_ranges_from_ocr(ocr_result, page_width)
                            if len(candidate_ranges) >= 3:
                                header_index = strip_index
                                column_ranges = candidate_ranges

                    if header_index is None:
                        pages_output.append(None)
                        continue

                    row_dicts: list[dict[str, str]] = []
                    for strip_index, ocr_result in enumerate(cached_ocr[header_index + 1:]):
                        row_dicts.append(_map_row_to_columns_from_ocr(ocr_result, column_ranges, page_width))

                    dataframe = pd.DataFrame(row_dicts)
                    pages_output.append({
                        "page_number": page_number,
                        "dataframe": dataframe,
                        "column_ranges": column_ranges,
                    })
                except Exception:
                    pages_output.append(None)
        finally:
            pdfium_pdf.close()

    return pages_output


def extract_native_pdf_as_dataframe(file_path: str) -> list[dict[str, Any]]:
    """Extract native-text PDF pages as normalized table DataFrames."""
    pages_output: list[dict[str, Any]] = []

    def _normalize_header_cell(cell: Any) -> str:
        normalized_cell = str(cell or "").strip().lower()
        if not normalized_cell:
            return ""
        if "withdrawal" in normalized_cell or "withdrawl" in normalized_cell:
            return "debit"
        if "deposit" in normalized_cell:
            return "credit"
        if "narration" in normalized_cell or "particulars" in normalized_cell or "description" in normalized_cell:
            return "description"
        if "balance" in normalized_cell:
            return "balance"
        if "date" in normalized_cell:
            return "date"
        if "ref" in normalized_cell or "chq" in normalized_cell:
            return "ref_no"
        return normalized_cell

    with pdfplumber.open(str(Path(file_path))) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_frames: list[pd.DataFrame] = []
            raw_tables = page.extract_tables() or []

            for table in raw_tables:
                if not table or len(table) < 2:
                    continue

                header_row = table[0]
                normalized_headers = [_normalize_header_cell(cell) for cell in header_row]
                table_rows = table[1:]

                if not any(normalized_headers):
                    continue

                df = pd.DataFrame(table_rows, columns=normalized_headers)
                df = df.loc[:, [column for column in df.columns if str(column).strip()]]
                if not df.empty:
                    page_frames.append(df)

            if page_frames:
                page_dataframe = pd.concat(page_frames, ignore_index=True, sort=False)
            else:
                page_dataframe = pd.DataFrame()

            pages_output.append({
                "page_number": page_number,
                "dataframe": page_dataframe,
            })

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
