from __future__ import annotations

import cv2
import numpy as np


ROW_GROUP_TOLERANCE_PX = 10
# Minimum cell width as a fraction of page width (8%)
MIN_CELL_WIDTH_RATIO = 0.08
MIN_CELL_HEIGHT_PX = 10
MAX_CELL_WIDTH_RATIO = 0.5
MAX_CELL_HEIGHT_RATIO = 0.5


def extract_table_cells(bgr_array: np.ndarray) -> list[list[np.ndarray]]:
    if bgr_array is None or getattr(bgr_array, "size", 0) == 0:
        return []

    page_height, page_width = bgr_array.shape[:2]
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY) if bgr_array.ndim == 3 else bgr_array.copy()
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(threshold, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cells: list[tuple[int, int, int, int]] = []
    max_width = int(page_width * MAX_CELL_WIDTH_RATIO)
    max_height = int(page_height * MAX_CELL_HEIGHT_RATIO)
    min_width = int(page_width * MIN_CELL_WIDTH_RATIO)

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        # filter by relative minimum width to ignore thin noise contours
        if width < min_width or height < MIN_CELL_HEIGHT_PX:
            continue
        if width > max_width or height > max_height:
            continue
        detected_cells.append((x, y, width, height))

    if not detected_cells:
        return []

    # Sort by top coordinate
    detected_cells.sort(key=lambda rect: rect[1])

    # Group rectangles into rows by overlapping vertical intervals
    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = []
    current_row_max_y: int | None = None

    for cell in detected_cells:
        x, y, width, height = cell
        y0 = y
        y1 = y + height

        if current_row_max_y is None:
            current_row = [cell]
            current_row_max_y = y1
            continue

        # If current rect overlaps vertically with the current row band (or is within tolerance), add it
        if y0 <= current_row_max_y + ROW_GROUP_TOLERANCE_PX:
            current_row.append(cell)
            if y1 > current_row_max_y:
                current_row_max_y = y1
        else:
            # finish previous row
            if current_row:
                current_row.sort(key=lambda rect: rect[0])
                rows.append(current_row)
            current_row = [cell]
            current_row_max_y = y1

    if current_row:
        current_row.sort(key=lambda rect: rect[0])
        rows.append(current_row)

    cropped_rows: list[list[np.ndarray]] = []
    for row in rows:
        row_crops: list[np.ndarray] = []
        for x, y, width, height in row:
            crop = bgr_array[y : y + height, x : x + width]
            if crop.size:
                row_crops.append(crop.copy())
        if row_crops:
            cropped_rows.append(row_crops)

    return cropped_rows