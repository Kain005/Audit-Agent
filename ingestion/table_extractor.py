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


def _density_segment_core(bgr_array: np.ndarray) -> list[np.ndarray]:
    if bgr_array is None or getattr(bgr_array, "size", 0) == 0:
        return []

    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY) if bgr_array.ndim == 3 else bgr_array.copy()
    inverted = cv2.bitwise_not(gray)
    density = np.sum(inverted, axis=1).astype(np.float32)

    kernel = np.ones(10, dtype=np.float32) / 10.0
    smoothed = np.convolve(density, kernel, mode="same")

    if smoothed.size == 0:
        return []

    candidate_minima: list[int] = []
    candidate_maxima: list[int] = []
    window_radius = 20

    for index in range(window_radius, len(smoothed) - window_radius):
        center_value = float(smoothed[index])
        left_window = smoothed[index - window_radius : index]
        right_window = smoothed[index + 1 : index + window_radius + 1]

        if left_window.size != window_radius or right_window.size != window_radius:
            continue

        if center_value < float(np.min(left_window)) and center_value < float(np.min(right_window)):
            candidate_minima.append(index)

        if center_value > float(np.max(left_window)) and center_value > float(np.max(right_window)):
            candidate_maxima.append(index)

    if not candidate_minima or not candidate_maxima:
        return []

    surviving_minima: list[int] = []
    for minimum_index in candidate_minima:
        left_max_index = None
        right_max_index = None

        for maximum_index in reversed(candidate_maxima):
            if maximum_index < minimum_index:
                left_max_index = maximum_index
                break

        for maximum_index in candidate_maxima:
            if maximum_index > minimum_index:
                right_max_index = maximum_index
                break

        if left_max_index is None or right_max_index is None:
            continue

        left_max_value = float(smoothed[left_max_index])
        right_max_value = float(smoothed[right_max_index])
        average_nearest_maxima = (left_max_value + right_max_value) / 2.0
        if average_nearest_maxima <= 0:
            continue

        if float(smoothed[minimum_index]) < 0.65 * average_nearest_maxima:
            surviving_minima.append(minimum_index)

    cut_points = sorted({cut for cut in surviving_minima if 0 < cut < bgr_array.shape[0]})
    if not cut_points:
        return []

    boundaries = [0, *cut_points, bgr_array.shape[0]]
    strips: list[np.ndarray] = []
    for top, bottom in zip(boundaries, boundaries[1:]):
        if bottom <= top:
            continue
        strip = bgr_array[top:bottom]
        if strip.size and 30 <= strip.shape[0] <= 300:
            strips.append(strip.copy())

    return strips


def segment_rows_by_density(bgr_array: np.ndarray) -> list[np.ndarray]:
    initial_strips = _density_segment_core(bgr_array)
    if len(initial_strips) < 3:
        return []

    final_strips: list[np.ndarray] = []
    for strip in initial_strips:
        if strip.shape[0] <= 140:
            final_strips.append(strip)
            continue

        re_detected_strips = _density_segment_core(strip)
        candidate_sub_strips = re_detected_strips if re_detected_strips else [strip]

        if candidate_sub_strips:
            median_height = float(np.median([candidate.shape[0] for candidate in candidate_sub_strips]))
        else:
            median_height = float(strip.shape[0])

        if median_height <= 0:
            median_height = float(strip.shape[0])

        for candidate in candidate_sub_strips:
            if candidate.shape[0] <= 140:
                final_strips.append(candidate)
                continue

            strip_height = int(candidate.shape[0])
            slice_count = max(2, int(round(strip_height / median_height)))
            slice_count = min(slice_count, strip_height)

            if slice_count <= 1:
                final_strips.append(candidate)
                continue

            boundaries = np.linspace(0, strip_height, slice_count + 1, dtype=int)
            for top, bottom in zip(boundaries, boundaries[1:]):
                if bottom <= top:
                    continue
                sliced = candidate[top:bottom]
                if sliced.size and sliced.shape[0] >= 30:
                    final_strips.append(sliced.copy())

    if len(final_strips) < 3:
        return []

    return final_strips