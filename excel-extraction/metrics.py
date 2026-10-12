"""Compare an extractor output against the ground truth.

Returns per-aspect metrics:

  cell_value_accuracy   - fraction of (sheet, cell) pairs with matching values
  formula_recall        - fraction of formula cells whose formula text is recovered
  merged_recall         - fraction of merged ranges recovered
  table_recall          - fraction of tables recovered
  chart_recall          - fraction of charts recovered
  hyperlink_recall      - fraction of hyperlinks recovered
  comment_recall        - fraction of comments recovered
  defined_names_recall  - fraction of defined names recovered
"""

from __future__ import annotations

import math
from typing import Any


def _close(a, b, *, rel: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            if math.isnan(a) and math.isnan(b):
                return True
        except TypeError:
            pass
        if abs(a - b) <= abs_tol:
            return True
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom <= rel
    # Strings: ignore trailing whitespace and case-sensitively compare; Excel
    # numbers sometimes come back as "1.0" from formulas lib, int 1 from openpyxl
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    if isinstance(a, str):
        try:
            return _close(float(a), b)
        except (TypeError, ValueError):
            return False
    if isinstance(b, str):
        try:
            return _close(a, float(b))
        except (TypeError, ValueError):
            return False
    return a == b


def _as_set(d: dict | list) -> set:
    if isinstance(d, dict):
        return set(d.keys())
    return set(d) if d else set()


def _gt_expected_value(gt_cell: dict) -> Any:
    """For a ground-truth cell, return the expected extracted value.

    For formula cells, prefer the computed value (if available), since most
    libraries return calculated numbers, not the formula string.
    """
    if gt_cell["formula"] is not None:
        # Computed value takes precedence; fallback to formula text.
        return gt_cell.get("value") if gt_cell.get("value") is not None else gt_cell["formula"]
    return gt_cell["value"]


def cell_value_accuracy(gt: dict, extracted: dict) -> dict:
    total = 0
    correct = 0
    missing = []
    mismatched = []
    sheets_gt = set(gt["sheets"].keys())
    sheets_ext = set(extracted.get("sheets", []))

    for sn, sheet_gt in gt["sheets"].items():
        extracted_vals = (extracted.get("values") or {}).get(sn, {})
        for coord, gt_cell in sheet_gt["cells"].items():
            expected = _gt_expected_value(gt_cell)
            if coord not in extracted_vals:
                # Treat as missing (libraries may skip empty cells)
                # but count if the extractor claims the sheet exists.
                if sn in sheets_ext:
                    total += 1
                    missing.append((sn, coord))
                continue
            actual = extracted_vals[coord]
            total += 1
            if _close(actual, expected):
                correct += 1
            else:
                mismatched.append({
                    "sheet": sn,
                    "cell": coord,
                    "expected": expected,
                    "actual": actual,
                    "is_formula": gt_cell["formula"] is not None,
                })

    return {
        "total_cells_in_gt": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_count": len(missing),
        "mismatched_count": len(mismatched),
        "mismatched_sample": mismatched[:10],
    }


def formula_recall(gt: dict, extracted: dict) -> dict:
    formula_cells_gt = []
    formula_cells_ext = []
    for sn, sheet_gt in gt["sheets"].items():
        for coord, c in sheet_gt["cells"].items():
            if c["formula"] is not None:
                formula_cells_gt.append((sn, coord))
    for sn, coords in (extracted.get("formulas") or {}).items():
        for coord in coords:
            formula_cells_ext.append((sn, coord))

    gt_set = set(formula_cells_gt)
    ext_set = set(formula_cells_ext)
    matched = gt_set & ext_set
    return {
        "formulas_in_gt": len(gt_set),
        "formulas_extracted": len(ext_set),
        "matched": len(matched),
        "recall": len(matched) / len(gt_set) if gt_set else 0.0,
    }


def _recall(gt_items, ext_items) -> dict:
    gt_set = {tuple(sorted(d.items())) if isinstance(d, dict) else d for d in gt_items}
    ext_set = {tuple(sorted(d.items())) if isinstance(d, dict) else d for d in ext_items}
    matched = gt_set & ext_set
    return {
        "gt_count": len(gt_set),
        "extracted_count": len(ext_set),
        "matched": len(matched),
        "recall": len(matched) / len(gt_set) if gt_set else 0.0,
    }


def merged_recall(gt: dict, extracted: dict) -> dict:
    gt_items, ext_items = [], []
    for sn, sheet in gt["sheets"].items():
        for r in sheet["merged_cells"]:
            gt_items.append((sn, r))
    for sn, ranges in (extracted.get("merged_cells") or {}).items():
        for r in ranges:
            ext_items.append((sn, r))
    return _recall(gt_items, ext_items)


def table_recall(gt: dict, extracted: dict) -> dict:
    gt_items, ext_items = [], []
    for sn, sheet in gt["sheets"].items():
        for t in sheet["tables"]:
            gt_items.append((sn, t["name"], t["ref"]))
    for sn, tables in (extracted.get("tables") or {}).items():
        for t in tables:
            ext_items.append((sn, t.get("name"), t.get("ref")))
    return _recall(gt_items, ext_items)


def chart_recall(gt: dict, extracted: dict) -> dict:
    """Recall on (sheet, type, title) tuples."""
    gt_items, ext_items = [], []
    for sn, sheet in gt["sheets"].items():
        for ch in sheet["charts"]:
            gt_items.append((sn, ch["type"], ch["title"]))
    for sn, charts in (extracted.get("charts") or {}).items():
        for ch in charts:
            ext_items.append((sn, ch.get("type"), ch.get("title")))
    return _recall(gt_items, ext_items)


def hyperlink_recall(gt: dict, extracted: dict) -> dict:
    gt_items, ext_items = [], []
    for sn, sheet in gt["sheets"].items():
        for h in sheet["hyperlinks"]:
            gt_items.append((sn, h["cell"], h["target"]))
    for sn, hls in (extracted.get("hyperlinks") or {}).items():
        for h in hls:
            ext_items.append((sn, h.get("cell"), h.get("target")))
    return _recall(gt_items, ext_items)


def comment_recall(gt: dict, extracted: dict) -> dict:
    gt_items, ext_items = [], []
    for sn, sheet in gt["sheets"].items():
        for c in sheet["comments"]:
            gt_items.append((sn, c["cell"], c["author"], c["text"]))
    for sn, comments in (extracted.get("comments") or {}).items():
        for c in comments:
            ext_items.append((sn, c.get("cell"), c.get("author"), c.get("text")))
    return _recall(gt_items, ext_items)


def defined_names_recall(gt: dict, extracted: dict) -> dict:
    gt_dn = gt.get("defined_names", {})
    ext_dn = extracted.get("defined_names", {})
    gt_items = list(gt_dn.items())
    ext_items = list(ext_dn.items())
    gt_set = set(gt_items)
    ext_set = set(ext_items)
    matched = gt_set & ext_set
    return {
        "gt_count": len(gt_set),
        "extracted_count": len(ext_set),
        "matched": len(matched),
        "recall": len(matched) / len(gt_set) if gt_set else 0.0,
    }


def evaluate(gt: dict, extracted: dict) -> dict:
    return {
        "cell_value_accuracy": cell_value_accuracy(gt, extracted),
        "formula_recall": formula_recall(gt, extracted),
        "merged_recall": merged_recall(gt, extracted),
        "table_recall": table_recall(gt, extracted),
        "chart_recall": chart_recall(gt, extracted),
        "hyperlink_recall": hyperlink_recall(gt, extracted),
        "comment_recall": comment_recall(gt, extracted),
        "defined_names_recall": defined_names_recall(gt, extracted),
    }
