"""Unit tests for the benchmark pipeline.

Guards against regressions in:
  - ground truth generation
  - extractors (openpyxl, pandas, python-calamine, pyexcel)
  - metrics (recall / accuracy comparisons)
  - value normalisation (dates, booleans, types)
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractors import EXTRACTORS, _norm  # noqa: E402
from metrics import (  # noqa: E402
    cell_value_accuracy,
    chart_recall,
    comment_recall,
    defined_names_recall,
    evaluate,
    formula_recall,
    hyperlink_recall,
    merged_recall,
    table_recall,
)


WORKBOOK = ROOT / "test_data" / "complex_workbook_cached.xlsx"
ORIGINAL = ROOT / "test_data" / "complex_workbook.xlsx"
GT_PATH = ROOT / "ground_truth" / "ground_truth.json"


@pytest.fixture(scope="module")
def gt():
    return json.loads(GT_PATH.read_text())


# ---------------------------------------------------------------- ground truth
def test_ground_truth_exists():
    assert GT_PATH.exists(), f"Missing ground truth at {GT_PATH}"


def test_ground_truth_has_all_sheets(gt):
    expected = {"Sales", "Inventory", "Quarterly P&L", "Dashboard", "README"}
    assert set(gt["sheets"].keys()) == expected


def test_ground_truth_formulas_have_computed_values(gt):
    missing = []
    for sn, sheet in gt["sheets"].items():
        for coord, cell in sheet["cells"].items():
            if cell["formula"] and cell["value"] is None:
                missing.append((sn, coord))
    assert not missing, f"Formula cells missing computed value: {missing[:10]}"


def test_ground_truth_charts_present(gt):
    total = sum(len(s["charts"]) for s in gt["sheets"].values())
    assert total == 6, f"Expected 6 charts, got {total}"


def test_ground_truth_tables_present(gt):
    tables = [t for s in gt["sheets"].values() for t in s["tables"]]
    assert any(t["name"] == "SalesTable" for t in tables)


def test_ground_truth_defined_names(gt):
    assert "Revenue" in gt["defined_names"]
    assert "Sales!" in gt["defined_names"]["Revenue"]


# ---------------------------------------------------------------- normalisation
def test_norm_handles_dates():
    import datetime as dt
    assert _norm(dt.date(2024, 1, 1)) == "2024-01-01"
    assert _norm(dt.datetime(2024, 1, 1, 12, 30)) == "2024-01-01"


def test_norm_handles_booleans():
    assert _norm(True) is True
    assert _norm(False) is False


def test_norm_handles_none():
    assert _norm(None) is None


def test_norm_handles_numpy():
    import numpy as np
    assert _norm(np.int64(5)) == 5
    assert _norm(np.float64(1.5)) == 1.5


# ---------------------------------------------------------------- extractors
@pytest.mark.parametrize("name", ["openpyxl", "openpyxl_dataonly", "pandas",
                                 "python-calamine", "pyexcel"])
def test_extractor_returns_dict(name):
    fn = EXTRACTORS[name]
    out = fn(str(WORKBOOK))
    assert isinstance(out, dict)
    assert "values" in out and "sheets" in out
    assert len(out["sheets"]) >= 5


def test_extractor_xlrd_fails_on_xlsx():
    out = EXTRACTORS["xlrd"](str(WORKBOOK))
    assert "error" in out, "xlrd should fail on .xlsx"
    assert out["sheets"] == []


@pytest.mark.parametrize("name", ["openpyxl", "openpyxl_dataonly", "pandas",
                                 "python-calamine", "pyexcel"])
def test_extractor_finds_sales_table(name):
    out = EXTRACTORS[name](str(WORKBOOK))
    tables = out.get("tables", {}).get("Sales", [])
    if name in ("openpyxl", "openpyxl_dataonly"):
        assert any(t["name"] == "SalesTable" for t in tables)
    else:
        assert tables == []


@pytest.mark.parametrize("name", ["openpyxl", "openpyxl_dataonly"])
def test_extractor_finds_charts(name):
    out = EXTRACTORS[name](str(WORKBOOK))
    total_charts = sum(len(v) for v in out.get("charts", {}).values())
    assert total_charts == 6, f"{name}: expected 6 charts, got {total_charts}"


def test_openpyxl_exposes_formulas():
    out = EXTRACTORS["openpyxl"](str(WORKBOOK))
    total = sum(len(v) for v in out.get("formulas", {}).values())
    assert total >= 188, f"openpyxl should expose >=188 formula cells, got {total}"


def test_openpyxl_exposes_defined_names():
    out = EXTRACTORS["openpyxl"](str(WORKBOOK))
    assert "Revenue" in out["defined_names"]


def test_openpyxl_exposes_hyperlinks():
    out = EXTRACTORS["openpyxl"](str(WORKBOOK))
    links = out.get("hyperlinks", {}).get("Dashboard", [])
    assert len(links) == 3, f"expected 3 hyperlinks in Dashboard, got {len(links)}"


def test_openpyxl_exposes_comments():
    out = EXTRACTORS["openpyxl"](str(WORKBOOK))
    comments = out.get("comments", {}).get("Dashboard", [])
    assert len(comments) == 1
    assert "Auto-generated" in comments[0]["text"]


# ---------------------------------------------------------------- metrics
def test_cell_value_accuracy_perfect_match():
    gt = {"sheets": {"S": {"cells": {"A1": {"value": 5, "formula": None}}}}}
    ext = {"sheets": ["S"], "values": {"S": {"A1": 5}}}
    res = cell_value_accuracy(gt, ext)
    assert res["accuracy"] == 1.0
    assert res["correct"] == 1


def test_cell_value_accuracy_handles_floats():
    gt = {"sheets": {"S": {"cells": {"A1": {"value": 0.43, "formula": None}}}}}
    ext = {"sheets": ["S"], "values": {"S": {"A1": 0.43000001}}}
    res = cell_value_accuracy(gt, ext)
    assert res["accuracy"] == 1.0


def test_cell_value_accuracy_handles_formula_value():
    gt = {"sheets": {"S": {"cells": {"A1": {"value": 42, "formula": "=B1+C1"}}}}}
    ext = {"sheets": ["S"], "values": {"S": {"A1": 42}}}
    res = cell_value_accuracy(gt, ext)
    assert res["accuracy"] == 1.0


def test_cell_value_accuracy_counts_missing():
    gt = {"sheets": {"S": {"cells": {"A1": {"value": 1, "formula": None},
                                       "A2": {"value": 2, "formula": None}}}}}
    ext = {"sheets": ["S"], "values": {"S": {"A1": 1}}}
    res = cell_value_accuracy(gt, ext)
    assert res["accuracy"] == 0.5


def test_formula_recall():
    gt = {"sheets": {"S": {"cells": {"A1": {"value": 1, "formula": "=1"},
                                       "A2": {"value": 2, "formula": "=2"},
                                       "B1": {"value": 3, "formula": None}}}}}
    ext = {"sheets": ["S"], "formulas": {"S": {"A1": "=1"}}}
    res = formula_recall(gt, ext)
    assert res["formulas_in_gt"] == 2
    assert res["matched"] == 1
    assert res["recall"] == 0.5


def test_recall_helpers():
    gt = {"sheets": {"S": {"tables": [{"name": "T1", "ref": "A1:B2"}],
                          "charts": [{"type": "BarChart", "title": "X", "series_count": 1}],
                          "merged_cells": ["A1:B1"],
                          "hyperlinks": [{"cell": "A1", "target": "#Sheet!A1"}],
                          "comments": [{"cell": "A1", "author": "me", "text": "hi"}],
                          "cells": {}}}}
    gt["defined_names"] = {"foo": "Sheet!$A$1"}
    ext = {
        "sheets": ["S"],
        "tables": {"S": [{"name": "T1", "ref": "A1:B2"}]},
        "charts": {"S": [{"type": "BarChart", "title": "X", "series_count": 1}]},
        "merged_cells": {"S": ["A1:B1"]},
        "hyperlinks": {"S": [{"cell": "A1", "target": "#Sheet!A1"}]},
        "comments": {"S": [{"cell": "A1", "author": "me", "text": "hi"}]},
        "defined_names": {"foo": "Sheet!$A$1"},
    }
    assert table_recall(gt, ext)["recall"] == 1.0
    assert chart_recall(gt, ext)["recall"] == 1.0
    assert merged_recall(gt, ext)["recall"] == 1.0
    assert hyperlink_recall(gt, ext)["recall"] == 1.0
    assert comment_recall(gt, ext)["recall"] == 1.0
    assert defined_names_recall(gt, ext)["recall"] == 1.0


def test_evaluate_returns_all_aspects():
    res = evaluate({"sheets": {}, "defined_names": {}},
                   {"sheets": [], "values": {}, "formulas": {}, "tables": {},
                    "charts": {}, "merged_cells": {}, "hyperlinks": {}, "comments": {},
                    "defined_names": {}})
    expected = {
        "cell_value_accuracy", "formula_recall", "merged_recall", "table_recall",
        "chart_recall", "hyperlink_recall", "comment_recall", "defined_names_recall",
    }
    assert expected.issubset(res.keys())


# ---------------------------------------------------------------- end-to-end
def test_openpyxl_dataonly_full_match(gt):
    out = EXTRACTORS["openpyxl_dataonly"](str(WORKBOOK))
    res = evaluate(gt, out)
    assert res["cell_value_accuracy"]["accuracy"] == 1.0, res["cell_value_accuracy"]


def test_pandas_full_match(gt):
    out = EXTRACTORS["pandas"](str(WORKBOOK))
    res = evaluate(gt, out)
    assert res["cell_value_accuracy"]["accuracy"] == 1.0, res["cell_value_accuracy"]


def test_calamine_full_match(gt):
    out = EXTRACTORS["python-calamine"](str(WORKBOOK))
    res = evaluate(gt, out)
    assert res["cell_value_accuracy"]["accuracy"] == 1.0, res["cell_value_accuracy"]


def test_pyexcel_full_match(gt):
    out = EXTRACTORS["pyexcel"](str(WORKBOOK))
    res = evaluate(gt, out)
    assert res["cell_value_accuracy"]["accuracy"] == 1.0, res["cell_value_accuracy"]
