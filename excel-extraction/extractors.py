"""Unified Excel extractors.

Each extractor returns a dict of the same shape so the benchmark can compare
apples to apples:

    {
        "values": { sheet_name: { "A1": value, ... } },
        "formulas": { sheet_name: { "A1": "=SUM(B1:B10)", ... } },
        "merged_cells": { sheet_name: [ "A1:D1", ... ] },
        "tables": { sheet_name: [ {"name": ..., "ref": ...}, ... ] },
        "charts": { sheet_name: [ {"type": ..., "title": ..., "series_count": ...}, ... ] },
        "hyperlinks": { sheet_name: [ {"cell": ..., "target": ..., "display": ...}, ... ] },
        "comments": { sheet_name: [ {"cell": ..., "author": ..., "text": ...}, ... ] },
        "defined_names": { name: attr_text, ... },
        "sheets": [sheet_name, ...],
    }

Each library differs in what it can populate; missing keys mean "not supported".
All values are normalised via `_norm` for fair comparison.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pyexcel
import python_calamine
import xlrd


def _norm_date(v):
    """Normalise date / datetime to a canonical 'YYYY-MM-DD' string.

    We do not preserve time-of-day because some libraries return datetime and
    others return date, and the test file does not store any time-of-day.
    """
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return v


def _norm(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return [_norm(x) for x in v.tolist()]
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, tuple):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for x in v.items()}
    if isinstance(v, (dt.date, dt.datetime)):
        return _norm_date(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, bool):
        return v
    return v


# ---------------------------------------------------------------- openpyxl
def extract_openpyxl(path: str, *, data_only: bool = False) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=data_only)

    out = {
        "values": {},
        "formulas": {},
        "merged_cells": {},
        "tables": {},
        "charts": {},
        "hyperlinks": {},
        "comments": {},
        "defined_names": {},
        "sheets": list(wb.sheetnames),
    }

    for sn in wb.sheetnames:
        ws = wb[sn]
        vals: dict[str, Any] = {}
        formulas: dict[str, str] = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                # In data_only mode, formula cells return the cached numeric value
                # (or the formula string if Excel hasn't cached it).
                if data_only and isinstance(cell.value, str) and cell.value.startswith("="):
                    # No cached value -> cell.value IS the formula string. Skip.
                    formulas[cell.coordinate] = cell.value
                elif isinstance(cell.value, str) and cell.value.startswith("="):
                    formulas[cell.coordinate] = cell.value
                    # In non-data_only mode we still record the raw value (formula text)
                    vals[cell.coordinate] = cell.value
                else:
                    vals[cell.coordinate] = _norm(cell.value)
        out["values"][sn] = vals
        out["formulas"][sn] = formulas
        out["merged_cells"][sn] = sorted(str(r) for r in ws.merged_cells.ranges)

        tables = []
        for name, ref in ws.tables.items():
            tables.append({"name": name, "ref": ref})
        out["tables"][sn] = tables

        charts = []
        for ch in ws._charts:
            try:
                series_count = len(ch.series) if hasattr(ch, "series") else 0
            except Exception:
                series_count = 0
            title = None
            try:
                if ch.title and hasattr(ch.title, "tx") and ch.title.tx and ch.title.tx.rich:
                    paras = ch.title.tx.rich.paragraphs
                    if paras and paras[0].r:
                        title = paras[0].r[0].t
            except Exception:
                pass
            charts.append({"type": type(ch).__name__, "title": title, "series_count": series_count})
        out["charts"][sn] = charts

        hyperlinks = []
        for row in ws.iter_rows():
            for cell in row:
                if cell.hyperlink is not None:
                    hyperlinks.append({
                        "cell": cell.coordinate,
                        "target": getattr(cell.hyperlink, "target", None) or getattr(cell.hyperlink, "location", None),
                        "display": cell.value,
                    })
        out["hyperlinks"][sn] = hyperlinks

        comments = []
        for row in ws.iter_rows():
            for cell in row:
                if cell.comment is not None:
                    comments.append({
                        "cell": cell.coordinate,
                        "author": cell.comment.author,
                        "text": cell.comment.text,
                    })
        out["comments"][sn] = comments

    out["defined_names"] = {n: wb.defined_names[n].attr_text for n in wb.defined_names}
    return out


# ---------------------------------------------------------------- pandas
def extract_pandas(path: str) -> dict:
    """pandas via openpyxl engine."""
    xls = pd.ExcelFile(path, engine="openpyxl")
    out = {
        "values": {},
        "formulas": {},
        "merged_cells": {},
        "tables": {},
        "charts": {},
        "hyperlinks": {},
        "comments": {},
        "defined_names": {},
        "sheets": list(xls.sheet_names),
    }
    for sn in xls.sheet_names:
        df = xls.parse(sn, header=None, dtype=object)
        vals = {}
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                if pd.isna(df.iat[r, c]):
                    continue
                from openpyxl.utils import get_column_letter
                coord = f"{get_column_letter(c + 1)}{r + 1}"
                vals[coord] = _norm(df.iat[r, c])
        out["values"][sn] = vals
    return out


# ----------------------------------------------------------- python-calamine
def extract_calamine(path: str) -> dict:
    wb = python_calamine.CalamineWorkbook.from_path(path) if hasattr(python_calamine, "CalamineWorkbook") else None
    # Newer API: load_workbook
    if wb is None:
        wb = python_calamine.load_workbook(path)
    out = {
        "values": {},
        "formulas": {},
        "merged_cells": {},
        "tables": {},
        "charts": {},
        "hyperlinks": {},
        "comments": {},
        "defined_names": {},
        "sheets": list(wb.sheet_names) if hasattr(wb, "sheet_names") else [s.name for s in wb.sheets],
    }
    for sn in out["sheets"]:
        sheet = wb.get_sheet_by_name(sn) if hasattr(wb, "get_sheet_by_name") else wb.get_sheet(sn)
        rows = sheet.to_python(skip_empty_area=False)
        vals = {}
        # rows is list[list[Optional[CellValue]]]
        from openpyxl.utils import get_column_letter
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, cell in enumerate(row, start=1):
                if cell is None:
                    continue
                coord = f"{get_column_letter(c_idx)}{r_idx}"
                vals[coord] = _norm(cell)
        out["values"][sn] = vals
    return out


# ---------------------------------------------------------------- pyexcel
def extract_pyexcel(path: str) -> dict:
    sheets = pyexcel.get_book_dict(file_name=path)
    out = {
        "values": {},
        "formulas": {},
        "merged_cells": {},
        "tables": {},
        "charts": {},
        "hyperlinks": {},
        "comments": {},
        "defined_names": {},
        "sheets": list(sheets.keys()),
    }
    from openpyxl.utils import get_column_letter
    for sn, rows in sheets.items():
        vals = {}
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, cell in enumerate(row, start=1):
                if cell in (None, ""):
                    continue
                coord = f"{get_column_letter(c_idx)}{r_idx}"
                vals[coord] = _norm(cell)
        out["values"][sn] = vals
    return out


# ---------------------------------------------------------------- xlrd
def extract_xlrd(path: str) -> dict:
    """xlrd 2.x is xls-only. Attempt to open .xlsx and capture the failure."""
    try:
        book = xlrd.open_workbook(path)
        out = {
            "values": {},
            "formulas": {},
            "merged_cells": {},
            "tables": {},
            "charts": {},
            "hyperlinks": {},
            "comments": {},
            "defined_names": {},
            "sheets": book.sheet_names(),
        }
        for sn in book.sheet_names():
            sh = book.sheet_by_name(sn)
            vals = {}
            from openpyxl.utils import get_column_letter
            for r in range(sh.nrows):
                for c in range(sh.ncols):
                    cell = sh.cell(r, c)
                    if cell.ctype == 0:  # XL_CELL_EMPTY
                        continue
                    coord = f"{get_column_letter(c + 1)}{r + 1}"
                    vals[coord] = _norm(cell.value)
            out["values"][sn] = vals
        return out
    except Exception as e:
        return {
            "values": {},
            "formulas": {},
            "merged_cells": {},
            "tables": {},
            "charts": {},
            "hyperlinks": {},
            "comments": {},
            "defined_names": {},
            "sheets": [],
            "error": f"{type(e).__name__}: {e}",
        }


def extract_openpyxl_dataonly(path: str) -> dict:
    """openpyxl with data_only=True: returns cached formula values, not formulas.

    Only works for workbooks that carry cached values (we precompute them in
    `complex_workbook_cached.xlsx`).
    """
    return extract_openpyxl(path, data_only=True)


EXTRACTORS = {
    "openpyxl": extract_openpyxl,
    "openpyxl_dataonly": extract_openpyxl_dataonly,
    "pandas": extract_pandas,
    "python-calamine": extract_calamine,
    "pyexcel": extract_pyexcel,
    "xlrd": extract_xlrd,
}
