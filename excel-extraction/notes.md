# Research Notes: Python Excel Data Extraction Library Benchmark

## Objective

Identify the **best Python library for extracting data from complex .xlsx
workbooks** — including charts, formulas, formatting, tables, hyperlinks,
comments, named ranges and merged cells — and quantify correctness + speed
empirically on a deliberately complex test file.

## Scope: libraries under test

All major actively-maintained, MIT/Apache, CPU-only Python libraries for
reading Excel workbooks:

| Library          | Backend       | License | Notes                                  |
|------------------|---------------|---------|----------------------------------------|
| openpyxl         | Pure Python   | MIT     | Reads/writes .xlsx, charts, formulas   |
| pandas           | openpyxl/etc. | BSD-3   | High-level DataFrames; no charts       |
| python-calamine  | Rust (calamine)| Apache-2| Fastest; .xlsx/.xls/.ods; no formulas  |
| pyexcel          | openpyxl/xlrd | BSD     | Wrapper; loses formatting              |
| xlrd             | C (legacy)    | BSD     | **2.x dropped .xlsx support**          |

Excluded:
- **xlwings** — requires Microsoft Excel installed (COM automation)
- **xlsxwriter** — write-only
- **pycel / formulas** — Excel formula evaluators, not general extractors
- **pandas + xlrd .xls path** — old binary format out of scope

## Test corpus (1 file, but a *complex* one)

`test_data/complex_workbook.xlsx` is built by `create_test_xlsx.py` and
contains:

- **5 sheets** (`Sales`, `Inventory`, `Quarterly P&L`, `Dashboard`, `README`)
- **374 cells** on `Sales`, **84** on `Inventory`, **42** on `Quarterly P&L`,
  **16** on `Dashboard`, **4** on `README` = **520 non-empty cells**
- **188 formulas** including SUM, SUMIF, MAX, AVERAGE, COUNTIF, IF, cross-sheet
  references, and a defined-range reference (`=SUM(Revenue)`)
- **1 structured table** (SalesTable)
- **5 merged-cell ranges** (4 on Dashboard, 1 on Sales)
- **6 charts**: bar (column), bar (horizontal), pie, line, scatter, area
- **3 hyperlinks** on the Dashboard "Quick Links" panel
- **1 cell comment** on Dashboard A14
- **1 defined name** (`Revenue`)
- **Conditional formatting** (colour scale + cell-is rule on Inventory!C)
- **Data types**: text, integer, float, percentage, currency, dates, booleans,
  intentional `#DIV/0!`-style string error in `Inventory`
- **Frozen panes** on Sales (`A2`) and Inventory (`B2`)
- **Custom column widths**

For libraries that read cached values (e.g. `openpyxl` with
`data_only=True`, `pandas`), we generate
`test_data/complex_workbook_cached.xlsx` by computing every formula with
the `formulas` library and injecting `<v>...</v>` into each sheet's XML.
This way libraries can be compared on equal footing.

## Metrics

### Accuracy
- **Cell value accuracy** — fraction of (sheet, cell) pairs whose extracted
  value matches ground truth (tolerance 1e-6 for floats; dates normalised to
  `YYYY-MM-DD`).
- **Formula recall** — fraction of formula cells whose formula text appears
  in the extractor's `formulas` dict. (`openpyxl` exposes formula strings;
  everything else only sees computed values, hence 0% for those.)
- **Merged-cell / table / chart / hyperlink / comment / defined-name recall**
  — fraction of expected structural items recovered exactly.

### Speed
- Median wall-clock time over 5 warm runs of `open(path) → full extraction`.
- Reported in milliseconds, log scale.

### Memory
- `tracemalloc` peak memory delta (best-effort).

## Ground truth

`generate_ground_truth.py` walks the workbook with `openpyxl` (which exposes
*all* of the features above) and produces `ground_truth/ground_truth.json`:

```json
{
  "sheets": {
    "Sales": {
      "cells":     {"F2": {"value": 62.5, "formula": "=D2*E2", ...}, ...},
      "charts":    [{"type": "BarChart", "title": "Revenue by Region", "series_count": 1}],
      "tables":    [{"name": "SalesTable", "ref": "A1:I41"}],
      "merged_cells": ["A42:E42"],
      "hyperlinks": [...],
      "comments":  [...]
    }, ...
  },
  "defined_names": {"Revenue": "Sales!$F$2:$F$41"}
}
```

Formula values are computed via `formulas.ExcelModel().calculate()`. (Note
that `formulas` upper-cases sheet names in its output keys; the GT loader
handles that with `.upper()` + `.rstrip("'")`.)

## Progress log

- 2026-07-19 — scaffolding + .venv with openpyxl/pandas/xlsxwriter/calamine/pyexcel/xlrd/formulas.
- 2026-07-19 — wrote `create_test_xlsx.py`, generated 21 KB workbook with 5 sheets / 6 charts / 188 formulas.
- 2026-07-19 — wrote `extractors.py` (one function per library, normalised return shape).
- 2026-07-19 — wrote `metrics.py` with per-aspect recall + value-accuracy helpers.
- 2026-07-19 — discovered openpyxl-with-`data_only=True` returns `None` for formula cells because
  Excel had not cached any values. Wrote `inject_cached_values.py` to compute and inject cached
  values into a parallel `complex_workbook_cached.xlsx`.
- 2026-07-19 — wrote `benchmark_runner.py` (median over 5 warm runs + `tracemalloc`).
- 2026-07-19 — first benchmark revealed xlrd 2.x no longer supports .xlsx (kept it for completeness).
- 2026-07-19 — wrote `visualize.py` (4 charts: speed / grouped bar / heatmap / scatter).
- 2026-07-19 — wrote pytest suite (38 tests, all passing).

## Open questions / future work

- **xlrd path** — could add a separate test with a `.xls` file to confirm
  xlrd works on legacy files (out of scope: most modern Excel is .xlsx).
- **Streaming** — none of the libraries tested support streaming; for huge
  workbooks consider `openpyxl.read_only=True` (still in-memory but lazy).
- **Pivot tables** — openpyxl 3.x can read pivot definitions but most
  libraries ignore them. Not tested here.
- **Conditional formatting / data validation rules** — openpyxl exposes them
  via dedicated APIs; not benchmarked in this round.
- **Macros (.xlsm)** — none of the tested libraries attempt to read VBA
  code (use `oletools` / `vba_extract` for that).

## Verdict (TL;DR)

- For **complete, faithful extraction** of any Excel feature
  (charts, tables, merges, hyperlinks, comments, defined names, formulas):
  **openpyxl wins**. The only library that recovers *every* structural
  feature in the test workbook.
- For **pure speed when you only need values** and the file already has
  cached formula values: **python-calamine** is ~12-15× faster than
  openpyxl and produces identical cell values for our test.
- For **DataFrame workflows**: `pandas.read_excel` is convenient but uses
  openpyxl under the hood and discards everything except values.
- For **old .xls files**: xlrd 2.x is the only option (or `pandas` with
  `xlrd<2.0`).
- **xlrd on .xlsx is dead** — use openpyxl or python-calamine.
