"""Build a deliberately complex .xlsx file for benchmarking extraction libraries.

Sheets created:
  1. Sales             - tabular data, formulas, merged cells, chart, table, frozen panes
  2. Inventory         - mixed types (date / bool / error), conditional formatting, bar chart
  3. Quarterly P&L     - cross-sheet formulas (VLOOKUP, SUMIF), pie + line charts
  4. Dashboard         - merged cells, hyperlinks, comments, scatter + area charts
  5. README            - plain text sheet (sanity)

Output:
  test_data/complex_workbook.xlsx
"""

from __future__ import annotations

import datetime as dt
import pathlib

from openpyxl import Workbook
from openpyxl.chart import AreaChart, BarChart, LineChart, PieChart, ScatterChart, Reference, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.table import Table, TableStyleInfo


HERE = pathlib.Path(__file__).parent
OUT = HERE / "test_data" / "complex_workbook.xlsx"
OUT.parent.mkdir(parents=True, exist_ok=True)


def styled_header(cell, fill_color: str = "4472C4", font_color: str = "FFFFFF") -> None:
    cell.font = Font(name="Calibri", size=11, bold=True, color=font_color)
    cell.fill = PatternFill("solid", fgColor=fill_color)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    thin = Side(border_style="thin", color="BFBFBF")
    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)


def build():
    wb = Workbook()

    # ------------------------------------------------------------------ Sales
    ws = wb.active
    ws.title = "Sales"
    ws.freeze_panes = "A2"

    headers = ["Date", "Region", "Product", "Units", "Unit Price", "Revenue", "Cost", "Profit", "Margin %"]
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col, value=h)
        styled_header(c)

    regions = ["North", "South", "East", "West"]
    products = [("Widget", 12.5, 7.0), ("Gadget", 25.0, 14.0), ("Sprocket", 4.0, 2.5), ("Cog", 1.5, 0.8)]
    start = dt.date(2024, 1, 1)
    rng = 0
    for i in range(40):
        row = i + 2
        d = start + dt.timedelta(days=i * 3)
        name, price, cost = products[i % len(products)]
        region = regions[i % len(regions)]
        units = 5 + (i * 3) % 47
        ws.cell(row=row, column=1, value=d).number_format = "yyyy-mm-dd"
        ws.cell(row=row, column=2, value=region)
        ws.cell(row=row, column=3, value=name)
        ws.cell(row=row, column=4, value=units)
        ws.cell(row=row, column=5, value=price).number_format = '"$"#,##0.00'
        # Formula columns
        ws.cell(row=row, column=6, value=f"=D{row}*E{row}").number_format = '"$"#,##0.00'
        ws.cell(row=row, column=7, value=f"=D{row}*{cost}").number_format = '"$"#,##0.00'
        ws.cell(row=row, column=8, value=f"=F{row}-G{row}").number_format = '"$"#,##0.00'
        ws.cell(row=row, column=9, value=f"=IF(F{row}=0,0,H{row}/F{row})").number_format = "0.0%"

    # Totals row with merged label
    total_row = 42
    ws.cell(row=total_row, column=1, value="Totals")
    ws.merge_cells(start_row=total_row, start_column=1, end_row=total_row, end_column=5)
    for col, formula in [
        (6, f"=SUM(F2:F{total_row-1})"),
        (7, f"=SUM(G2:G{total_row-1})"),
        (8, f"=SUM(H2:H{total_row-1})"),
        (9, f"=IF(F{total_row}=0,0,H{total_row}/F{total_row})"),
    ]:
        c = ws.cell(row=total_row, column=col, value=formula)
        c.number_format = '"$"#,##0.00' if col < 9 else "0.0%"
        c.font = Font(bold=True)

    # Column widths
    widths = [12, 10, 12, 8, 12, 14, 12, 14, 12]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # Structured table (Excel Table)
    tbl = Table(displayName="SalesTable", ref=f"A1:I{total_row-1}")
    tbl.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False,
                                        showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    ws.add_table(tbl)

    # Bar chart on Sales
    bar = BarChart()
    bar.type = "col"
    bar.style = 11
    bar.title = "Revenue by Region"
    bar.y_axis.title = "Revenue"
    bar.x_axis.title = "Region"
    data_ref = Reference(ws, min_col=6, min_row=1, max_row=total_row - 1, max_col=6)
    cats_ref = Reference(ws, min_col=2, min_row=2, max_row=total_row - 1)
    bar.add_data(data_ref, titles_from_data=True)
    bar.set_categories(cats_ref)
    bar.height = 9
    bar.width = 18
    ws.add_chart(bar, "K2")

    # Named range "Revenue"
    dn = DefinedName("Revenue", attr_text=f"Sales!$F$2:$F${total_row-1}")
    wb.defined_names["Revenue"] = dn

    # ----------------------------------------------------------- Inventory
    inv = wb.create_sheet("Inventory")
    inv.freeze_panes = "B2"
    invh = ["SKU", "Item", "Stock", "Reorder?", "Last Restock", "Unit Cost", "Notes"]
    for c, h in enumerate(invh, start=1):
        styled_header(inv.cell(row=1, column=c, value=h))

    rows = [
        ("A-001", "Bearing 6mm", 320, False, dt.date(2024, 5, 12), 1.20, "high turnover"),
        ("A-002", "Bearing 8mm", 18, True, dt.date(2024, 2, 1), 1.50, "reorder now"),
        ("A-003", "Hex bolt M4", 1042, False, dt.date(2024, 6, 18), 0.05, "ok"),
        ("A-004", "Hex bolt M6", 0, True, dt.date(2023, 11, 4), 0.09, "out of stock"),
        ("A-005", "Hex nut M6", 540, False, dt.date(2024, 6, 2), 0.04, "ok"),
        ("A-006", "Washer 6mm", 75, False, dt.date(2024, 5, 30), 0.02, "watch"),
        ("A-007", "Spring 12mm", 3, True, dt.date(2023, 9, 9), 0.85, "reorder now"),
        ("A-008", "Spring 18mm", 220, False, dt.date(2024, 3, 22), 1.10, "ok"),
        ("A-009", "O-ring 10mm", 0, True, dt.date(2023, 12, 1), 0.15, "out of stock"),
        ("A-010", "O-ring 20mm", 410, False, dt.date(2024, 4, 9), 0.25, "ok"),
        ("DIV0!", "#DIV/0! sample", 12, False, dt.date(2024, 1, 1), 0.10, "intentional error"),
    ]
    for i, (sku, name, stock, reorder, dtv, cost, note) in enumerate(rows, start=2):
        inv.cell(row=i, column=1, value=sku)
        inv.cell(row=i, column=2, value=name)
        inv.cell(row=i, column=3, value=stock)
        inv.cell(row=i, column=4, value=reorder)
        inv.cell(row=i, column=5, value=dtv).number_format = "yyyy-mm-dd"
        inv.cell(row=i, column=6, value=cost).number_format = '"$"#,##0.00'
        inv.cell(row=i, column=7, value=note)

    # Conditional formatting: low-stock red, ok green
    inv.conditional_formatting.add(
        "C2:C12",
        CellIsRule(operator="lessThan", formula=["50"], stopIfTrue=False,
                   fill=PatternFill("solid", fgColor="FFC7CE")),
    )
    inv.conditional_formatting.add(
        "C2:C12",
        ColorScaleRule(start_type="min", start_color="63BE7B",
                       mid_type="percentile", mid_value=50, mid_color="FFEB84",
                       end_type="max", end_color="F8696B"),
    )

    inv.column_dimensions["A"].width = 8
    inv.column_dimensions["B"].width = 18
    inv.column_dimensions["E"].width = 14
    inv.column_dimensions["G"].width = 22

    bar2 = BarChart()
    bar2.type = "bar"
    bar2.title = "Stock by SKU"
    data_ref = Reference(inv, min_col=3, min_row=1, max_row=12, max_col=3)
    cats_ref = Reference(inv, min_col=1, min_row=2, max_row=12)
    bar2.add_data(data_ref, titles_from_data=True)
    bar2.set_categories(cats_ref)
    bar2.height = 9
    bar2.width = 18
    inv.add_chart(bar2, "I2")

    # ------------------------------------------------------------- Quarterly P&L
    qpl = wb.create_sheet("Quarterly P&L")
    qplh = ["Quarter", "Revenue", "COGS", "Gross Profit", "OpEx", "Net Income"]
    for c, h in enumerate(qplh, start=1):
        styled_header(qpl.cell(row=1, column=c, value=h))

    qpl_data = [
        ("Q1 2024", 482_300, 218_900),
        ("Q2 2024", 555_120, 244_100),
        ("Q3 2024", 612_450, 271_800),
        ("Q4 2024", 701_900, 309_500),
    ]
    for i, (q, rev, cogs) in enumerate(qpl_data, start=2):
        qpl.cell(row=i, column=1, value=q)
        qpl.cell(row=i, column=2, value=rev).number_format = '"$"#,##0'
        qpl.cell(row=i, column=3, value=cogs).number_format = '"$"#,##0'
        qpl.cell(row=i, column=4, value=f"=B{i}-C{i}").number_format = '"$"#,##0'
        qpl.cell(row=i, column=5, value=f"=B{i}*0.32").number_format = '"$"#,##0'
        qpl.cell(row=i, column=6, value=f"=D{i}-E{i}").number_format = '"$"#,##0'

    # Totals
    qpl.cell(row=6, column=1, value="Total")
    for col, col_letter in enumerate(["B", "C", "D", "E", "F"], start=2):
        c = qpl.cell(row=6, column=col, value=f"=SUM({col_letter}2:{col_letter}5)")
        c.number_format = '"$"#,##0'
        c.font = Font(bold=True)

    # Cross-sheet formulas: look up Sales totals
    qpl.cell(row=8, column=1, value="Sales total (cross-sheet):")
    qpl.cell(row=8, column=2, value=f"='Sales'!F42").number_format = '"$"#,##0'
    qpl.cell(row=9, column=1, value="Best quarter revenue:")
    qpl.cell(row=9, column=2, value="=MAX(B2:B5)").number_format = '"$"#,##0'
    qpl.cell(row=10, column=1, value="SUMIF (revenue > 500k):")
    qpl.cell(row=10, column=2, value="=SUMIF(B2:B5,\">500000\")").number_format = '"$"#,##0'

    for col, w in zip("ABCDEF", [14, 14, 14, 16, 14, 14]):
        qpl.column_dimensions[col].width = w

    pie = PieChart()
    pie.title = "Revenue Share"
    pie.dataLabels = DataLabelList(showPercent=True)
    pie_ref = Reference(qpl, min_col=2, min_row=2, max_row=5, max_col=2)
    pie_cats = Reference(qpl, min_col=1, min_row=2, max_row=5)
    pie.add_data(pie_ref, titles_from_data=True)
    pie.set_categories(pie_cats)
    pie.height = 9
    pie.width = 14
    qpl.add_chart(pie, "H2")

    line = LineChart()
    line.title = "Quarterly Trend"
    line.y_axis.title = "USD"
    line.x_axis.title = "Quarter"
    line_data = Reference(qpl, min_col=2, min_row=1, max_row=5, max_col=2)
    line_cats = Reference(qpl, min_col=1, min_row=2, max_row=5)
    line.add_data(line_data, titles_from_data=True)
    line.set_categories(line_cats)
    line.height = 9
    line.width = 18
    qpl.add_chart(line, "H20")

    # ------------------------------------------------------------- Dashboard
    dash = wb.create_sheet("Dashboard")
    dash.merge_cells("A1:H1")
    title = dash["A1"]
    title.value = "Operations Dashboard — 2024"
    title.font = Font(name="Calibri", size=18, bold=True, color="1F4E78")
    title.alignment = Alignment(horizontal="center", vertical="center")
    dash.row_dimensions[1].height = 30

    dash["A3"] = "Key Metrics"
    styled_header(dash["A3"], fill_color="70AD47")
    dash.merge_cells("A3:B3")
    dash["A4"] = "Total Revenue"
    dash["B4"] = "=SUM(Revenue)"
    dash["B4"].number_format = '"$"#,##0.00'
    dash["A5"] = "Total Units Sold"
    dash["B5"] = "=SUM('Sales'!D2:D41)"
    dash["A6"] = "Avg Margin"
    dash["B6"] = "=AVERAGE('Sales'!I2:I41)"
    dash["B6"].number_format = "0.0%"
    dash["A7"] = "Items needing reorder"
    dash["B7"] = '=COUNTIF(Inventory!D2:D12,TRUE)'

    dash["A9"] = "Quick Links"
    styled_header(dash["A9"], fill_color="70AD47")
    dash.merge_cells("A9:B9")
    dash["A10"] = "Sales sheet"
    dash["A10"].hyperlink = "#'Sales'!A1"
    dash["A10"].font = Font(color="0563C1", underline="single")
    dash["A11"] = "Inventory sheet"
    dash["A11"].hyperlink = "#'Inventory'!A1"
    dash["A11"].font = Font(color="0563C1", underline="single")
    dash["A12"] = "Quarterly P&L"
    dash["A12"].hyperlink = "#'Quarterly P&L'!A1"
    dash["A12"].font = Font(color="0563C1", underline="single")

    dash["D3"] = "Scatter: Price vs Units"
    styled_header(dash["D3"], fill_color="ED7D31")
    dash.merge_cells("D3:H3")

    dash["A14"] = "Note: see dashboard for visualisations."
    dash["A14"].comment = Comment("Auto-generated by create_test_xlsx.py", "Open SWE")

    for col, w in zip("ABCDEFGH", [22, 18, 4, 14, 14, 14, 14, 14]):
        dash.column_dimensions[col].width = w

    sc = ScatterChart()
    sc.title = "Unit Price vs Units Sold"
    sc.style = 13
    sc.x_axis.title = "Unit Price"
    sc.y_axis.title = "Units"
    x_ref = Reference(ws, min_col=5, min_row=2, max_row=41)  # unit price
    y_ref = Reference(ws, min_col=4, min_row=2, max_row=41)  # units
    series = Series(values=y_ref, xvalues=x_ref, title="Sales")
    sc.series.append(series)
    sc.height = 10
    sc.width = 18
    dash.add_chart(sc, "D4")

    area = AreaChart()
    area.title = "Cumulative Inventory"
    area.grouping = "stacked"
    area_data = Reference(inv, min_col=3, min_row=1, max_row=12, max_col=3)
    area_cats = Reference(inv, min_col=1, min_row=2, max_row=12)
    area.add_data(area_data, titles_from_data=True)
    area.set_categories(area_cats)
    area.height = 9
    area.width = 18
    dash.add_chart(area, "D24")

    # ------------------------------------------------------------- README sheet
    readme = wb.create_sheet("README")
    readme["A1"] = "This workbook is generated for benchmarking Excel extraction libraries."
    readme["A2"] = "It contains: Sales, Inventory, Quarterly P&L, Dashboard, and this README."
    readme["A3"] = "Charts: bar, line, pie, scatter, area."
    readme["A4"] = "Features: formulas, merged cells, conditional formatting, hyperlinks, comments, tables, named ranges, frozen panes, dates, booleans, errors."

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
