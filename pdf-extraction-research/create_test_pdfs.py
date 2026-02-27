"""
Create test PDFs that represent real-world content:
- Tables with various complexity
- Mixed text + images
- Multi-column layouts
- Headers/footers
- Mathematical formulas
- Nested tables

These PDFs come with ground truth data for evaluation.
"""

import os
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image,
    HRFlowable, PageBreak
)
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import io

OUTPUT_DIR = "test_pdfs"
GROUND_TRUTH_DIR = "ground_truth"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)


def create_simple_table_pdf():
    """PDF with a simple financial table - clear ground truth."""
    filename = f"{OUTPUT_DIR}/simple_table.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Q3 2024 Financial Summary Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3 * inch))

    intro = Paragraph(
        "This report summarizes key financial metrics for Q3 2024. "
        "All figures are in thousands of USD unless otherwise noted.",
        styles['Normal']
    )
    story.append(intro)
    story.append(Spacer(1, 0.2 * inch))

    # Table data
    table_data = [
        ['Product', 'Q1 2024', 'Q2 2024', 'Q3 2024', 'YoY Growth'],
        ['Widget A', '$12,450', '$13,200', '$15,800', '+26.9%'],
        ['Widget B', '$8,320', '$9,100', '$11,200', '+34.6%'],
        ['Service Pack', '$5,600', '$6,800', '$7,200', '+28.6%'],
        ['Support', '$3,200', '$3,400', '$3,100', '-3.1%'],
        ['Total', '$29,570', '$32,500', '$37,300', '+26.1%'],
    ]

    table = Table(table_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#F8F9FA')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    conclusion = Paragraph(
        "Revenue grew by 26.1% year-over-year, driven primarily by strong performance "
        "in Widget B (+34.6%) and Widget A (+26.9%). Support revenue declined slightly "
        "by 3.1%, which management attributes to improved product quality reducing "
        "support ticket volumes.",
        styles['Normal']
    )
    story.append(conclusion)

    doc.build(story)

    # Ground truth
    ground_truth = {
        "filename": "simple_table.pdf",
        "description": "Simple financial table with 5 columns and 6 rows",
        "tables": [
            {
                "rows": 6,
                "cols": 5,
                "header": ["Product", "Q1 2024", "Q2 2024", "Q3 2024", "YoY Growth"],
                "data": table_data[1:],
                "total_cells": 30
            }
        ],
        "text_sections": [
            "Q3 2024 Financial Summary Report",
            "This report summarizes key financial metrics for Q3 2024.",
            "All figures are in thousands of USD unless otherwise noted.",
            "Revenue grew by 26.1% year-over-year"
        ],
        "key_numbers": ["$15,800", "$11,200", "$37,300", "26.1%", "34.6%"],
        "images": 0
    }
    with open(f"{GROUND_TRUTH_DIR}/simple_table.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created: {filename}")
    return ground_truth


def create_complex_table_pdf():
    """PDF with merged cells, nested tables, complex layout."""
    filename = f"{OUTPUT_DIR}/complex_table.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Annual Performance Review Matrix", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    section_hdr = Paragraph("Department Performance Scorecard", styles['Heading2'])
    story.append(section_hdr)

    # Complex table with merged cells (using SPAN)
    data = [
        ['Department', 'H1 Performance', '', 'H2 Performance', '', 'Annual'],
        ['', 'Sales', 'Ops', 'Sales', 'Ops', 'Total'],
        ['Engineering', '87', '92', '91', '88', '89.5'],
        ['Marketing', '76', '84', '82', '79', '80.3'],
        ['Finance', '91', '95', '94', '96', '94.0'],
        ['HR', '88', '90', '85', '91', '88.5'],
        ['Average', '85.5', '90.3', '88.0', '88.5', '88.1'],
    ]

    table = Table(data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        # Merge header cells
        ('SPAN', (1, 0), (2, 0)),  # H1 Performance spans 2 cols
        ('SPAN', (3, 0), (4, 0)),  # H2 Performance spans 2 cols
        ('SPAN', (0, 0), (0, 1)),  # Department spans 2 rows
        ('SPAN', (5, 0), (5, 1)),  # Annual spans 2 rows
        ('BACKGROUND', (0, 0), (-1, 1), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 1), colors.white),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#85C1E9')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 2), (-1, -2), [colors.white, colors.HexColor('#EBF5FB')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # Second table - comparison
    story.append(Paragraph("Year-over-Year Comparison", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    yoy_data = [
        ['Metric', '2022', '2023', '2024', 'Delta (23-24)', 'Trend'],
        ['Revenue ($M)', '142.3', '158.7', '187.2', '+$28.5M', '↑ 18.0%'],
        ['Customers', '2,341', '2,789', '3,412', '+623', '↑ 22.3%'],
        ['NPS Score', '42', '51', '67', '+16 pts', '↑ 31.4%'],
        ['Employee Count', '234', '298', '341', '+43', '↑ 14.4%'],
        ['Churn Rate', '8.2%', '6.7%', '4.1%', '-2.6 pts', '↓ 38.8%'],
        ['Avg Deal Size ($K)', '60.8', '56.9', '54.9', '-$2K', '↓ 3.5%'],
    ]

    table2 = Table(yoy_data, colWidths=[1.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EAF2FF')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table2)

    doc.build(story)

    ground_truth = {
        "filename": "complex_table.pdf",
        "description": "Complex table with merged cells + second comparison table",
        "tables": [
            {
                "rows": 7,
                "cols": 6,
                "merged_cells": True,
                "header": ["Department", "H1 Performance", "H2 Performance", "Annual"],
                "description": "Performance scorecard with merged header cells"
            },
            {
                "rows": 7,
                "cols": 6,
                "header": ["Metric", "2022", "2023", "2024", "Delta (23-24)", "Trend"],
                "key_values": {
                    "Revenue 2024": "$187.2M",
                    "NPS 2024": "67",
                    "Churn 2024": "4.1%"
                }
            }
        ],
        "text_sections": [
            "Annual Performance Review Matrix",
            "Department Performance Scorecard",
            "Year-over-Year Comparison"
        ],
        "images": 0
    }
    with open(f"{GROUND_TRUTH_DIR}/complex_table.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created: {filename}")
    return ground_truth


def create_image_placeholder(width, height, color, label):
    """Create a simple colored rectangle as image placeholder."""
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    from reportlab.lib.colors import HexColor
    import io

    # We'll just use a simple colored rectangle via Table for simplicity
    return None


def create_mixed_content_pdf():
    """PDF with text, tables and image placeholders."""
    filename = f"{OUTPUT_DIR}/mixed_content.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title and abstract
    story.append(Paragraph("Machine Learning in Healthcare: A Review", styles['Title']))
    story.append(Spacer(1, 0.15 * inch))

    authors = Paragraph(
        "<b>Authors:</b> Smith, J., Johnson, A., Williams, R., Brown, M.",
        styles['Normal']
    )
    story.append(authors)
    story.append(Spacer(1, 0.1 * inch))

    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'],
                                    leftIndent=30, rightIndent=30,
                                    borderPad=5)
    story.append(Paragraph("<b>Abstract</b>", styles['Heading3']))
    story.append(Paragraph(
        "This paper reviews recent advances in machine learning applications for healthcare, "
        "focusing on diagnostic imaging, clinical decision support, and drug discovery. "
        "We analyze 127 peer-reviewed studies published between 2020-2024 and identify "
        "key trends, challenges, and future directions. Our meta-analysis reveals that "
        "deep learning models achieve an average AUC of 0.91 across diagnostic tasks, "
        "outperforming traditional methods by 12.3 percentage points.",
        abstract_style
    ))
    story.append(Spacer(1, 0.2 * inch))

    # Section 1
    story.append(Paragraph("1. Introduction", styles['Heading1']))
    story.append(Paragraph(
        "The integration of machine learning (ML) into clinical practice has accelerated "
        "dramatically over the past decade. Electronic health records, medical imaging, "
        "genomic data, and wearable sensors generate vast amounts of data that traditional "
        "analytical methods struggle to process effectively.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Image placeholder (using colored table)
    story.append(Paragraph("<b>Figure 1:</b> Growth of ML Publications in Healthcare (2015-2024)", styles['Normal']))
    # Simulate figure with colored table
    fig_data = [['Year', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
                 ['Papers', '234', '312', '445', '623', '891', '1243', '1687', '2134', '2891', '3412']]
    fig_table = Table(fig_data)
    fig_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980B9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(fig_table)
    story.append(Spacer(1, 0.2 * inch))

    # Section 2 with stats table
    story.append(Paragraph("2. Performance Comparison", styles['Heading1']))
    story.append(Paragraph(
        "Table 1 summarizes performance metrics across major ML approaches in healthcare "
        "diagnostics. Deep learning consistently outperforms classical methods.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1 * inch))

    perf_data = [
        ['Method', 'Accuracy', 'AUC-ROC', 'Sensitivity', 'Specificity', 'F1 Score'],
        ['CNN (ResNet-50)', '94.2%', '0.97', '93.8%', '94.6%', '0.94'],
        ['Vision Transformer', '95.1%', '0.98', '94.7%', '95.5%', '0.95'],
        ['Random Forest', '87.3%', '0.91', '86.1%', '88.5%', '0.87'],
        ['SVM (RBF)', '85.6%', '0.89', '84.9%', '86.3%', '0.86'],
        ['Logistic Regression', '79.2%', '0.84', '78.1%', '80.3%', '0.79'],
        ['XGBoost', '89.7%', '0.93', '88.9%', '90.5%', '0.90'],
    ]

    perf_table = Table(perf_data, colWidths=[1.5*inch, 0.9*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.8*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E8449')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EAFAF1')]),
        # Highlight best row
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#A9DFBF')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("<b>Table 1:</b> ML Method Performance Comparison", styles['Normal']))
    story.append(Spacer(1, 0.05 * inch))
    story.append(perf_table)
    story.append(Spacer(1, 0.2 * inch))

    # Section 3
    story.append(Paragraph("3. Conclusion", styles['Heading1']))
    story.append(Paragraph(
        "Our analysis demonstrates that machine learning, particularly deep learning approaches, "
        "offer significant improvements over traditional diagnostic methods. Vision Transformers "
        "achieved the highest accuracy (95.1%) and AUC-ROC (0.98), followed closely by CNN "
        "architectures. Key challenges remain around data privacy, model interpretability, "
        "and regulatory approval.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15 * inch))

    # References
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("References", styles['Heading3']))
    refs = [
        "[1] LeCun, Y. et al. (2015). Deep learning. Nature, 521(7553), 436-444.",
        "[2] Rajpurkar, P. et al. (2022). AI in health and medicine. Nature Medicine, 28, 31-38.",
        "[3] Topol, E.J. (2019). High-performance medicine: the convergence of human and AI. Nature Medicine.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['Normal']))

    doc.build(story)

    ground_truth = {
        "filename": "mixed_content.pdf",
        "description": "Academic paper with mixed content: text, tables, figure caption",
        "tables": [
            {
                "id": "figure_table",
                "rows": 2,
                "cols": 11,
                "content": "Publication counts by year 2015-2024"
            },
            {
                "id": "performance_table",
                "rows": 7,
                "cols": 6,
                "header": ["Method", "Accuracy", "AUC-ROC", "Sensitivity", "Specificity", "F1 Score"],
                "best_method": "Vision Transformer",
                "best_accuracy": "95.1%",
                "best_auc": "0.98"
            }
        ],
        "sections": ["Abstract", "1. Introduction", "2. Performance Comparison", "3. Conclusion", "References"],
        "key_facts": {
            "avg_auc": "0.91",
            "improvement": "12.3 percentage points",
            "studies_analyzed": "127",
            "best_accuracy": "95.1%"
        },
        "images": 0,
        "references": 3
    }
    with open(f"{GROUND_TRUTH_DIR}/mixed_content.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created: {filename}")
    return ground_truth


def create_multicolumn_pdf():
    """PDF with multi-column layout, harder to parse."""
    filename = f"{OUTPUT_DIR}/multicolumn.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=0.5*inch, leftMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("The Future of Renewable Energy", styles['Title']))
    story.append(Paragraph("Technology Review | Volume 42, Issue 7 | 2024", styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1))
    story.append(Spacer(1, 0.1 * inch))

    # Simulate two-column via table
    col_text_1 = """
    Solar and wind energy capacity has surged globally, with installed capacity
    reaching 2.5 terawatts in 2023, a 15% increase from the previous year.
    Investment in renewable energy exceeded $500 billion for the first time,
    driven by government incentives and falling technology costs.
    The levelized cost of electricity (LCOE) for utility-scale solar has dropped
    to $0.033/kWh, making it the cheapest electricity source in history.
    """

    col_text_2 = """
    Battery storage technology has kept pace with renewable expansion.
    Global battery storage capacity reached 85 GWh by end of 2023.
    Lithium iron phosphate (LFP) batteries now dominate the market at 62%
    share, valued for safety and long cycle life exceeding 4,000 cycles.
    Grid-scale projects increasingly combine solar/wind with storage to
    provide firm, dispatchable renewable power around the clock.
    """

    two_col = Table([[
        Paragraph(col_text_1, styles['Normal']),
        Paragraph(col_text_2, styles['Normal'])
    ]], colWidths=[3.5*inch, 3.5*inch])
    two_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 0.2 * inch))

    # Stats table
    story.append(Paragraph("Global Renewable Energy Statistics 2024", styles['Heading2']))
    stats_data = [
        ['Energy Source', 'Capacity (GW)', 'Growth YoY', 'LCOE ($/kWh)', 'Jobs Created'],
        ['Solar PV', '1,419', '+28%', '$0.033', '4.2M'],
        ['Wind (Onshore)', '892', '+12%', '$0.041', '1.4M'],
        ['Wind (Offshore)', '72', '+24%', '$0.081', '0.3M'],
        ['Hydropower', '1,392', '+2%', '$0.048', '2.0M'],
        ['Geothermal', '15', '+4%', '$0.071', '0.1M'],
        ['Total Renewable', '3,790', '+16%', '-', '7.9M'],
    ]

    stats_table = Table(stats_data, colWidths=[1.4*inch, 1.1*inch, 1.0*inch, 1.1*inch, 1.1*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E67E22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#FEF9E7')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FDEBD0')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_table)

    doc.build(story)

    ground_truth = {
        "filename": "multicolumn.pdf",
        "description": "Magazine-style with two-column text layout and stats table",
        "layout": "two-column",
        "tables": [
            {
                "rows": 7,
                "cols": 5,
                "header": ["Energy Source", "Capacity (GW)", "Growth YoY", "LCOE ($/kWh)", "Jobs Created"],
                "key_values": {
                    "Solar PV capacity": "1,419 GW",
                    "Total renewable": "3,790 GW",
                    "Solar LCOE": "$0.033/kWh"
                }
            }
        ],
        "key_facts": {
            "total_capacity_gw": "2,500+ TW equivalent",
            "investment_usd": "$500 billion",
            "battery_storage_gwh": "85 GWh",
            "solar_lcoe": "$0.033/kWh"
        },
        "images": 0
    }
    with open(f"{GROUND_TRUTH_DIR}/multicolumn.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created: {filename}")
    return ground_truth


def create_text_heavy_pdf():
    """Text-heavy PDF to test text extraction accuracy."""
    filename = f"{OUTPUT_DIR}/text_heavy.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Understanding Large Language Models", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    sections = [
        ("1. What are Large Language Models?",
         "Large Language Models (LLMs) are a type of artificial intelligence system trained on "
         "massive datasets of text to understand and generate human language. These models use "
         "transformer architectures with billions of parameters, enabling them to perform tasks "
         "ranging from text summarization and translation to code generation and question answering. "
         "The most notable examples include GPT-4, Claude, Gemini, and Llama, each representing "
         "different approaches to training and deployment."),

        ("2. Training Methodology",
         "LLMs are typically trained using a two-phase approach. The first phase, pre-training, "
         "involves processing hundreds of billions of tokens from web crawls, books, code repositories, "
         "and academic papers. The model learns to predict the next token in a sequence, gradually "
         "developing representations of language structure, factual knowledge, and reasoning patterns. "
         "The second phase, fine-tuning with RLHF (Reinforcement Learning from Human Feedback), "
         "aligns the model with human preferences for helpfulness, harmlessness, and honesty."),

        ("3. Capabilities and Limitations",
         "Modern LLMs demonstrate remarkable capabilities including: multi-step reasoning, code "
         "generation and debugging, creative writing, mathematical problem-solving, and cross-lingual "
         "understanding. However, they also exhibit characteristic limitations: hallucination of "
         "false information, difficulty with precise counting and arithmetic, sensitivity to prompt "
         "phrasing, potential biases from training data, and a knowledge cutoff date. Understanding "
         "these trade-offs is crucial for effective deployment in production systems."),

        ("4. Context Windows and Memory",
         "A critical parameter of LLMs is the context window - the maximum number of tokens the "
         "model can process in a single inference. Earlier models were limited to 2,048 tokens, "
         "while current models support 100,000 to 1,000,000 tokens. Larger context windows enable "
         "processing of entire codebases, legal documents, or books in a single pass. However, "
         "research shows that model attention degrades for content in the middle of very long contexts, "
         "a phenomenon called the 'lost in the middle' problem, requiring careful prompt engineering."),

        ("5. Practical Applications in Enterprise",
         "Enterprise adoption of LLMs has accelerated through 2024, with use cases spanning: "
         "customer service automation (reducing ticket resolution time by 40-60%), software development "
         "acceleration (GitHub Copilot reports 55% faster task completion), document analysis and "
         "summarization (processing thousands of legal or financial documents per hour), and "
         "knowledge management systems that answer employee questions from internal documentation. "
         "The key success factors are robust prompt engineering, retrieval-augmented generation (RAG) "
         "for current information, and human oversight for critical decisions."),
    ]

    for heading, content in sections:
        story.append(Paragraph(heading, styles['Heading1']))
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph(content, styles['Normal']))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)

    ground_truth = {
        "filename": "text_heavy.pdf",
        "description": "Text-heavy academic document with 5 sections",
        "sections": [s[0] for s in sections],
        "key_phrases": [
            "Large Language Models",
            "transformer architectures",
            "Reinforcement Learning from Human Feedback",
            "hallucination",
            "context window",
            "retrieval-augmented generation",
            "100,000 to 1,000,000 tokens",
            "40-60%",
            "55% faster"
        ],
        "tables": [],
        "images": 0,
        "word_count_approx": 550
    }
    with open(f"{GROUND_TRUTH_DIR}/text_heavy.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created: {filename}")
    return ground_truth


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Creating test PDFs with ground truth...")
    create_simple_table_pdf()
    create_complex_table_pdf()
    create_mixed_content_pdf()
    create_multicolumn_pdf()
    create_text_heavy_pdf()
    print("\nAll test PDFs created successfully!")
    print(f"PDFs saved to: {OUTPUT_DIR}/")
    print(f"Ground truth saved to: {GROUND_TRUTH_DIR}/")
