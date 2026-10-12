"""
Generate visualizations from benchmark results.
Creates charts for the README.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "benchmark_results.json"
OUTPUT_DIR = BASE_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    "PyMuPDF": "#2C3E50",
    "pdfplumber": "#E74C3C",
    "pypdf": "#3498DB",
    "pdfminer.six": "#27AE60",
    "camelot-lattice": "#9B59B6",
    "camelot-stream": "#F39C12",
}

LIGHT_COLORS = {
    "PyMuPDF": "#85929E",
    "pdfplumber": "#F1948A",
    "pypdf": "#85C1E9",
    "pdfminer.six": "#82E0AA",
    "camelot-lattice": "#C39BD3",
    "camelot-stream": "#FAD7A0",
}


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def make_radar_chart(categories, values_dict, title, filename):
    """Create a radar/spider chart comparing multiple extractors."""
    num_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    for name, values in values_dict.items():
        vals = values + values[:1]
        color = COLORS.get(name, '#555555')
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=name, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8, color='grey')
    ax.grid(color='grey', alpha=0.3)

    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_bar_chart_grouped(labels, data_dict, title, ylabel, filename, colors_map=None):
    """Create a grouped bar chart."""
    x = np.arange(len(labels))
    n_groups = len(data_dict)
    width = 0.75 / n_groups
    offsets = np.linspace(-(n_groups - 1) / 2 * width, (n_groups - 1) / 2 * width, n_groups)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')

    for i, (name, values) in enumerate(data_dict.items()):
        color = (colors_map or COLORS).get(name, '#888888')
        bars = ax.bar(x + offsets[i], values, width, label=name, color=color,
                      alpha=0.9, edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('PDF Document', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.axhline(y=0.8, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_speed_comparison(results, filename):
    """Log-scale speed comparison chart."""
    text_ext = results["text_extraction"]
    table_ext = results["table_extraction"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    fig.suptitle('Processing Speed Comparison (per PDF, seconds)', fontsize=15, fontweight='bold', y=1.01)

    for ax, (ext_results, title, use_log) in zip(
        axes,
        [(text_ext, "Text Extraction Speed", True),
         (table_ext, "Table Extraction Speed", True)]
    ):
        ax.set_facecolor('#F8F9FA')
        names = list(ext_results.keys())
        pdfs = [r["pdf"] for r in list(ext_results.values())[0]]

        x = np.arange(len(pdfs))
        n = len(names)
        width = 0.7 / n
        offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

        for i, name in enumerate(names):
            speeds = [r["elapsed"] for r in ext_results[name]]
            color = COLORS.get(name, '#888888')
            ax.bar(x + offsets[i], speeds, width, label=name, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

        if use_log:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('PDF Document', fontsize=10)
        ax.set_ylabel('Time (seconds, log scale)' if use_log else 'Time (seconds)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_heatmap(data_matrix, row_labels, col_labels, title, filename,
                 colormap='YlGnBu', vmin=0, vmax=1):
    """Create a heatmap of scores."""
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.2),
                                     max(5, len(row_labels) * 0.8)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(data_matrix, cmap=colormap, vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='Score (0-1)')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([c.replace('_', '\n') for c in col_labels], fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data_matrix[i][j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_summary_dashboard(results, filename):
    """Create a comprehensive dashboard showing all key metrics."""
    text_summary = results["text_summary"]
    table_summary = results["table_summary"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle('PDF Extraction Library Benchmark Dashboard', fontsize=18,
                 fontweight='bold', y=0.98)

    # --- 1. Overall Text Score Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    names = list(text_summary.keys())
    scores = [text_summary[n].get('avg_overall', 0) for n in names]
    colors = [COLORS.get(n, '#888888') for n in names]
    bars = ax1.barh(names, scores, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlim(0, 1.05)
    ax1.set_title('Text Extraction\nOverall Score', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Score (0-1)', fontsize=9)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_facecolor('#F8F9FA')

    # --- 2. Keyword Recall Bar Chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    recalls = [text_summary[n].get('avg_keyword_recall', 0) for n in names]
    bars2 = ax2.barh(names, recalls, color=colors, edgecolor='white', height=0.6)
    ax2.set_xlim(0, 1.05)
    ax2.set_title('Keyword Recall\n(Text)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Recall (0-1)', fontsize=9)
    for bar, score in zip(bars2, recalls):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_facecolor('#F8F9FA')

    # --- 3. Text Speed Bar Chart ---
    ax3 = fig.add_subplot(gs[0, 2])
    speeds = [text_summary[n].get('avg_speed', 0) for n in names]
    bars3 = ax3.barh(names, speeds, color=colors, edgecolor='white', height=0.6)
    ax3.set_title('Text Extraction\nAvg Speed (s/PDF)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontsize=9)
    for bar, speed in zip(bars3, speeds):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{speed:.2f}s', va='center', fontsize=9, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_facecolor('#F8F9FA')

    # --- 4. Table Detection Rate ---
    ax4 = fig.add_subplot(gs[1, 0])
    t_names = list(table_summary.keys())
    t_colors = [COLORS.get(n, '#888888') for n in t_names]
    det_rates = [table_summary[n].get('avg_detection_rate', 0) for n in t_names]
    bars4 = ax4.barh(t_names, det_rates, color=t_colors, edgecolor='white', height=0.6)
    ax4.set_xlim(0, 1.05)
    ax4.set_title('Table Detection Rate', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Detection Rate (0-1)', fontsize=9)
    for bar, score in zip(bars4, det_rates):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_facecolor('#F8F9FA')

    # --- 5. Table Cell Accuracy ---
    ax5 = fig.add_subplot(gs[1, 1])
    cell_acc = [table_summary[n].get('avg_cell_accuracy', 0) for n in t_names]
    bars5 = ax5.barh(t_names, cell_acc, color=t_colors, edgecolor='white', height=0.6)
    ax5.set_xlim(0, 1.05)
    ax5.set_title('Table Cell Accuracy', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Accuracy (0-1)', fontsize=9)
    for bar, score in zip(bars5, cell_acc):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    ax5.set_facecolor('#F8F9FA')

    # --- 6. Table Speed ---
    ax6 = fig.add_subplot(gs[1, 2])
    t_speeds = [table_summary[n].get('avg_speed', 0) for n in t_names]
    bars6 = ax6.barh(t_names, t_speeds, color=t_colors, edgecolor='white', height=0.6)
    ax6.set_title('Table Extraction\nAvg Speed (s/PDF)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (seconds)', fontsize=9)
    for bar, speed in zip(bars6, t_speeds):
        ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{speed:.2f}s', va='center', fontsize=9, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    ax6.set_facecolor('#F8F9FA')

    # --- 7. Per-PDF keyword recall heatmap (text) ---
    ax7 = fig.add_subplot(gs[2, :2])
    text_ext = results["text_extraction"]
    pdfs = [r["pdf"] for r in list(text_ext.values())[0]]
    text_names_list = list(text_ext.keys())
    matrix = np.zeros((len(text_names_list), len(pdfs)))
    for i, name in enumerate(text_names_list):
        for j, r in enumerate(text_ext[name]):
            matrix[i][j] = r.get("keyword_recall", 0)

    im = ax7.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax7, label='Keyword Recall')
    ax7.set_xticks(range(len(pdfs)))
    ax7.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=8, rotation=45, ha='right')
    ax7.set_yticks(range(len(text_names_list)))
    ax7.set_yticklabels(text_names_list, fontsize=10, fontweight='bold')
    for i in range(len(text_names_list)):
        for j in range(len(pdfs)):
            val = matrix[i][j]
            ax7.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if val < 0.5 else 'black')
    ax7.set_title('Keyword Recall by PDF (Text Extractors)', fontsize=11, fontweight='bold')
    ax7.set_facecolor('#F8F9FA')

    # --- 8. Speed vs Quality scatter ---
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor('#F8F9FA')
    ax8.set_title('Speed vs Quality\n(Text Extractors)', fontsize=11, fontweight='bold')

    for name in text_summary:
        s = text_summary[name]
        speed = s.get('avg_speed', 0)
        quality = s.get('avg_overall', 0)
        color = COLORS.get(name, '#888')
        ax8.scatter(speed, quality, s=200, color=color, zorder=5, edgecolor='white', linewidth=1.5)
        ax8.annotate(name, (speed, quality), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color=color, fontweight='bold')

    ax8.set_xlabel('Avg Speed (s/PDF)', fontsize=9)
    ax8.set_ylabel('Overall Quality Score', fontsize=9)
    ax8.set_xlim(-0.1)
    ax8.set_ylim(0.9, 1.01)
    ax8.grid(alpha=0.3)
    # Add "sweet spot" annotation
    ax8.axhline(y=0.95, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax8.text(max(s.get('avg_speed', 0) for s in text_summary.values()) * 0.5,
             0.952, 'Target quality', fontsize=7, color='green', alpha=0.7)

    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_per_pdf_breakdown(results, filename):
    """Per-PDF breakdown chart for keyword recall."""
    text_ext = results["text_extraction"]
    pdfs = [r["pdf"] for r in list(text_ext.values())[0]]
    # Group by category
    cats = {}
    for r in list(text_ext.values())[0]:
        cats[r["pdf"]] = r.get("category", "unknown")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Per-PDF Performance Breakdown', fontsize=15, fontweight='bold')

    # Top: keyword recall per PDF per extractor
    ax = axes[0]
    ax.set_facecolor('#F8F9FA')
    x = np.arange(len(pdfs))
    n = len(text_ext)
    width = 0.7 / n
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    for i, (name, records) in enumerate(text_ext.items()):
        recalls = [r.get("keyword_recall", 0) for r in records]
        color = COLORS.get(name, '#888')
        bars = ax.bar(x + offsets[i], recalls, width, label=name, color=color,
                      alpha=0.85, edgecolor='white')

    ax.set_title('Keyword Recall per PDF', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=8)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc='lower right')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(len(pdfs) - 1, 0.81, 'Min threshold (80%)', fontsize=8, color='red', alpha=0.7, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Bottom: processing time per PDF
    ax2 = axes[1]
    ax2.set_facecolor('#F8F9FA')
    for i, (name, records) in enumerate(text_ext.items()):
        times = [r.get("elapsed", 0) for r in records]
        color = COLORS.get(name, '#888')
        ax2.bar(x + offsets[i], times, width, label=name, color=color,
                alpha=0.85, edgecolor='white')

    ax2.set_title('Processing Time per PDF (seconds)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=8)
    ax2.set_ylabel('Seconds', fontsize=10)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Add category labels
    for j, pdf in enumerate(pdfs):
        cat = cats.get(pdf, "")
        if cat:
            ax.text(j, -0.12, cat.replace('_', '\n'), ha='center', fontsize=6,
                    color='grey', transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_table_comparison_chart(results, filename):
    """Detailed table extraction comparison."""
    table_ext = results["table_extraction"]
    pdfs = [r["pdf"] for r in list(table_ext.values())[0]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Table Extraction Detailed Comparison', fontsize=15, fontweight='bold')

    names = list(table_ext.keys())
    x = np.arange(len(pdfs))
    n = len(names)
    width = 0.7 / n
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    # Plot 1: Tables found per PDF
    ax = axes[0, 0]
    ax.set_facecolor('#F8F9FA')
    for i, name in enumerate(names):
        counts = [r.get("tables_found", 0) for r in table_ext[name]]
        color = COLORS.get(name, '#888')
        ax.bar(x + offsets[i], counts, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax.set_title('Tables Found per PDF', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=7)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Detection rate
    ax2 = axes[0, 1]
    ax2.set_facecolor('#F8F9FA')
    for i, name in enumerate(names):
        rates = [r.get("detection_rate", 0) for r in table_ext[name]]
        color = COLORS.get(name, '#888')
        ax2.bar(x + offsets[i], rates, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax2.set_title('Table Detection Rate per PDF', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=7)
    ax2.set_ylabel('Rate (0-1)', fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Cell accuracy
    ax3 = axes[1, 0]
    ax3.set_facecolor('#F8F9FA')
    for i, name in enumerate(names):
        accs = [r.get("cell_accuracy", 0) for r in table_ext[name]]
        color = COLORS.get(name, '#888')
        ax3.bar(x + offsets[i], accs, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax3.set_title('Cell Content Accuracy', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=7)
    ax3.set_ylabel('Accuracy (0-1)', fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Processing time
    ax4 = axes[1, 1]
    ax4.set_facecolor('#F8F9FA')
    for i, name in enumerate(names):
        times = [r.get("elapsed", 0) for r in table_ext[name]]
        color = COLORS.get(name, '#888')
        ax4.bar(x + offsets[i], times, width, label=name, color=color, alpha=0.85, edgecolor='white')
    ax4.set_title('Processing Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([p.replace('_', '\n') for p in pdfs], fontsize=7)
    ax4.set_ylabel('Seconds (log scale)', fontsize=10)
    ax4.set_yscale('log')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def make_winner_scorecard(results, filename):
    """Create a final scorecard showing overall winner per category."""
    text_summary = results["text_summary"]
    table_summary = results["table_summary"]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Build scorecard data
    categories = [
        "Overall\nText Score",
        "Keyword\nRecall",
        "Number\nPreservation",
        "Text\nSpeed",
        "Table\nDetection",
        "Cell\nAccuracy",
        "Table\nSpeed",
    ]

    # Text extractors
    text_libs = list(text_summary.keys())
    table_libs = list(table_summary.keys())
    all_libs = list(dict.fromkeys(text_libs + table_libs))

    rows_data = []
    for lib in all_libs:
        row = [lib]
        if lib in text_summary:
            ts = text_summary[lib]
            row.extend([
                f"{ts.get('avg_overall', 0):.3f}",
                f"{ts.get('avg_keyword_recall', 0):.3f}",
                f"{ts.get('avg_number_preservation', 0):.3f}",
                f"{ts.get('avg_speed', 0):.2f}s",
            ])
        else:
            row.extend(["N/A", "N/A", "N/A", "N/A"])

        if lib in table_summary:
            ts2 = table_summary[lib]
            row.extend([
                f"{ts2.get('avg_detection_rate', 0):.3f}",
                f"{ts2.get('avg_cell_accuracy', 0):.3f}",
                f"{ts2.get('avg_speed', 0):.2f}s",
            ])
        else:
            row.extend(["N/A", "N/A", "N/A"])

        rows_data.append(row)

    col_labels = ["Library"] + categories
    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color rows
    row_colors = list(COLORS.values())
    for i in range(len(rows_data)):
        lib_name = rows_data[i][0]
        base_color = COLORS.get(lib_name, '#CCCCCC')
        table[(i + 1, 0)].set_facecolor(base_color)
        table[(i + 1, 0)].set_text_props(color='white', fontweight='bold')

        # Color numeric cells based on value
        for j in range(1, len(col_labels)):
            val_str = rows_data[i][j]
            if val_str != "N/A":
                try:
                    val = float(val_str.rstrip('s'))
                    is_speed = 's' in val_str
                    if is_speed:
                        # Lower is better for speed
                        max_speed = 15.0
                        ratio = 1.0 - min(val / max_speed, 1.0)
                    else:
                        ratio = val
                    # Green -> Red based on ratio
                    if ratio >= 0.95:
                        cell_color = '#A9DFBF'
                    elif ratio >= 0.80:
                        cell_color = '#D5F5E3'
                    elif ratio >= 0.60:
                        cell_color = '#FDEBD0'
                    else:
                        cell_color = '#FADBD8'
                    table[(i + 1, j)].set_facecolor(cell_color)
                except ValueError:
                    pass

    ax.set_title('PDF Extraction Library Benchmark Scorecard', fontsize=16,
                 fontweight='bold', pad=20, y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    print("Generating benchmark visualizations...")
    results = load_results()

    # 1. Summary dashboard (key overview)
    print("\n1. Summary Dashboard:")
    make_summary_dashboard(results, "01_summary_dashboard.png")

    # 2. Per-PDF breakdown
    print("\n2. Per-PDF Breakdown:")
    make_per_pdf_breakdown(results, "02_per_pdf_breakdown.png")

    # 3. Table extraction detailed
    print("\n3. Table Extraction Charts:")
    make_table_comparison_chart(results, "03_table_extraction_detail.png")

    # 4. Speed comparison
    print("\n4. Speed Comparison:")
    make_speed_comparison(results, "04_speed_comparison.png")

    # 5. Radar chart for text extractors
    print("\n5. Radar Charts:")
    text_summary = results["text_summary"]
    radar_values = {}
    for name, s in text_summary.items():
        # Normalize speed: 0-1 where 1=fastest
        max_speed = max(v.get('avg_speed', 0.001) for v in text_summary.values())
        speed_score = 1.0 - (s.get('avg_speed', 0) / max_speed)
        radar_values[name] = [
            s.get('avg_keyword_recall', 0),
            s.get('avg_number_preservation', 0),
            s.get('avg_printable_ratio', 0),
            speed_score,
        ]

    make_radar_chart(
        ["Keyword\nRecall", "Number\nPreservation", "Text\nQuality", "Speed\n(relative)"],
        radar_values,
        "Text Extractor Radar Chart",
        "05_radar_text.png"
    )

    # 6. Radar chart for table extractors
    table_summary = results["table_summary"]
    radar_table = {}
    for name, s in table_summary.items():
        max_speed = max(v.get('avg_speed', 0.001) for v in table_summary.values())
        speed_score = 1.0 - (s.get('avg_speed', 0) / max_speed)
        radar_table[name] = [
            s.get('avg_detection_rate', 0),
            s.get('avg_cell_accuracy', 0),
            speed_score,
        ]

    make_radar_chart(
        ["Table\nDetection", "Cell\nAccuracy", "Speed\n(relative)"],
        radar_table,
        "Table Extractor Radar Chart",
        "06_radar_tables.png"
    )

    # 7. Winner scorecard
    print("\n7. Winner Scorecard:")
    make_winner_scorecard(results, "07_scorecard.png")

    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
    return [str(p) for p in sorted(OUTPUT_DIR.glob("*.png"))]


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    charts = main()
    print(f"\nGenerated {len(charts)} charts:")
    for c in charts:
        print(f"  {c}")
