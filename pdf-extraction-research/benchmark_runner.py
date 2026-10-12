"""
Comprehensive PDF extraction benchmark runner.

Tests all extractors against:
1. Our custom test PDFs (with ground truth)
2. Real-world arXiv papers (complex layouts, figures, tables, math)

Metrics collected:
- Text extraction accuracy (exact match of key phrases)
- Character-level similarity
- Table detection rate
- Processing speed (seconds)
- Error rate
"""

import os
import sys
import json
import time
import re
import warnings
import traceback
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
TEST_PDFS_DIR = BASE_DIR / "test_pdfs"


# ============================================================
# EXTRACTOR IMPLEMENTATIONS
# ============================================================

def extract_pymupdf_text(pdf_path: str) -> Tuple[str, float, Optional[str]]:
    """Extract text using PyMuPDF."""
    start = time.time()
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text, time.time() - start, None
    except Exception as e:
        return "", time.time() - start, str(e)


def extract_pymupdf_tables(pdf_path: str) -> Tuple[List, float, Optional[str]]:
    """Extract tables using PyMuPDF's built-in table finder."""
    start = time.time()
    try:
        import fitz
        doc = fitz.open(pdf_path)
        tables = []
        for page in doc:
            found = page.find_tables()
            for t in found:
                data = t.extract()
                if data:
                    tables.append(data)
        doc.close()
        return tables, time.time() - start, None
    except Exception as e:
        return [], time.time() - start, str(e)


def extract_pdfplumber_text(pdf_path: str) -> Tuple[str, float, Optional[str]]:
    """Extract text using pdfplumber."""
    start = time.time()
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts), time.time() - start, None
    except Exception as e:
        return "", time.time() - start, str(e)


def extract_pdfplumber_tables(pdf_path: str) -> Tuple[List, float, Optional[str]]:
    """Extract tables using pdfplumber."""
    start = time.time()
    try:
        import pdfplumber
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        return all_tables, time.time() - start, None
    except Exception as e:
        return [], time.time() - start, str(e)


def extract_pypdf_text(pdf_path: str) -> Tuple[str, float, Optional[str]]:
    """Extract text using pypdf."""
    start = time.time()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text, time.time() - start, None
    except Exception as e:
        return "", time.time() - start, str(e)


def extract_pdfminer_text(pdf_path: str) -> Tuple[str, float, Optional[str]]:
    """Extract text using pdfminer.six."""
    start = time.time()
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        return text or "", time.time() - start, None
    except Exception as e:
        return "", time.time() - start, str(e)


def extract_camelot_tables(pdf_path: str) -> Tuple[List, float, Optional[str]]:
    """Extract tables using camelot-py (lattice mode)."""
    start = time.time()
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        result = []
        for t in tables:
            result.append(t.df.values.tolist())
        return result, time.time() - start, None
    except Exception as e:
        return [], time.time() - start, str(e)


def extract_camelot_stream_tables(pdf_path: str) -> Tuple[List, float, Optional[str]]:
    """Extract tables using camelot-py (stream mode - better for borderless tables)."""
    start = time.time()
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        result = []
        for t in tables:
            result.append(t.df.values.tolist())
        return result, time.time() - start, None
    except Exception as e:
        return [], time.time() - start, str(e)


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def char_similarity(text1: str, text2: str) -> float:
    """Compute character-level similarity between two strings."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def keyword_recall(text: str, keywords: List[str]) -> float:
    """Fraction of keywords found in extracted text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def count_numbers_preserved(text: str, reference_numbers: List[str]) -> float:
    """Check what fraction of specific number strings survived extraction."""
    if not reference_numbers:
        return 1.0
    found = sum(1 for num in reference_numbers if num in text)
    return found / len(reference_numbers)


def assess_text_quality(text: str) -> Dict[str, float]:
    """Assess general quality of extracted text."""
    if not text.strip():
        return {"printable_ratio": 0.0, "word_density": 0.0, "has_content": False}
    printable = sum(1 for c in text if c.isprintable() or c in '\n\t ')
    printable_ratio = printable / max(len(text), 1)
    words = len(text.split())
    chars = len(text)
    word_density = words / max(chars / 5, 1)  # normalize: avg word ~5 chars
    return {
        "printable_ratio": printable_ratio,
        "word_density": min(word_density, 1.0),
        "has_content": len(text.strip()) > 100
    }


def score_table_extraction(tables: List, gt: Dict) -> Dict[str, Any]:
    """Score table extraction quality against ground truth."""
    expected_count = len(gt.get("tables", []))
    found_count = len(tables)

    if expected_count == 0:
        detection_rate = 1.0 if found_count == 0 else 0.5
        return {"detection_rate": detection_rate, "tables_found": found_count, "tables_expected": 0}

    detection_rate = min(found_count / max(expected_count, 1), 1.0)

    # Check data quality if tables were found
    cell_accuracy = 0.0
    if tables:
        all_cell_text = ' '.join(
            str(cell) for table in tables for row in table for cell in (row if row else [])
            if cell is not None
        ).lower()
        # Collect expected values
        expected_vals = []
        for t in gt.get("tables", []):
            if "header" in t:
                expected_vals.extend(t["header"])
            if "key_values" in t:
                expected_vals.extend(t["key_values"].values())
            if "data" in t:
                for row in t["data"]:
                    expected_vals.extend(str(v) for v in row)

        if expected_vals:
            found_expected = sum(1 for v in expected_vals if str(v).lower() in all_cell_text)
            cell_accuracy = found_expected / len(expected_vals)
        else:
            cell_accuracy = 1.0 if tables else 0.0

    return {
        "detection_rate": detection_rate,
        "tables_found": found_count,
        "tables_expected": expected_count,
        "cell_accuracy": cell_accuracy
    }


# ============================================================
# BENCHMARK DEFINITION
# ============================================================

def get_benchmark_cases():
    """Returns all benchmark test cases with metadata."""
    cases = []

    # Custom PDFs with full ground truth
    for gt_file in sorted(GROUND_TRUTH_DIR.glob("*.json")):
        name = gt_file.stem
        pdf = TEST_PDFS_DIR / f"{name}.pdf"
        if pdf.exists():
            with open(gt_file) as f:
                gt = json.load(f)
            cases.append({
                "name": name,
                "pdf_path": str(pdf),
                "ground_truth": gt,
                "category": "custom_controlled",
                "has_tables": len(gt.get("tables", [])) > 0,
                "has_images": gt.get("images", 0) > 0,
                "key_phrases": gt.get("key_phrases", gt.get("text_sections", [])),
                "key_numbers": gt.get("key_numbers", []),
            })

    # Real-world PDFs (partial ground truth - check for known content)
    real_world = [
        {
            "name": "attention_paper",
            "pdf_path": str(TEST_PDFS_DIR / "attention_paper.pdf"),
            "category": "academic_paper",
            "description": "Attention Is All You Need - transformer paper",
            "has_tables": True,
            "has_images": True,
            "key_phrases": [
                "Attention Is All You Need",
                "transformer",
                "multi-head attention",
                "BLEU",
                "encoder",
                "decoder",
            ],
            "key_numbers": ["512", "8", "65536", "0.1"],
            "ground_truth": {"tables": [{"rows": 3}, {"rows": 4}]},
        },
        {
            "name": "pdf_parsing_study",
            "pdf_path": str(TEST_PDFS_DIR / "pdf_parsing_study.pdf"),
            "category": "academic_paper",
            "description": "PDF parsing tools comparison study paper",
            "has_tables": True,
            "has_images": True,
            "key_phrases": [
                "PyMuPDF", "pdfplumber", "DocLayNet", "table extraction",
                "parsing", "document"
            ],
            "key_numbers": [],
            "ground_truth": {"tables": [{"rows": 3}]},
        },
        {
            "name": "survey_llm",
            "pdf_path": str(TEST_PDFS_DIR / "survey_llm.pdf"),
            "category": "academic_survey",
            "description": "Large language model survey paper",
            "has_tables": True,
            "has_images": True,
            "key_phrases": [
                "language model", "transformer", "GPT",
                "training", "fine-tuning", "benchmark"
            ],
            "key_numbers": [],
            "ground_truth": {"tables": [{"rows": 3}]},
        },
        {
            "name": "table_benchmark",
            "pdf_path": str(TEST_PDFS_DIR / "table_benchmark.pdf"),
            "category": "academic_paper",
            "description": "Table-rich academic paper",
            "has_tables": True,
            "has_images": True,
            "key_phrases": ["table", "model", "dataset", "accuracy", "baseline"],
            "key_numbers": [],
            "ground_truth": {"tables": [{"rows": 3}, {"rows": 3}]},
        },
    ]

    for rw in real_world:
        if Path(rw["pdf_path"]).exists():
            cases.append(rw)

    return cases


# ============================================================
# MAIN BENCHMARK RUNNER
# ============================================================

def run_benchmarks() -> Dict:
    """Run all benchmarks and collect results."""
    cases = get_benchmark_cases()
    print(f"\nFound {len(cases)} benchmark PDFs\n")

    # Text extractor registry
    text_extractors = {
        "PyMuPDF": extract_pymupdf_text,
        "pdfplumber": extract_pdfplumber_text,
        "pypdf": extract_pypdf_text,
        "pdfminer.six": extract_pdfminer_text,
    }

    # Table extractor registry
    table_extractors = {
        "PyMuPDF": extract_pymupdf_tables,
        "pdfplumber": extract_pdfplumber_tables,
        "camelot-lattice": extract_camelot_tables,
        "camelot-stream": extract_camelot_stream_tables,
    }

    all_results = {
        "text_extraction": {},
        "table_extraction": {},
        "benchmark_cases": [c["name"] for c in cases],
        "metadata": {}
    }

    # --- TEXT EXTRACTION BENCHMARKS ---
    print("=" * 60)
    print("TEXT EXTRACTION BENCHMARK")
    print("=" * 60)

    for extractor_name, extract_fn in text_extractors.items():
        print(f"\n  Testing: {extractor_name}")
        extractor_results = []

        for case in cases:
            print(f"    PDF: {case['name']}", end=" ... ")
            text, elapsed, error = extract_fn(case["pdf_path"])

            if error:
                print(f"ERROR: {error[:60]}")
                extractor_results.append({
                    "pdf": case["name"],
                    "category": case.get("category", "unknown"),
                    "error": error,
                    "elapsed": elapsed,
                    "text_length": 0,
                    "keyword_recall": 0.0,
                    "number_preservation": 0.0,
                    "printable_ratio": 0.0,
                    "overall_score": 0.0,
                })
                continue

            quality = assess_text_quality(text)
            kw_recall = keyword_recall(text, case.get("key_phrases", []))
            num_pres = count_numbers_preserved(text, case.get("key_numbers", []))

            # Overall score (weighted)
            overall = (kw_recall * 0.5 + num_pres * 0.3 + quality["printable_ratio"] * 0.2)

            print(f"OK ({elapsed:.2f}s, {len(text)} chars, recall={kw_recall:.0%})")
            extractor_results.append({
                "pdf": case["name"],
                "category": case.get("category", "unknown"),
                "elapsed": elapsed,
                "text_length": len(text),
                "keyword_recall": kw_recall,
                "number_preservation": num_pres,
                "printable_ratio": quality["printable_ratio"],
                "overall_score": overall,
                "error": None,
            })

        all_results["text_extraction"][extractor_name] = extractor_results

    # --- TABLE EXTRACTION BENCHMARKS ---
    print("\n" + "=" * 60)
    print("TABLE EXTRACTION BENCHMARK")
    print("=" * 60)

    for extractor_name, extract_fn in table_extractors.items():
        print(f"\n  Testing: {extractor_name}")
        extractor_results = []

        for case in cases:
            print(f"    PDF: {case['name']}", end=" ... ")
            tables, elapsed, error = extract_fn(case["pdf_path"])

            if error:
                print(f"ERROR: {error[:60]}")
                extractor_results.append({
                    "pdf": case["name"],
                    "category": case.get("category", "unknown"),
                    "error": error,
                    "elapsed": elapsed,
                    "tables_found": 0,
                    "detection_rate": 0.0,
                    "cell_accuracy": 0.0,
                    "overall_score": 0.0,
                })
                continue

            score = score_table_extraction(tables, case.get("ground_truth", {}))
            overall = score["detection_rate"] * 0.5 + score.get("cell_accuracy", 0) * 0.5

            print(f"OK ({elapsed:.2f}s, {len(tables)} tables, detect={score['detection_rate']:.0%})")
            extractor_results.append({
                "pdf": case["name"],
                "category": case.get("category", "unknown"),
                "elapsed": elapsed,
                "tables_found": len(tables),
                "tables_expected": score.get("tables_expected", "?"),
                "detection_rate": score["detection_rate"],
                "cell_accuracy": score.get("cell_accuracy", 0.0),
                "overall_score": overall,
                "error": None,
            })

        all_results["table_extraction"][extractor_name] = extractor_results

    # --- AGGREGATE SCORES ---
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    text_summary = {}
    for extractor_name, results in all_results["text_extraction"].items():
        valid = [r for r in results if r.get("error") is None]
        if valid:
            text_summary[extractor_name] = {
                "avg_keyword_recall": sum(r["keyword_recall"] for r in valid) / len(valid),
                "avg_number_preservation": sum(r["number_preservation"] for r in valid) / len(valid),
                "avg_printable_ratio": sum(r["printable_ratio"] for r in valid) / len(valid),
                "avg_speed": sum(r["elapsed"] for r in valid) / len(valid),
                "avg_overall": sum(r["overall_score"] for r in valid) / len(valid),
                "error_count": len(results) - len(valid),
                "tested": len(valid),
            }
        else:
            text_summary[extractor_name] = {"error_count": len(results), "avg_overall": 0.0}

    table_summary = {}
    for extractor_name, results in all_results["table_extraction"].items():
        valid = [r for r in results if r.get("error") is None]
        if valid:
            table_summary[extractor_name] = {
                "avg_detection_rate": sum(r["detection_rate"] for r in valid) / len(valid),
                "avg_cell_accuracy": sum(r["cell_accuracy"] for r in valid) / len(valid),
                "avg_speed": sum(r["elapsed"] for r in valid) / len(valid),
                "avg_overall": sum(r["overall_score"] for r in valid) / len(valid),
                "total_tables_found": sum(r["tables_found"] for r in valid),
                "error_count": len(results) - len(valid),
                "tested": len(valid),
            }
        else:
            table_summary[extractor_name] = {"error_count": len(results), "avg_overall": 0.0}

    all_results["text_summary"] = text_summary
    all_results["table_summary"] = table_summary

    print("\nText Extraction Summary:")
    for name, s in text_summary.items():
        if "avg_overall" in s:
            print(f"  {name:20s}: overall={s['avg_overall']:.3f}, "
                  f"recall={s.get('avg_keyword_recall', 0):.3f}, "
                  f"speed={s.get('avg_speed', 0):.2f}s")

    print("\nTable Extraction Summary:")
    for name, s in table_summary.items():
        if "avg_overall" in s:
            print(f"  {name:20s}: overall={s['avg_overall']:.3f}, "
                  f"detect={s.get('avg_detection_rate', 0):.3f}, "
                  f"cell_acc={s.get('avg_cell_accuracy', 0):.3f}, "
                  f"speed={s.get('avg_speed', 0):.2f}s")

    return all_results


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    results = run_benchmarks()

    output_path = BASE_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
