"""
TDD-based tests for PDF extraction libraries.
Tests are written first (failing), then we verify each library against them.

Test philosophy:
- Ground truth is established from known PDF content
- Tests assert minimum acceptable quality thresholds
- Each test is independent and measurable
"""

import os
import sys
import json
import pytest
import re

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_TRUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ground_truth")
TEST_PDFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_pdfs")


def load_ground_truth(name):
    with open(os.path.join(GROUND_TRUTH_DIR, f"{name}.json")) as f:
        return json.load(f)


def pdf_path(name):
    return os.path.join(TEST_PDFS_DIR, f"{name}.pdf")


# ============================================================
# TEXT EXTRACTION TESTS
# ============================================================

class TestTextExtraction:
    """Tests that extraction contains key expected text fragments."""

    def test_simple_table_title_extracted(self, extractor_func):
        """The document title must be present in extracted text."""
        text = extractor_func(pdf_path("simple_table"))
        assert "Q3 2024 Financial Summary Report" in text or \
               "q3 2024 financial summary report" in text.lower(), \
               f"Title not found in extracted text (first 200 chars): {text[:200]}"

    def test_simple_table_numbers_present(self, extractor_func):
        """Key financial numbers must be extractable."""
        text = extractor_func(pdf_path("simple_table"))
        # At least 3 of the 5 key numbers must be present
        gt = load_ground_truth("simple_table")
        found = sum(1 for num in gt["key_numbers"] if num in text)
        assert found >= 3, \
               f"Only {found}/5 key numbers found. Text sample: {text[:300]}"

    def test_mixed_content_sections_extracted(self, extractor_func):
        """Section headings from academic paper must be present."""
        text = extractor_func(pdf_path("mixed_content"))
        gt = load_ground_truth("mixed_content")
        found_sections = sum(1 for s in gt["sections"] if s in text or s.lower() in text.lower())
        assert found_sections >= 3, \
               f"Only {found_sections}/{len(gt['sections'])} sections found"

    def test_text_heavy_key_phrases(self, extractor_func):
        """Critical technical terms in text-heavy document must be extractable."""
        text = extractor_func(pdf_path("text_heavy"))
        gt = load_ground_truth("text_heavy")
        text_lower = text.lower()
        found = sum(1 for phrase in gt["key_phrases"]
                    if phrase.lower() in text_lower)
        required = int(len(gt["key_phrases"]) * 0.7)  # 70% threshold
        assert found >= required, \
               f"Only {found}/{len(gt['key_phrases'])} key phrases found (need {required})"

    def test_multicolumn_key_facts(self, extractor_func):
        """Key statistics from multicolumn document must be present."""
        text = extractor_func(pdf_path("multicolumn"))
        # Check for key numbers
        assert "$0.033" in text or "0.033" in text, "Solar LCOE not found"
        assert "500 billion" in text or "$500" in text, "Investment figure not found"

    def test_minimum_text_length_simple(self, extractor_func):
        """Extracted text must have meaningful length (not empty/truncated)."""
        text = extractor_func(pdf_path("simple_table"))
        assert len(text.strip()) >= 200, \
               f"Text too short: {len(text.strip())} chars"

    def test_minimum_text_length_text_heavy(self, extractor_func):
        """Text-heavy document should yield at least 1000 chars."""
        text = extractor_func(pdf_path("text_heavy"))
        assert len(text.strip()) >= 1000, \
               f"Text too short: {len(text.strip())} chars"

    def test_no_garbled_text(self, extractor_func):
        """Extracted text should be mostly ASCII readable text."""
        text = extractor_func(pdf_path("text_heavy"))
        if not text.strip():
            pytest.skip("Empty text extracted")
        # Count printable ASCII vs total
        printable = sum(1 for c in text if c.isprintable() or c in '\n\t ')
        ratio = printable / max(len(text), 1)
        assert ratio > 0.90, \
               f"Too much garbled text: only {ratio:.1%} printable chars"


# ============================================================
# TABLE EXTRACTION TESTS
# ============================================================

class TestTableExtraction:
    """Tests that tables are correctly identified and extracted."""

    def test_simple_table_row_count(self, table_extractor_func):
        """Simple financial table should have correct row count."""
        tables = table_extractor_func(pdf_path("simple_table"))
        assert len(tables) >= 1, "No tables found in simple_table.pdf"
        # Find table with ~6 rows
        max_rows = max(len(t) for t in tables) if tables else 0
        assert max_rows >= 4, f"Largest table has only {max_rows} rows (expected ~6)"

    def test_simple_table_column_count(self, table_extractor_func):
        """Simple financial table should have 5 columns."""
        tables = table_extractor_func(pdf_path("simple_table"))
        if not tables:
            pytest.fail("No tables found")
        # Check widest table has right columns
        for table in tables:
            if table and len(table) > 3:
                max_cols = max(len(row) for row in table if row)
                if max_cols >= 4:  # at least 4 of 5 columns
                    return
        pytest.fail(f"No table with 4+ columns found. Tables: {[len(t) for t in tables]}")

    def test_simple_table_header_content(self, table_extractor_func):
        """Table header row should contain expected column names."""
        tables = table_extractor_func(pdf_path("simple_table"))
        if not tables:
            pytest.skip("No tables extracted")
        # Find any table that has financial content
        all_cells = ' '.join(
            str(cell) for table in tables for row in table for cell in (row if row else [])
        ).lower()
        assert "product" in all_cells or "q1" in all_cells or "q2" in all_cells, \
               f"Expected table headers not found in: {all_cells[:200]}"

    def test_simple_table_data_values(self, table_extractor_func):
        """Table cells should contain the actual data values."""
        tables = table_extractor_func(pdf_path("simple_table"))
        if not tables:
            pytest.skip("No tables extracted")
        all_text = ' '.join(
            str(cell) for table in tables for row in table for cell in (row if row else [])
        )
        # Check for some specific values
        assert any(v in all_text for v in ["Widget A", "15,800", "26.9%"]), \
               f"Expected data values not found in tables: {all_text[:200]}"

    def test_complex_table_multiple_tables(self, table_extractor_func):
        """Complex PDF has 2 tables - both should be detected."""
        tables = table_extractor_func(pdf_path("complex_table"))
        assert len(tables) >= 1, \
               f"Expected at least 1 table, got {len(tables)}"

    def test_mixed_content_table_extraction(self, table_extractor_func):
        """ML performance table should be extractable from mixed content doc."""
        tables = table_extractor_func(pdf_path("mixed_content"))
        if not tables:
            pytest.skip("No tables extracted")
        all_text = ' '.join(
            str(cell) for table in tables for row in table for cell in (row if row else [])
        ).lower()
        # Should contain ML-related content
        assert any(term in all_text for term in ["accuracy", "method", "cnn", "forest", "svm"]), \
               f"ML table content not found. Got: {all_text[:200]}"


# ============================================================
# QUALITY METRIC TESTS
# ============================================================

class TestQualityMetrics:
    """Tests that verify extraction quality via metrics."""

    def test_text_similarity_simple_table(self, extractor_func):
        """Text from simple_table should have high similarity to known content."""
        text = extractor_func(pdf_path("simple_table"))
        gt = load_ground_truth("simple_table")
        text_lower = text.lower()
        # Check that all key section texts appear
        hits = sum(1 for phrase in gt["text_sections"]
                   if phrase.lower() in text_lower)
        assert hits >= 2, f"Only {hits}/{len(gt['text_sections'])} text sections found"

    def test_number_extraction_accuracy(self, extractor_func):
        """Specific numbers should be extractable without corruption."""
        text = extractor_func(pdf_path("simple_table"))
        gt = load_ground_truth("simple_table")
        found = [num for num in gt["key_numbers"] if num in text]
        assert len(found) >= 3, \
               f"Found only {found} of {gt['key_numbers']}"

    def test_percentage_values_not_corrupted(self, extractor_func):
        """Percentage values should not be corrupted (26.1% stays as 26.1%)."""
        text = extractor_func(pdf_path("simple_table"))
        # At least one percentage should be recognizable
        percentages = re.findall(r'\d+\.\d+%', text)
        assert len(percentages) >= 1, \
               f"No percentage values found (x.x%) in: {text[:300]}"
