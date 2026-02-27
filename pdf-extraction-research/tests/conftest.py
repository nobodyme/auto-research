"""
pytest conftest.py - fixtures for parametrizing tests across libraries.
"""
import pytest


def get_pymupdf_text(pdf_path):
    """Extract text using PyMuPDF (fitz)."""
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_pdfplumber_text(pdf_path):
    """Extract text using pdfplumber."""
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def get_pypdf_text(pdf_path):
    """Extract text using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def get_pdfminer_text(pdf_path):
    """Extract text using pdfminer.six."""
    from pdfminer.high_level import extract_text
    return extract_text(pdf_path)


def get_pymupdf_tables(pdf_path):
    """Extract tables using PyMuPDF built-in table detector."""
    import fitz
    doc = fitz.open(pdf_path)
    all_tables = []
    for page in doc:
        tables = page.find_tables()
        for table in tables:
            extracted = table.extract()
            if extracted:
                all_tables.append(extracted)
    doc.close()
    return all_tables


def get_pdfplumber_tables(pdf_path):
    """Extract tables using pdfplumber."""
    import pdfplumber
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)
    return all_tables


def get_camelot_tables(pdf_path):
    """Extract tables using camelot-py."""
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        result = []
        for t in tables:
            result.append(t.df.values.tolist())
        return result
    except Exception as e:
        return []


def get_pdfminer_tables(pdf_path):
    """pdfminer doesn't have native table extraction - return empty."""
    return []


# Registry of extractors
TEXT_EXTRACTORS = {
    "pymupdf": get_pymupdf_text,
    "pdfplumber": get_pdfplumber_text,
    "pypdf": get_pypdf_text,
    "pdfminer": get_pdfminer_text,
}

TABLE_EXTRACTORS = {
    "pymupdf": get_pymupdf_tables,
    "pdfplumber": get_pdfplumber_tables,
    "camelot": get_camelot_tables,
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "extractor: mark test with extractor name"
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests with all available extractors."""
    if "extractor_func" in metafunc.fixturenames:
        extractor_names = list(TEXT_EXTRACTORS.keys())
        metafunc.parametrize(
            "extractor_func",
            [TEXT_EXTRACTORS[name] for name in extractor_names],
            ids=extractor_names
        )
    if "table_extractor_func" in metafunc.fixturenames:
        extractor_names = list(TABLE_EXTRACTORS.keys())
        metafunc.parametrize(
            "table_extractor_func",
            [TABLE_EXTRACTORS[name] for name in extractor_names],
            ids=extractor_names
        )
