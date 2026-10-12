# PDF Extraction Research Notes

**Date**: 2026-02-27

## Goal
Compare popular Python PDF extraction libraries for quality of content extraction, focusing on:
- Text extraction accuracy
- Table extraction quality
- Image handling
- Real-world PDF handling (scanned docs, mixed content)

## Libraries to Test
1. **PyMuPDF (fitz)** - Fast C-based library, good for text/images
2. **pdfplumber** - Built on pdfminer, great for tables
3. **pypdf** - Pure Python, basic extraction
4. **pdfminer.six** - Low-level text mining
5. **unstructured** - AI-powered document parsing
6. **marker-pdf** - ML-based, converts to markdown
7. **docling** - IBM's document parser with table/structure support
8. **MinerU** - OpenDataLab's open-source extraction tool

## Benchmark Datasets
- **DocLayNet** - IBM dataset for document layout analysis
- **DP-Bench** - Document parsing benchmark (Upstage)
- **PubTables-1M** - Table extraction benchmark (too large, sample)
- **OmniDocBench** - Recent comprehensive benchmark (CVPR 2025)
- **FinTabNet** - Financial tables from Fortune 500 companies
- **olmOCR-Bench** - Allen AI benchmark with 7,000+ unit tests

## Metrics to Measure
- Text similarity (BLEU, character-level accuracy)
- Table structure accuracy (TEDS score)
- Image extraction count
- Processing speed

---

## Detailed Benchmark Research (2026-02-27)

### 1. OmniDocBench (CVPR 2025)

**Status**: Best-in-class, most comprehensive recent benchmark

**URLs**:
- GitHub: https://github.com/opendatalab/OmniDocBench
- HuggingFace: https://huggingface.co/datasets/opendatalab/OmniDocBench
- ArXiv: https://arxiv.org/abs/2412.07626
- CVPR 2025 paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf

**Dataset Contents**:
- 1,355 PDF pages (v1.0), expanded with v1.5
- 9 document types: academic papers, financial reports, newspapers, textbooks, handwritten notes, books, PPTs, test papers, magazines
- 4 layout types
- 3 language types (English, Chinese, mixed)
- 15 block-level annotation categories (>20k total elements)
- 4 span-level annotation categories (>80k total elements)
- Reading order annotations

**Download Commands**:
```bash
pip install huggingface_hub
huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir ./OmniDocBench

# For faster downloads:
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download opendatalab/OmniDocBench --repo-type dataset --local-dir ./OmniDocBench
```

**Folder structure**:
```
OmniDocBench/
├── images/         # JPG images for each page
├── pdfs/           # PDF versions (same pages, for models requiring PDFs)
├── ori_pdfs/       # Original PDFs with metadata
└── OmniDocBench.json  # Ground truth annotations
```

**Metrics Used**:
- Normalized Edit Distance (NED)
- BLEU
- METEOR
- TEDS (for tables)
- COCODet (for layout detection)
- Four major modules evaluated: text, tables, formulas, reading order

**Evaluation Setup**:
```bash
conda create -n omnidocbench python=3.10
conda activate omnidocbench
pip install -r requirements.txt
python run.py --config configs/end2end.yaml
```

**Key Finding**: Pipeline methods (MinerU, Mathpix) outperform on standard docs; VLMs (GPT-4o, Qwen2-VL) better on unusual types (slides, handwriting)

---

### 2. DP-Bench (Upstage)

**Status**: Focused benchmark with novel NID metric, good for text+table evaluation

**URLs**:
- HuggingFace: https://huggingface.co/datasets/upstage/dp-bench
- Docling integration docs: https://github.com/docling-project/docling-eval/blob/main/docs/DP-Bench_benchmarks.md

**Dataset Contents**:
- 200 document pages total
  - 90 samples from Library of Congress
  - 90 samples from Open Educational Resources
  - 20 from Upstage internal documents
- 12 element types: Table, Paragraph, Figure, Chart, Header, Footer, Caption, Equation, Heading1, List, Index, Footnote
- Wide range of layouts: text-heavy to complex tables

**Metrics Used**:
- **NID (Normalized Indel Distance)**: Measures how well a parser detects and serializes elements in human reading order. Excludes substitutions (insertion/deletion only). Formula: `NID = 1 − distance / (len(reference) + len(prediction))`. Higher NID = better. Excludes tables, figures, charts (uses TEDS for those).
- **TEDS (Tree Edit Distance-based Similarity)**: Measures table structure + content similarity. Formula: `TEDS(Ta, Tb) = 1 − EditDist(Ta, Tb) / max(|Ta|, |Tb|)`. Higher = better.
- **TEDS-S (Struct)**: Like TEDS but ignores cell content, only measures structural accuracy (row/column alignment).

**Leaderboard Results** (as of Oct 2024):
| Source | TEDS | TEDS-S | NID | Avg Time (s) |
|--------|------|--------|-----|--------------|
| Upstage (best) | 92.06 | 93.81 | 96.23 | 3.79 |
| AWS | 86.39 | 90.22 | 95.94 | 14.47 |
| Microsoft | 85.54 | 89.07 | 87.03 | 4.44 |
| LlamaParse | 73.36 | 76.29 | 92.22 | 4.14 |
| Unstructured | 64.49 | 69.90 | 90.42 | 13.14 |

**Download**:
```python
from datasets import load_dataset
ds = load_dataset("upstage/dp-bench")
```

---

### 3. DocLayNet (IBM Research)

**Status**: Large, high-quality layout annotation dataset; used as test set in academic comparisons

**URLs**:
- GitHub: https://github.com/DS4SD/DocLayNet
- HuggingFace (original): https://huggingface.co/datasets/ds4sd/DocLayNet
- HuggingFace (v1.1 with text): https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1
- HuggingFace (v1.2 with PDFs): https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2
- ICDAR 2023 competition: https://huggingface.co/datasets/ds4sd/icdar2023-doclaynet
- ArXiv: https://arxiv.org/abs/2206.01062
- ACM KDD 2022: https://dl.acm.org/doi/10.1145/3534678.3539043

**Dataset Contents**:
- 80,863 PDF pages total
- 11 class labels: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
- 6 document categories: Financial Reports, Scientific Articles, Laws & Regulations, Government Tenders, Manuals, Patents
- Some documents in German, French, Japanese (mostly English)
- Split: 69,103 train / 6,480 val / 4,994 test

**Size**:
- doclaynet_core.zip: ~28 GiB
- doclaynet_extra.zip: ~7.5 GiB (needed for OCR text associations)
- v1.2 with embedded PDFs: ~35.6 GB train alone

**Download Commands**:
```python
from datasets import load_dataset
# Original COCO format
ds = load_dataset("ds4sd/DocLayNet")
# v1.1 with text content
ds = load_dataset("ds4sd/DocLayNet-v1.1")
# v1.2 with embedded PDFs
ds = load_dataset("ds4sd/DocLayNet-v1.2")
```
```bash
# CLI download
huggingface-cli download ds4sd/DocLayNet --repo-type dataset
```

**Metrics Used**:
- mAP (mean Average Precision) for layout detection
- F1, BLEU, Local Alignment for text extraction (in the arXiv:2410.09871 study)

**Key Academic Use**: The paper "A Comparative Study of PDF Parsing Tools Across Diverse Document Categories" (arXiv:2410.09871, Oct 2024) used DocLayNet as its ground truth for evaluating 10 PDF parsing tools. First study to do so.

**License**: CDLA-Permissive-1.0

---

### 4. PubTables-1M (Microsoft Research, CVPR 2022)

**Status**: Largest table-specific benchmark, comprehensive but very large

**URLs**:
- GitHub (Table Transformer): https://github.com/microsoft/table-transformer
- HuggingFace: https://huggingface.co/datasets/bsmock/pubtables-1m
- Kaggle (structure subset): https://www.kaggle.com/datasets/bsmock/pubtables-1m-structure
- Microsoft Research Open Data: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
- ArXiv: https://arxiv.org/abs/2110.00061

**Dataset Contents**:
- ~947,642 annotated tables
- From scientific articles (PubMed Central)
- Three tasks: table detection, table structure recognition, functional analysis
- Full bounding boxes in image AND PDF coordinates
- Row, column, cell annotations (including blank cells)
- Column header and projected row header annotations

**Download**:
```bash
# Via HuggingFace (tar.gz files)
# Switch to "Files and versions" tab on HuggingFace
# Or use wget/curl to download .tar.gz files
# Then extract with included script:
bash extract_structure_dataset.sh

# Via Microsoft Research Open Data + azcopy:
azcopy copy "https://msropendataset01.blob.core.windows.net/pubtables1m?[SAS_TOKEN]" \
  "/path/to/download/" --recursive
```

**Metrics Used**:
- **GriTS (Grid Table Similarity)**: Custom metric for table structure recognition
- Standard detection metrics: AP, AR at IoU thresholds

**Tasks**:
1. Table Detection: Find tables in page images
2. Table Structure Recognition (TSR): Parse internal table structure
3. Functional Analysis: Identify header cells vs. data cells

---

### 5. FinTabNet (IBM Research)

**Status**: Real-world financial tables from Fortune 500 companies

**URLs**:
- IBM Developer (official): https://developer.ibm.com/data/fintabnet/
- HuggingFace (improved FinTabNet.c): https://huggingface.co/datasets/bsmock/FinTabNet.c
- Papers With Code: https://paperswithcode.com/dataset/fintabnet
- Docling-eval integration: https://github.com/docling-project/docling-eval/blob/main/docs/FinTabNet_benchmarks.md

**Dataset Contents**:
- 10,635 tables in test set (9,681 after cleaning)
- Financial earnings documents from Fortune 500 companies
- Complex financial tables with merged cells, multi-level headers
- Annotated through automatic matching between PDF and HTML

**Download**:
```bash
# Direct CDN example:
wget https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/examples.tar.gz
# Full dataset: visit IBM Developer page for download link
```

**Metrics Used**:
- TEDS (Tree Edit Distance-based Similarity)
- TEDS-S (structural only)

---

### 6. SCORE-Bench (Unstructured, 2025)

**Status**: Enterprise-focused, production document diversity, newest benchmark

**URLs**:
- Blog/Announcement: https://unstructured.io/blog/introducing-score-bench-an-open-benchmark-for-document-parsing
- HuggingFace: https://huggingface.co/unstructuredio (look for SCORE-Bench dataset)
- GitHub: https://github.com/Unstructured-IO

**Dataset Contents**:
- Real-world enterprise documents
- Complex nested tables, merged cells
- Scanned documents, forms, handwriting
- Healthcare, finance, legal, public sector domains
- All human-annotated by domain experts

**Metrics Used**:
- **Adjusted CCT (Corrected Cell Text)**: Word-weighted fuzzy alignment
- Cell Content Accuracy
- Cell Level Index Accuracy
- TEDS (corrected implementation)
- Additional: content fidelity, hallucination control, coverage, structural understanding

**Key Results** (from blog):
- Unstructured High-Res + Claude Sonnet 4 pipeline: 0.773 cell content accuracy (top)
- Top 3 pipelines all from Unstructured

---

### 7. olmOCR-Bench (Allen AI, 2025)

**Status**: Novel "unit test" approach for PDF linearization quality

**URLs**:
- GitHub: https://github.com/allenai/olmocr
- ArXiv: https://arxiv.org/abs/2502.18443
- Review: https://www.llamaindex.ai/blog/olmocr-bench-review-insights-and-pitfalls-on-an-ocr-benchmark

**Dataset Contents**:
- 7,010 test instances across 1,403 PDFs
- Scientific papers, historical manuscripts, diverse document types
- Pass/Fail unit tests (binary rules)

**Metrics Used**:
- Pass/Fail unit tests for: text presence, reading order, tables, formulas, baseline functionality
- Binary rules rather than fuzzy matching

**Key Finding**: olmOCR-7B outperforms GPT-4o, Gemini Flash 2, Qwen-2.5-VL on this benchmark

---

### 8. gipplab/pdf-benchmark (Göttingen University)

**Status**: Academic PDF tool comparison, 10 tools evaluated

**URLs**:
- GitHub: https://github.com/gipplab/pdf-benchmark
- ArXiv: https://arxiv.org/abs/2303.09957

**Dataset Contents**:
- Based on DocBank (1.5M annotated elements from 500K pages of arXiv papers)
- Academic document focused
- Multi-domain: metadata, references, tables, other elements

**Tools Evaluated**:
- GROBID, CERMINE, Science Parse, Adobe Extract, and 6 others

**Metrics**: Token-level evaluation metrics, multi-task evaluation

**Key Findings**:
- GROBID: best metadata and reference extraction
- Adobe Extract: best table extraction (but still low absolute performance)
- All tools struggle with: lists, footers, equations

---

### 9. py-pdf/benchmarks (Speed-Focused)

**Status**: Pure speed/quality benchmarks for Python PDF libraries

**URLs**:
- GitHub: https://github.com/py-pdf/benchmarks
- Live results: https://py-pdf.github.io/pypdf/dev/bench/

**Scope**: Digital (born-digital) PDFs only, not scanned

**Speed Results** (text extraction, 14 test files):
| Library | Avg Time |
|---------|----------|
| PyMuPDF | 0.5s |
| pypdfium2 | 1.1s |
| pypdf | 4.2s |
| pdfminer.six | 7.4s |

---

## Comparison Papers and Blog Posts

### Academic Papers

1. **"A Comparative Study of PDF Parsing Tools Across Diverse Document Categories"**
   - ArXiv: https://arxiv.org/abs/2410.09871
   - Date: October 2024
   - Tools: PyPDF, pdfminer.six, PyMuPDF, pdfplumber, pypdfium2, Unstructured, Tabula, Camelot, Nougat, TATR
   - Dataset: DocLayNet (6 document categories, 800 balanced docs each)
   - Metrics: F1, BLEU, Local Alignment
   - Key findings:
     - Text: PyMuPDF and pypdfium2 best overall; all struggle with Scientific/Patent docs
     - Tables: TATR best for Financial/Patent/Law/Scientific; Camelot best for Gov Tenders; PyMuPDF best for Manuals
     - Deep learning (Nougat) superior for challenging categories

2. **"Benchmarking Document Parsers on Mathematical Formula Extraction from PDFs"**
   - ArXiv: https://arxiv.org/abs/2512.09874
   - Date: December 2024
   - Novel: Synthetic PDFs with known LaTeX ground truth + LLM-as-judge evaluation
   - Results: VLMs (Qwen3-VL, Gemini 3 Pro, PaddleOCR-VL, Mathpix) score >9.6/10; traditional parsers much lower

3. **"Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion"**
   - ArXiv: https://arxiv.org/abs/2408.09869 (Technical Report, Jan 2025)
   - Speed benchmark: Docling 3.1 sec/page (CPU), MinerU 3.3 sec/page, Unstructured 4.2 sec/page, Marker 16+ sec/page
   - GPU (CUDA L4): MinerU 0.21 sec/page, Docling 0.49 sec/page, Marker 0.86 sec/page

### Blog Posts and Practical Comparisons

1. **Procycons PDF Data Extraction Benchmark 2025**
   - URL: https://procycons.com/en/blogs/pdf-data-extraction-benchmark/
   - Tools: Docling, Unstructured, LlamaParse
   - Dataset: 5 corporate sustainability reports, diverse complexity
   - Key findings: Docling 97.9% complex table accuracy; LlamaParse fastest (~6s constant); Unstructured 100% simple tables but 75% complex

2. **Comparative Evaluation of 12 Open-Source PDF Parsers**
   - URL: https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html
   - Tools: MinerU, PaddleOCR, Marker, Unstructured, Zerox, Sparrow, pdf-extract-api, DFlux, Mathpix, others
   - MinerU: best all-rounder for Markdown conversion; Zerox: best OCR; Llama Parse: best table extraction

3. **py-pdf benchmarks blog summary**
   - Tools: PyMuPDF best speed overall at 0.5s average

4. **"I Tested 7 Python PDF Extractors So You Don't Have To (2025 Edition)"**
   - URL: https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257
   - Recommendation: pymupdf4llm for general use; unstructured for RAG chunks; pypdfium2 for pure speed

5. **"Best Python PDF to Text Parser Libraries: A 2026 Evaluation"**
   - URL: https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/

6. **Nicolas' Notebook: PDF Parsing for LLM Input**
   - URL: https://nbrosse.github.io/posts/pdf-parsing/pdf-parsing.html
   - Qualitative comparison: Docling, Marker, MinerU, PyMuPDF (open-source) vs. LlamaParse, Gemini, Mistral (closed)

7. **Reducto Document Parser Comparison**
   - URL: https://llms.reducto.ai/document-parser-comparison
   - Tools: Docling, LlamaParse, Unstructured, Reducto

## docling-eval Framework

For running structured benchmarks against multiple datasets:
- GitHub: https://github.com/docling-project/docling-eval
- Supports: OmniDocBench, DocLayNetV1, FinTabNet, PubTabNet, Pub1M
- Modalities: layout, table structure, reading order, markdown text
- CLI commands: create, create-eval, create-gt, evaluate, visualize

```bash
pip install docling-eval
# Create ground truth
docling-eval create-gt --dataset dp-bench
# Generate predictions
docling-eval create-eval --dataset dp-bench
# Evaluate
docling-eval evaluate --dataset dp-bench
```

## Key Metric Definitions Summary

| Metric | What It Measures | Higher = Better | Used In |
|--------|-----------------|-----------------|---------|
| TEDS | Table edit distance (structure + content) | Yes | DP-Bench, FinTabNet, PubTabNet |
| TEDS-S | Table edit distance (structure only) | Yes | DP-Bench, FinTabNet |
| NID | Text serialization accuracy (no substitutions) | Lower = better (0=perfect) | DP-Bench |
| GriTS | Grid table similarity (cells, rows, cols) | Yes | PubTables-1M |
| NED | Normalized edit distance for full text | Lower = better | OmniDocBench |
| BLEU | N-gram text overlap | Yes | Multiple |
| mAP | Layout detection accuracy | Yes | DocLayNet |
| F1 | Precision/recall harmonic mean | Yes | DocLayNet study |

## Our Experimental Findings (2026-02-27)

### Test Setup
- **Libraries tested**: PyMuPDF 1.27.1, pdfplumber 0.11.9, pypdf 6.7.4, pdfminer.six 20251230, camelot-py 1.0.9
- **PDFs tested**: 9 total (5 custom with ground truth + 4 real arXiv papers)
- **TDD**: 62 tests written, 61 pass, 1 real bug found

### Text Extraction Results (9 PDFs)
| Library | Overall Score | Keyword Recall | Number Preservation | Avg Speed |
|---------|--------------|----------------|--------------------|-----------|
| PyMuPDF | **0.967** | 95.0% | 97.2% | **0.11s** |
| pdfplumber | 0.967 | 95.0% | 97.2% | 2.48s |
| pypdf | 0.967 | 95.0% | 97.2% | 0.64s |
| pdfminer.six | 0.967 | 95.0% | 97.2% | 1.71s |

**Key insight**: All 4 libraries achieve near-identical accuracy. PyMuPDF is 23x faster than pdfplumber for text extraction.

### Table Extraction Results (9 PDFs)
| Library | Overall Score | Detection Rate | Cell Accuracy | Avg Speed | Tables Found |
|---------|--------------|----------------|--------------|-----------|--------------|
| PyMuPDF | **0.919** | 100% | 83.9% | **1.48s** | 35 |
| pdfplumber | 0.919 | 100% | 83.9% | 2.69s | 61 |
| camelot-stream | 0.892 | 94.4% | 83.9% | 2.90s | 100 |
| camelot-lattice | 0.882 | 94.4% | 82.0% | 7.58s | 28 |

**Key insight**: pdfplumber finds more tables (61 vs 35) but many are over-split fragments. camelot-lattice is 5x slower than PyMuPDF.

### Critical Bug Found (TDD-driven discovery)
**camelot-lattice** fails `test_simple_table_header_content`: extracts data rows but skips header row when header has background color styling. This is a real production bug.

### Speed Summary
- PyMuPDF is the clear winner for speed: **23x faster** than pdfplumber for text, **5x faster** than camelot-lattice for tables
- For LLM preprocessing pipelines, PyMuPDF is the recommended choice
- pdfplumber's advantage is its API giving fine-grained word/bbox access

## Progress Log
- 2026-02-27: Started investigation, setting up environment
- 2026-02-27: Completed benchmark research — found all 5 requested + 4 additional benchmarks
- 2026-02-27: Found 6+ comparison papers/blogs
- 2026-02-27: Installed pymupdf, pdfplumber, pypdf, pdfminer.six, camelot-py
- 2026-02-27: Created 5 custom test PDFs with ground truth JSON
- 2026-02-27: Wrote 62 TDD tests (61 pass, 1 real bug found in camelot-lattice)
- 2026-02-27: Downloaded 4 real-world arXiv PDFs for benchmarking
- 2026-02-27: Ran comprehensive benchmark (9 PDFs × 4 text + 4 table extractors)
- 2026-02-27: Generated 7 visualization charts
