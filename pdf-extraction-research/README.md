# PDF Extraction Benchmarks: Research Report

**Date**: 2026-02-27
**Purpose**: Catalog available benchmarks for evaluating PDF extraction quality in Python, with download instructions, metrics, and comparison resources.

---

## Table of Contents

1. [Summary](#summary)
2. [Primary Benchmarks](#primary-benchmarks)
   - [OmniDocBench](#1-omnidocbench-cvpr-2025)
   - [DP-Bench](#2-dp-bench-upstage)
   - [DocLayNet](#3-doclaynet-ibm-research)
   - [PubTables-1M](#4-pubtables-1m-microsoft-research)
   - [FinTabNet](#5-fintabnet-ibm-research)
   - [SCORE-Bench](#6-score-bench-unstructured)
   - [olmOCR-Bench](#7-olmocr-bench-allen-ai)
   - [gipplab/pdf-benchmark](#8-gipplabpdf-benchmark-göttingen-university)
   - [py-pdf/benchmarks](#9-py-pdfbenchmarks-speed-only)
3. [Metric Glossary](#metric-glossary)
4. [Comparison Papers and Blogs](#comparison-papers-and-blogs)
5. [Evaluation Frameworks](#evaluation-frameworks)
6. [Recommended Starting Points](#recommended-starting-points)

---

## Summary

Nine benchmarks were identified for evaluating PDF extraction quality. The table below gives a quick overview:

| Benchmark | Size | Focus | Publicly Available | Key Metrics |
|-----------|------|-------|--------------------|-------------|
| **OmniDocBench** | 1,355 pages | Full doc parsing (text, tables, formulas, layout) | Yes — HuggingFace | NED, BLEU, TEDS, METEOR |
| **DP-Bench** | 200 pages | Text serialization + table structure | Yes — HuggingFace | NID, TEDS, TEDS-S |
| **DocLayNet** | 80,863 pages | Layout detection | Yes — HuggingFace | mAP, F1, BLEU |
| **PubTables-1M** | ~948K tables | Table extraction only | Yes — HuggingFace / MS Research | GriTS, AP, AR |
| **FinTabNet** | 10,635 tables | Financial table extraction | Yes — IBM Developer + HuggingFace | TEDS, TEDS-S |
| **SCORE-Bench** | Unknown | Enterprise/production documents | Yes — HuggingFace (unstructuredio org) | Adjusted CCT, TEDS |
| **olmOCR-Bench** | 7,010 tests / 1,403 PDFs | PDF linearization unit tests | Yes — GitHub | Pass/Fail unit tests |
| **gipplab/pdf-benchmark** | DocBank-based | Academic PDF metadata/refs/tables | Yes — GitHub | Token-level F1 |
| **py-pdf/benchmarks** | 14 files | Speed only (digital PDFs) | Yes — GitHub | Processing time (s) |

---

## Primary Benchmarks

### 1. OmniDocBench (CVPR 2025)

**The most comprehensive and current benchmark for full document parsing.**

- **Repository**: https://github.com/opendatalab/OmniDocBench
- **Dataset (HuggingFace)**: https://huggingface.co/datasets/opendatalab/OmniDocBench
- **Paper (arXiv)**: https://arxiv.org/abs/2412.07626
- **CVPR 2025 Paper PDF**: https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf

#### Dataset Contents

- **1,355 PDF pages** (v1.0), expanded in v1.5 with more diverse content
- **9 document types**: academic papers, financial reports, newspapers, textbooks, handwritten notes, books, PPTs, test papers, magazines
- **4 layout types**, **3 language types** (English, Chinese, mixed)
- **20,000+ block-level annotations** (15 categories): text paragraphs, titles, tables, figures, formulas, etc.
- **80,000+ span-level annotations** (4 categories): text lines, inline formulas, superscripts/subscripts
- Reading order annotations for all components
- Separate `images/`, `pdfs/`, and `ori_pdfs/` directories for different model input formats

#### Download

```bash
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download opendatalab/OmniDocBench \
  --repo-type dataset \
  --local-dir ./OmniDocBench
```

Expected folder structure after download:
```
OmniDocBench/
├── images/              # JPG page images
├── pdfs/                # PDF versions of same pages
├── ori_pdfs/            # Original PDFs with metadata
├── OmniDocBench.json    # Ground truth annotations
└── with_mask.json       # Pages with masked regions
```

#### Evaluation Setup

```bash
git clone https://github.com/opendatalab/OmniDocBench
cd OmniDocBench
conda create -n omnidocbench python=3.10
conda activate omnidocbench
pip install -r requirements.txt

# Run end-to-end evaluation
# (model output must be .md files matching image filenames)
python run.py --config configs/end2end.yaml
```

#### Metrics

| Metric | Applies To | Description |
|--------|-----------|-------------|
| NED (Normalized Edit Distance) | Text | Character-level text similarity |
| BLEU | Text | N-gram overlap |
| METEOR | Text | Synonym-aware text overlap |
| TEDS | Tables | Tree edit distance (structure + content) |
| COCODet | Layout | Layout element detection accuracy |

Evaluation covers four modules independently: **text**, **tables**, **formulas**, and **reading order**. Results can be filtered by document type, language, or layout attributes.

#### Key Benchmark Results (from paper)

- Pipeline methods (MinerU, Mathpix) outperform on standard document types
- VLMs (GPT-4o, Qwen2-VL) outperform on unusual types (slides, handwriting, degraded scans)
- No single method dominates across all categories

---

### 2. DP-Bench (Upstage)

**Focused benchmark with novel NID metric for text serialization quality. Best for text + table evaluation.**

- **Dataset (HuggingFace)**: https://huggingface.co/datasets/upstage/dp-bench
- **Docling-eval integration**: https://github.com/docling-project/docling-eval/blob/main/docs/DP-Bench_benchmarks.md

#### Dataset Contents

- **200 document pages**
  - 90 samples from Library of Congress (public domain books, historical documents)
  - 90 samples from Open Educational Resources (textbooks, course materials)
  - 20 samples from Upstage internal documents (business documents)
- **12 layout element types**: Table, Paragraph, Figure, Chart, Header, Footer, Caption, Equation, Heading1, List, Index, Footnote
- Wide range of layouts: simple text-only to complex multi-column with tables

#### Download

```python
from datasets import load_dataset
ds = load_dataset("upstage/dp-bench")
```

#### Metrics

**NID (Normalized Indel Distance)**
- Measures how well parsers detect and serialize elements in correct reading order
- Only counts insertions and deletions (excludes substitutions) — sensitive to length differences
- Formula: `NID = 1 − (insertions + deletions) / (len(reference) + len(prediction))`
- Range: 0 (perfect) to 1 (completely dissimilar). **Lower NID = better**
- Note: Tables, figures, charts excluded from NID (evaluated with TEDS instead)

**TEDS (Tree Edit Distance-based Similarity)**
- Measures both table structure AND cell content accuracy
- Formula: `TEDS(Ta, Tb) = 1 − EditDist(Ta, Tb) / max(|Ta|, |Tb|)`
- Range: 0–1. **Higher = better**

**TEDS-S (Structure Only)**
- Same formula as TEDS but ignores cell text content
- Measures only structural accuracy: row/column alignment, merge patterns

#### Benchmark Leaderboard (as of October 2024)

| System | TEDS | TEDS-S | NID | Avg Time (s) |
|--------|------|--------|-----|--------------|
| Upstage (Document AI) | **92.06** | **93.81** | **96.23** | 3.79 |
| AWS Textract | 86.39 | 90.22 | 95.94 | 14.47 |
| Microsoft Azure | 85.54 | 89.07 | 87.03 | 4.44 |
| LlamaParse | 73.36 | 76.29 | 92.22 | 4.14 |
| Google Document AI | 64.64 | 70.95 | 90.09 | 5.85 |
| Unstructured | 64.49 | 69.90 | 90.42 | 13.14 |

---

### 3. DocLayNet (IBM Research)

**The largest human-annotated layout dataset. Used as ground truth for academic comparisons across document categories.**

- **Repository**: https://github.com/DS4SD/DocLayNet
- **HuggingFace (COCO format)**: https://huggingface.co/datasets/ds4sd/DocLayNet
- **HuggingFace (v1.1, with text)**: https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1
- **HuggingFace (v1.2, with embedded PDFs)**: https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2
- **ICDAR 2023 competition version**: https://huggingface.co/datasets/ds4sd/icdar2023-doclaynet
- **Paper (arXiv)**: https://arxiv.org/abs/2206.01062
- **Paper (KDD 2022)**: https://dl.acm.org/doi/10.1145/3534678.3539043

#### Dataset Contents

- **80,863 unique PDF pages** from 6 document categories:
  - Financial Reports (annual reports, earnings releases)
  - Scientific Articles (research papers)
  - Laws & Regulations (legal documents)
  - Government Tenders (procurement documents)
  - Manuals (technical manuals)
  - Patents (patent filings)
- **11 annotation classes**: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
- Languages: Primarily English, some German, French, Japanese
- Splits: 69,103 train / 6,480 validation / 4,994 test
- Some pages have 2–3 independent human annotations for quality measurement

#### Download

```python
from datasets import load_dataset

# Option 1: Original COCO format (bounding boxes only)
ds = load_dataset("ds4sd/DocLayNet")

# Option 2: v1.1 — includes PDF cell text content
ds = load_dataset("ds4sd/DocLayNet-v1.1")

# Option 3: v1.2 — includes embedded PDFs (35+ GB)
ds = load_dataset("ds4sd/DocLayNet-v1.2")
```

```bash
# CLI download (original)
huggingface-cli download ds4sd/DocLayNet --repo-type dataset --local-dir ./DocLayNet
```

**Size warning**: Core zip is ~28 GiB; Extra zip is ~7.5 GiB (needed for text-bounding-box associations). v1.2 with embedded PDFs is ~35.6 GB for train split alone.

#### Metrics

- **mAP** (mean Average Precision): Primary metric for layout detection models
- **F1, BLEU, Local Alignment**: Used in the October 2024 comparative study (arXiv:2410.09871)

#### License

CDLA-Permissive-1.0

---

### 4. PubTables-1M (Microsoft Research)

**The largest table-specific benchmark. Focuses exclusively on table detection, structure recognition, and functional analysis.**

- **Repository**: https://github.com/microsoft/table-transformer
- **HuggingFace**: https://huggingface.co/datasets/bsmock/pubtables-1m
- **Kaggle (structure subset)**: https://www.kaggle.com/datasets/bsmock/pubtables-1m-structure
- **MS Research publication**: https://www.microsoft.com/en-us/research/publication/pubtables-1m/
- **Paper (arXiv)**: https://arxiv.org/abs/2110.00061
- **Paper (CVPR 2022 PDF)**: https://openaccess.thecvf.com/content/CVPR2022/papers/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.pdf

#### Dataset Contents

- **~947,642 fully annotated tables** from PubMed Central scientific articles
- Three tasks, each with separate data splits:
  1. **Table Detection**: Find table bounding boxes in page images (575,305 page images)
  2. **Table Structure Recognition (TSR)**: Parse internal table structure (rows, columns, cells)
  3. **Functional Analysis**: Classify cells as column headers, projected row headers, or data cells
- Full bounding boxes in both **image coordinates** and **PDF coordinates**
- Annotations for rows, columns, cells (including blank cells), headers

#### Download

```bash
# Option 1: HuggingFace (tar.gz files)
# Go to "Files and versions" tab at:
# https://huggingface.co/datasets/bsmock/pubtables-1m
# Download .tar.gz files, then:
bash extract_structure_dataset.sh

# Option 2: Microsoft Research Open Data + azcopy
# Visit https://www.microsoft.com/en-us/research/publication/pubtables-1m/
# Click "Download" to get a unique SAS token URL, then:
azcopy copy "https://msropendataset01.blob.core.windows.net/pubtables1m?[SAS_TOKEN]" \
  "/path/to/download/" --recursive
```

Archive files from Microsoft Research Open Data:
- `PubTables-1M-Image_Page_Detection_PASCAL_VOC.tar.gz` — 575K page images + annotations for detection
- `PubTables-1M-Image_Page_Words_JSON.tar.gz` — Word bounding boxes and text
- `PubTables-1M-PDF_Annotations_JSON.tar.gz` — Detailed annotations in PDF coordinates

#### Metrics

- **GriTS (Grid Table Similarity)**: Custom metric measuring table structure similarity at cell, row, column level. Variants: GriTS_Top (topology), GriTS_Con (content), GriTS_Loc (location)
- Standard detection metrics: AP (Average Precision), AR (Average Recall) at various IoU thresholds

---

### 5. FinTabNet (IBM Research)

**Financial table benchmark. Complex real-world tables with merged cells, multi-level headers.**

- **IBM Developer (official)**: https://developer.ibm.com/data/fintabnet/
- **HuggingFace (FinTabNet.c, cleaned)**: https://huggingface.co/datasets/bsmock/FinTabNet.c
- **Papers With Code**: https://paperswithcode.com/dataset/fintabnet
- **Docling-eval integration**: https://github.com/docling-project/docling-eval/blob/main/docs/FinTabNet_benchmarks.md

#### Dataset Contents

- Tables extracted from Fortune 500 financial earnings documents
- Test set: 10,635 tables (9,681 after removing annotation errors)
- Complex financial structures: merged cells, multi-level column/row headers
- Annotations generated by matching PDF content with corresponding HTML files
- FinTabNet.c (cleaned version on HuggingFace) has stricter quality filtering

#### Download

```bash
# Full dataset — visit IBM Developer page:
# https://developer.ibm.com/data/fintabnet/

# Cleaned version via HuggingFace:
from datasets import load_dataset
ds = load_dataset("bsmock/FinTabNet.c")

# Example files (CDN direct):
wget https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/examples.tar.gz
```

#### Metrics

- TEDS (Tree Edit Distance-based Similarity)
- TEDS-S (structural only)

---

### 6. SCORE-Bench (Unstructured, 2025)

**Enterprise-focused benchmark with real-world production document complexity. Most recent addition.**

- **Announcement blog**: https://unstructured.io/blog/introducing-score-bench-an-open-benchmark-for-document-parsing
- **Evaluation methodology blog**: https://unstructured.io/blog/benchmarking-document-parsing-and-what-actually-matters
- **Results blog**: https://unstructured.io/blog/unstructured-leads-in-document-parsing-quality-benchmarks-tell-the-full-story
- **HuggingFace org**: https://huggingface.co/unstructuredio (search for SCORE-Bench)
- **GitHub org**: https://github.com/Unstructured-IO

#### Dataset Contents

- Real-world enterprise documents, manually annotated by domain experts
- Domains: healthcare, finance, legal, public sector
- Complex challenges: nested tables, merged cells, scanned docs with skew, handwriting, multi-column layouts
- Focuses on documents that break brittle parsing systems

#### Metrics

**Adjusted CCT (Corrected Cell Text)**:
- Word-weighted fuzzy alignment across document elements
- Recognizes semantic equivalence despite structural variation
- Corrects for the fact that multiple valid representations of the same content exist

Additional metrics:
- **Cell Content Accuracy**: Was the text inside a cell correctly extracted?
- **Cell Level Index Accuracy**: Was the cell placed in the correct row/column?
- **TEDS** (corrected implementation)
- Content fidelity, hallucination control, coverage, structural understanding

#### Key Results

- Top pipeline: Unstructured High-Res Refined + Claude Sonnet 4 → **0.773 cell content accuracy**
- Evaluated against: Reducto, LlamaParse, Docling, Snowflake AI_PARSE_DOCUMENT, Databricks ai_parse, NVIDIA nemoretriever-parse

---

### 7. olmOCR-Bench (Allen AI, 2025)

**Novel unit-test approach to evaluating PDF linearization. Binary pass/fail instead of fuzzy matching.**

- **Repository**: https://github.com/allenai/olmocr
- **Paper (arXiv)**: https://arxiv.org/abs/2502.18443
- **Benchmark review**: https://www.llamaindex.ai/blog/olmocr-bench-review-insights-and-pitfalls-on-an-ocr-benchmark

#### Dataset Contents

- **7,010 test instances** across **1,403 PDFs**
- Document types: scientific papers, historical manuscripts, diverse web-crawled PDFs
- Each test is a binary pass/fail rule (e.g., "does this text appear?", "is reading order correct?")
- Categories: text presence, reading order, tables, formulas, baseline functionality

#### Evaluation Approach

```bash
pip install olmocr
# The benchmark is included in the olmocr package
# Run evaluation against any PDF parsing tool
```

#### Metrics

- **Binary unit test pass rate** (not fuzzy matching)
- Separate scores for: text extraction, reading order, table extraction, formula extraction

#### Key Results

olmOCR-7B outperforms GPT-4o, Gemini Flash 2, Qwen-2.5-VL on olmOCR-Bench.

**Known limitation**: Tests were partially generated using Claude Sonnet and Gemini, creating potential bias toward models distilled from those APIs.

---

### 8. gipplab/pdf-benchmark (Göttingen University)

**Academic focus: metadata, references, tables from scientific PDFs. Tests 10 open-source tools.**

- **Repository**: https://github.com/gipplab/pdf-benchmark
- **Paper (arXiv)**: https://arxiv.org/abs/2303.09957

#### Dataset Contents

- Based on **DocBank** dataset: 1.5M annotated content elements from 500K arXiv paper pages
- Tasks: document metadata, bibliographic references, tables, other content elements
- Multi-domain academic documents

#### Tools Evaluated (10)

GROBID, CERMINE, Science Parse, Adobe Extract API, and 6 others

#### Metrics

- Token-level precision, recall, F1
- Per-task evaluation

#### Key Findings

- **GROBID**: Best for metadata and reference extraction
- **Adobe Extract API**: Best for table extraction (though absolute performance still low)
- **All tools**: Poor performance on lists, footers, equations

---

### 9. py-pdf/benchmarks (Speed Only)

**Not a quality benchmark — purely measures text extraction speed on digital PDFs.**

- **Repository**: https://github.com/py-pdf/benchmarks
- **Live results**: https://py-pdf.github.io/pypdf/dev/bench/

#### Scope

- Digital (born-digital) PDFs only
- 14 test files
- Measures wall-clock time for text extraction

#### Speed Results

| Library | Average Time |
|---------|-------------|
| **PyMuPDF** | 0.5s |
| **pypdfium2** | 1.1s |
| **pypdf** | 4.2s |
| **pdfminer.six** | 7.4s |

---

## Metric Glossary

| Metric | Full Name | Description | Range | Better |
|--------|-----------|-------------|-------|--------|
| **NED** | Normalized Edit Distance | Character-level edit distance, normalized | 0–1 | Lower |
| **NID** | Normalized Indel Distance | Like NED but only insertions/deletions; sensitive to length differences | 0–1 | Lower |
| **TEDS** | Tree Edit Distance-based Similarity | Compares predicted vs. ground-truth table trees (structure + content) | 0–1 | Higher |
| **TEDS-S** | TEDS-Struct | Same as TEDS but ignores cell content, structure only | 0–1 | Higher |
| **GriTS** | Grid Table Similarity | Microsoft's metric for table structure at cell/row/column level | 0–1 | Higher |
| **BLEU** | Bilingual Evaluation Understudy | N-gram overlap between extracted and ground-truth text | 0–1 | Higher |
| **METEOR** | Metric for Evaluation of Translation with Explicit ORdering | BLEU variant with synonym and stemming matching | 0–1 | Higher |
| **mAP** | Mean Average Precision | Standard object detection metric for layout bounding boxes | 0–1 | Higher |
| **F1** | F1 Score | Harmonic mean of precision and recall | 0–1 | Higher |
| **Adjusted CCT** | Corrected Cell Text | Unstructured's word-weighted fuzzy cell alignment | 0–1 | Higher |

---

## Comparison Papers and Blogs

### Academic Papers

#### 1. Comparative Study of 10 PDF Parsers on DocLayNet (October 2024)
- **arXiv**: https://arxiv.org/abs/2410.09871
- **Full HTML**: https://arxiv.org/html/2410.09871v1
- **Tools**: PyPDF, pdfminer.six, PyMuPDF, pdfplumber, pypdfium2, Unstructured, Tabula, Camelot, Nougat, Table Transformer (TATR)
- **Dataset**: DocLayNet (80K pages, 6 document categories, 800 docs per category)
- **Metrics**: F1, BLEU, Local Alignment
- **Key Findings**:
  - Text extraction: PyMuPDF and pypdfium2 best overall; all parsers struggle with Scientific/Patent documents
  - Table detection: TATR (deep learning) best across Financial/Patent/Law/Scientific; Camelot best for Government Tenders; PyMuPDF best for Manuals
  - Learning-based tools (Nougat) superior for challenging categories like Scientific/Patent
  - Rule-based tools: faster, interpretable, easier to deploy; better for known document types

#### 2. Formula Extraction Benchmark (December 2024)
- **arXiv**: https://arxiv.org/abs/2512.09874
- **HTML**: https://arxiv.org/html/2512.09874v1
- **Novel approach**: Synthetic PDFs with known LaTeX ground truth + LLM-as-judge evaluation (correlates with human judgment)
- **Key Findings**:
  - State-of-the-art VLMs (Qwen3-VL, Gemini 3 Pro) and specialized OCR (PaddleOCR-VL, Mathpix): scores >9.6/10
  - Traditional rule-based parsers: much lower scores
  - LLM-as-judge correlates better with human evaluation than character-level string matching

#### 3. Docling Technical Report (January 2025)
- **arXiv**: https://arxiv.org/abs/2408.09869
- **Speed Benchmarks** (page processing time):

  | Tool | x86 CPU | M3 Max (SoC) | CUDA L4 (GPU) |
  |------|---------|--------------|----------------|
  | MinerU | 3.3 s/page | N/A | **0.21 s/page** |
  | **Docling** | **3.1 s/page** | **1.27 s/page** | 0.49 s/page |
  | Unstructured | 4.2 s/page | 2.7 s/page | No GPU benefit |
  | Marker | 16+ s/page | 4.2 s/page | 0.86 s/page |

#### 4. olmOCR Paper (February 2025)
- **arXiv**: https://arxiv.org/abs/2502.18443
- **Website**: https://olmocr.allenai.org/
- Introduces fine-tuned Qwen2-VL-7B model + benchmark of 7,010 unit tests
- olmOCR-7B outperforms GPT-4o, Gemini Flash 2, Qwen-2.5-VL on olmOCR-Bench

### Blog Posts and Practical Comparisons

#### Procycons PDF Data Extraction Benchmark 2025
- **URL**: https://procycons.com/en/blogs/pdf-data-extraction-benchmark/
- **Tools**: Docling, Unstructured, LlamaParse
- **Dataset**: 5 corporate sustainability reports (diverse complexity, 4,500–34,000 words each)
- **Results**:
  - Docling: 97.9% complex table accuracy, best overall quality
  - LlamaParse: Fastest (~6s constant regardless of doc size), struggles with multi-column and complex tables
  - Unstructured: 100% simple tables, 75% complex tables, slowest

#### 12 Open-Source Parser Comparison
- **URL**: https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html
- **Tools**: MinerU, PaddleOCR, Marker, Unstructured, Zerox, Sparrow, pdf-extract-api, DFlux, Mathpix, and more
- **Results**: MinerU best all-rounder; Zerox best OCR; LlamaParse best table extraction; MarkItDown fastest

#### Reducto Document Parser Comparison
- **URL**: https://llms.reducto.ai/document-parser-comparison
- **Tools**: Docling, LlamaParse, Unstructured, Reducto
- Reducto self-comparison (potential bias, but useful for methodology reference)

#### "I Tested 7 Python PDF Extractors" (2025 Edition)
- **URL**: https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257
- Practical hands-on test; recommends pymupdf4llm for general use, unstructured for RAG chunks, pypdfium2 for speed

#### PDF Parsing for LLM Input (Nicolas' Notebook)
- **URL**: https://nbrosse.github.io/posts/pdf-parsing/pdf-parsing.html
- Qualitative comparison: Docling, Marker, MinerU, PyMuPDF vs. LlamaParse, Gemini, Mistral
- Diverse test PDFs: slides, reports, scanned documents, complex tables

#### PyMuPDF Performance Comparison (Official Docs)
- **URL**: https://pymupdf.readthedocs.io/en/latest/app4.html
- Official methodology for benchmarking PyMuPDF vs. competitors

#### Best Python PDF Libraries (Unstract, 2026 update)
- **URL**: https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/
- Updated 2026 comparison of PDF-to-text libraries

---

## Evaluation Frameworks

### docling-eval

The most structured open-source evaluation framework. Supports multiple benchmarks and tools.

- **Repository**: https://github.com/docling-project/docling-eval
- **DP-Bench integration**: https://github.com/docling-project/docling-eval/blob/main/docs/DP-Bench_benchmarks.md
- **FinTabNet integration**: https://github.com/docling-project/docling-eval/blob/main/docs/FinTabNet_benchmarks.md
- **PubTabNet integration**: https://github.com/docling-project/docling-eval/blob/main/docs/PubTabNet_benchmarks.md

**Supported benchmarks**: OmniDocBench, DocLayNetV1, FinTabNet, PubTabNet, PubTables-1M (Pub1M)
**Evaluation modalities**: layout, table structure, reading order, markdown text

```bash
pip install docling-eval

# Create ground truth dataset
docling-eval create-gt --dataset dp-bench --output-dir ./gt

# Generate predictions with docling
docling-eval create-eval --dataset dp-bench --output-dir ./eval

# Evaluate
docling-eval evaluate --dataset dp-bench --gt-dir ./gt --eval-dir ./eval

# Visualize results
docling-eval visualize --dataset dp-bench
```

### OmniDocBench Evaluation Code

Included in the main OmniDocBench repository:

```bash
git clone https://github.com/opendatalab/OmniDocBench
cd OmniDocBench
pip install -r requirements.txt
python run.py --config configs/end2end.yaml
```

Supports: end-to-end evaluation, md2md evaluation, filtered evaluation by document type/language/attribute.

---

## Recommended Starting Points

### For a Quick Start (Smallest Dataset)

**DP-Bench** is the easiest entry point:
- Only 200 pages (manageable size)
- Clear metrics (NID, TEDS, TEDS-S)
- Pre-existing leaderboard for comparison
- Simple download via `load_dataset("upstage/dp-bench")`

### For Comprehensive Evaluation

**OmniDocBench** is the gold standard:
- Covers all content types (text, tables, formulas, layout, reading order)
- 9 document categories
- Active development, CVPR 2025 acceptance
- Built-in evaluation scripts

### For Table-Specific Testing

Use **FinTabNet** (financial complexity) or **PubTables-1M** (scale) depending on target domain.

### For Production/Enterprise Documents

**SCORE-Bench** tests on real-world production complexity (scanned docs, complex tables, diverse domains).

### For Speed Benchmarking

**py-pdf/benchmarks** for digital PDF processing speed; **Docling Technical Report** for AI-based tool speed.

---

## Quick Summary: Library Recommendations

Based on the benchmarks reviewed:

| Use Case | Recommended Library | Why |
|----------|--------------------|----|
| General text extraction (speed) | PyMuPDF / pymupdf4llm | Fastest rule-based, good quality |
| RAG/LLM chunking | unstructured | Best semantic chunking |
| Table extraction (high accuracy) | Docling | 97.9% complex table accuracy |
| Full document structure | MinerU or Docling | Best GPU-accelerated pipeline |
| Scientific papers with formulas | MinerU or Marker | Formula-to-LaTeX conversion |
| Scanned documents | Zerox or Unstructured | Best OCR capabilities |
| Large-scale batch processing | olmOCR-7B | Cost-efficient at $176/million pages |

---

*Research conducted: 2026-02-27*
*Sources: OmniDocBench (CVPR 2025), DP-Bench (Upstage), DocLayNet (IBM/KDD 2022), PubTables-1M (Microsoft/CVPR 2022), FinTabNet (IBM), SCORE-Bench (Unstructured), olmOCR-Bench (Allen AI), gipplab/pdf-benchmark (Göttingen University), py-pdf/benchmarks*
