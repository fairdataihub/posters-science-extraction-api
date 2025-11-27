# Poster Science - Poster Extraction Beta

Automated extraction of structured metadata from scientific poster PDFs and images using Large Language Models.

## Overview

This pipeline converts scientific posters (PDF and image formats) into structured JSON following the [posters-science JSON schema](https://github.com/fairdataihub/posters-science-schema). The system achieves **100% compliance** (10/10 posters) on validation metrics with a ≥0.75 threshold across all measures.

## Architecture

### Two-Stage Pipeline

**Stage 1: Raw Text Extraction**
- **PDF files**: `pdfalto` - XML-based layout analysis that preserves document structure and reading order
- **Image files (JPG/PNG)**: `Qwen2-VL-7B` - Vision-language model for optical character recognition

**Stage 2: JSON Structuring**
- **Model**: `Meta Llama 3.1 8B Instruct`
- **Strategy**: Dual-prompt architecture with adaptive fallback for token-constrained inputs
  - Primary prompt: Section-aware extraction with explicit header disambiguation
  - Fallback prompt: Condensed format for documents requiring extended output generation

## Evaluation Metrics

The pipeline is validated against manually annotated reference JSONs using four complementary metrics:

| Metric | Description | Threshold | Rationale |
|--------|-------------|-----------|-----------|
| **Word Capture (w)** | Proportion of reference vocabulary present in extracted text | ≥0.75 | Measures lexical completeness |
| **ROUGE-L (r)** | Longest common subsequence similarity with section-aware matching | ≥0.75 | Captures sequential text preservation |
| **Number Capture (n)** | Proportion of numeric values preserved | ≥0.75 | Validates quantitative data integrity |
| **Field Proportion (f)** | Ratio of extracted to reference JSON structural elements | 0.30–2.50 | Accommodates layout variability |

### Metric Implementation

**Text Normalization**
- Unicode normalization (NFKD) for character standardization
- Whitespace consolidation and trimming
- Quotation mark and dash character unification across encoding variants

**Section-Aware ROUGE-L**
- Computes pairwise similarity between extracted and reference sections
- Returns maximum of global document score and section-averaged score
- Accounts for structural reorganization in poster layouts

**Field Proportion Range**
- Extended acceptance range (0.30–2.50) accommodates inherent variability in poster organization
- Some posters contain nested subsections; others use flat structures
- Metric validates structural completeness without penalizing format differences

**Number Capture Filtering**
- Excludes DOI components and publication years from reference sections
- Focuses on scientifically meaningful numeric content (measurements, statistics, counts)

## Validation Results

**Pipeline v41 - Production Release**: 10/10 (100%) passing

| Poster ID | Word | ROUGE-L | Numbers | Fields | OCR Method |
|-----------|------|---------|---------|--------|------------|
| 10890106 | 0.97 | 0.81 | 0.96 | 0.90 | pdfalto |
| 15963941 | 0.97 | 0.90 | 0.97 | 0.95 | pdfalto |
| 16083265 | 0.98 | 0.89 | 1.00 | 0.96 | pdfalto |
| 17268692 | 1.00 | 0.87 | 0.94 | 1.91 | pdfalto |
| 42 | 0.99 | 0.89 | 0.97 | 0.76 | pdfalto |
| 4737132 | 0.94 | 0.84 | 0.95 | 1.32 | qwen_vision |
| 5128504 | 0.99 | 0.99 | 0.97 | 1.16 | pdfalto |
| 6724771 | 0.91 | 0.95 | 0.82 | 1.05 | pdfalto |
| 8228476 | 0.95 | 0.90 | 0.89 | 0.86 | pdfalto |
| 8228568 | 0.99 | 0.82 | 0.91 | 0.96 | pdfalto |

**Aggregate Performance**: w=0.969, r=0.887, n=0.936, f=1.083

## Usage

```bash
# Activate environment
source ~/myenv/bin/activate

# Run extraction
python poster_extraction.py \
    --annotation-dir "./manual_poster_annotation" \
    --output-dir "./output"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--annotation-dir` | Directory containing poster PDFs/images | Required |
| `--output-dir` | Directory for extracted JSON outputs | Required |

## System Requirements

### Hardware
- CUDA-capable GPU with ≥16GB VRAM (tested on NVIDIA RTX 4090)
- Sufficient system RAM for model loading (~32GB recommended)

### Software
- Python 3.10+
- CUDA 11.8+ with compatible drivers

### Dependencies

```
transformers>=4.40.0
torch>=2.0.0
rouge-score
qwen-vl-utils
accelerate
Pillow
numpy
```

### External Tools
- `pdfalto` - PDF layout analysis tool (compiled binary required)
  - Installation: https://github.com/kermitt2/pdfalto

## Output Structure

```
output/
├── {poster_id}_raw.txt        # Extracted raw text from OCR
├── {poster_id}_extracted.json # Structured JSON per schema
└── results.json               # Evaluation metrics summary
```

### JSON Schema

Output JSONs conform to the posters-science schema:

```json
{
  "creators": [{"name": "LastName, FirstName", "affiliation": [{"name": "Institution"}]}],
  "titles": [{"title": "Poster Title"}],
  "posterContent": {
    "posterTitle": "Poster Title",
    "sections": [
      {"sectionTitle": "Abstract", "sectionContent": "..."},
      {"sectionTitle": "Methods", "sectionContent": "..."}
    ]
  },
  "imageCaption": [{"caption1": "Figure 1 description"}],
  "tableCaption": [{"caption1": "Table 1 description"}]
}
```

## Directory Structure

```
posters-science-posterextraction-beta/
├── README.md
├── poster_extraction.py       # Main extraction pipeline
├── manual_poster_annotation/  # Reference posters and ground truth JSONs
│   ├── {poster_id}/
│   │   ├── {poster_id}.pdf    # Source poster
│   │   └── {poster_id}_sub-json.json  # Reference annotation
└── example_output/            # Sample extraction results
```

## Methodology

### OCR Selection Logic

The pipeline automatically selects the appropriate OCR method based on file type:
- **PDF files**: Processed via `pdfalto` which preserves layout structure through XML intermediate representation
- **Image files**: Processed via `Qwen2-VL-7B` vision-language model for direct pixel-to-text conversion

### Prompt Engineering

The JSON structuring stage employs section-aware prompting that:
1. Explicitly enumerates common scientific poster sections (Abstract, Introduction, Methods, Results, Discussion, Conclusions, References)
2. Distinguishes between semantically similar sections (e.g., "Key Findings" vs "References")
3. Instructs verbatim text preservation to maintain scientific accuracy

### Truncation Handling

For documents exceeding model context limits:
1. Initial attempt with primary prompt (18,000 output tokens)
2. Retry with extended token budget (24,000 tokens)
3. Fallback to condensed prompt format if truncation persists

## License

MIT License

## Citation

Part of the [FAIR Data Innovations Hub](https://fairdataihub.org/) posters-science project.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

