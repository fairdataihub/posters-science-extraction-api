# Poster Science - Poster Extraction Beta

Automated extraction of structured metadata from scientific poster PDFs and images using Large Language Models.

## Overview

This pipeline converts scientific posters (PDF and image formats) into structured JSON following the [posters-science JSON schema](https://github.com/fairdataihub/posters-science-schema). The system achieves **100% compliance** (10/10 posters) on validation metrics with a ≥0.75 threshold across all measures.

## Models Used

This pipeline leverages the following Large Language Models:

| Model | Provider | Parameters | Purpose |
|-------|----------|------------|---------|
| **Llama 3.1 8B Instruct** | Meta AI | 8B | JSON structuring and text-to-schema conversion |
| **Qwen2-VL-7B-Instruct** | Alibaba | 7B | Vision-language OCR for image posters |

### Meta Llama 3.1 8B Instruct

The core JSON structuring is performed by [Meta's Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), selected for:
- Strong instruction-following capabilities for structured output generation
- 128K context window supporting full poster text processing
- Efficient inference on consumer GPUs (16GB+ VRAM)

### Qwen2-VL-7B-Instruct

Image-based posters (JPG/PNG) are processed using [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), a vision-language model that provides:
- Direct pixel-to-text extraction without traditional OCR preprocessing
- Multi-language support for international poster content
- Layout-aware text recognition preserving reading order

## Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Input Poster   │────▶│  Raw Text       │────▶│  Structured     │
│  (PDF/Image)    │     │  Extraction     │     │  JSON Output    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                    ┌─────────┴─────────┐    ┌────────┴────────┐
                    │                   │    │                 │
               [PDF Files]        [Image Files]    [Llama 3.1 8B]
                    │                   │              │
               [pdfalto]         [Qwen2-VL-7B]   Section-aware
               XML Layout        Vision OCR      JSON Generation
```

### Stage 1: Raw Text Extraction

The pipeline automatically selects the extraction method based on input file type:

**PDF Files → pdfalto**
- Converts PDF to ALTO XML format preserving layout structure
- Extracts text blocks with spatial coordinates
- Maintains reading order through XML hierarchy analysis
- Handles multi-column layouts and complex poster designs

**Image Files → Qwen2-VL-7B-Instruct**
- Loads image directly into vision-language model
- Generates text transcription via multimodal inference
- Prompt: "Extract ALL text from this scientific poster image exactly as written"
- Outputs raw text preserving section headers and content

### Stage 2: JSON Structuring (Llama 3.1 8B)

Raw text is converted to structured JSON using Meta's Llama 3.1 8B Instruct:

**Primary Prompt Strategy**
- Section-aware extraction identifying: Abstract, Introduction, Methods, Results, Key Findings, Discussion, Conclusions, References, Contact
- Explicit disambiguation between semantically similar sections (e.g., "Key Findings" vs "References")
- Verbatim text preservation instructions to maintain scientific accuracy

**Adaptive Fallback Mechanism**
1. Initial generation with 18,000 output tokens
2. If truncation detected → retry with 24,000 tokens
3. If still truncating → switch to condensed prompt format (saves input tokens for output)

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

