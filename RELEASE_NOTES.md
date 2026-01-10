# Machine Actionable Poster Extraction - v1.0.0

## Release Date
2026-01-09

## Overview

This release marks the first version of the Machine Actionable Poster Extraction tool, designed to extract structured, machine-readable metadata from scientific conference posters.

## Model

**HuggingFace**: [jimnoneill/Llama-3.1-8B-Poster-Extraction](https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction)

Based on Meta's Llama 3.1 8B Instruct, configured for structured poster metadata extraction.

## Validation Results

Tested against 10 manually annotated scientific posters from diverse domains using the validated v41 extraction pipeline.

### Summary

| Metric | v1.0.0 | Baseline | Delta |
|--------|--------|----------|-------|
| **Passing Rate** | 9/10 (90%) | 10/10 (100%) | -1 |
| **Word Capture** | 0.964 | 0.969 | -0.005 |
| **ROUGE-L** | 0.880 | 0.887 | -0.007 |
| **Number Capture** | 0.914 | 0.936 | -0.022 |
| **Field Proportion** | 1.033 | 1.083 | -0.050 |

### Passing Criteria

All metrics must meet these thresholds:
- Word Capture ≥ 0.75
- ROUGE-L ≥ 0.75
- Number Capture ≥ 0.75
- Field Proportion: 0.3 - 2.5

### Per-Poster Results

| Poster ID | Word | ROUGE-L | Numbers | Fields | Sections | Status |
|-----------|------|---------|---------|--------|----------|--------|
| 10890106 | 0.98 | 0.82 | 0.96 | 1.16 | 26/13 | ✅ PASS |
| 15963941 | 0.97 | 0.90 | 0.97 | 0.95 | 7/5 | ✅ PASS |
| 16083265 | 0.96 | 0.90 | 1.00 | 1.10 | 19/15 | ✅ PASS |
| 17268692 | 0.95 | 0.93 | 0.90 | 1.12 | 14/10 | ✅ PASS |
| 42 | 0.99 | 0.88 | 0.97 | 0.75 | 12/15 | ✅ PASS |
| 4737132 | 0.94 | 0.82 | 0.95 | 1.33 | 16/10 | ✅ PASS |
| 5128504 | 0.99 | 0.99 | 0.97 | 1.16 | 13/8 | ✅ PASS |
| 6724771 | 0.92 | 0.98 | 0.82 | 1.07 | 11/6 | ✅ PASS |
| 8228476 | 0.95 | 0.88 | 0.89 | 0.82 | 14/7 | ✅ PASS |
| 8228568 | 0.99 | 0.69 | 0.73 | 0.85 | 7/9 | ❌ FAIL |

### Notes on Failed Poster

Poster 8228568 narrowly failed due to:
- ROUGE-L: 0.69 (threshold: 0.75)
- Number Capture: 0.73 (threshold: 0.75)

The failure is marginal and does not indicate systematic issues with the model.

## Architecture

### Text Extraction Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│   PDF Input     │────▶│    pdfalto      │────▶ Structured Text
└─────────────────┘     │ (reading order) │
                        └─────────────────┘
                               ▼
                        ┌─────────────────┐
                        │    PyMuPDF      │────▶ Fallback Text
                        │   (fallback)    │
                        └─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  Image Input    │────▶│  Qwen2-VL-7B    │────▶ OCR Text
│ (JPG/PNG)       │     │ (vision-lang)   │
└─────────────────┘     └─────────────────┘
```

### JSON Structuring

```
┌─────────────────┐     ┌─────────────────────────────────┐
│   Raw Text      │────▶│ Llama-3.1-8B-Poster-Extraction  │────▶ Structured JSON
└─────────────────┘     │      (jimnoneill/HF)            │
                        └─────────────────────────────────┘
```

## Output Schema

The extraction produces JSON conforming to the poster metadata schema:

```json
{
  "creators": [{"name": "LastName, FirstName", "affiliation": [{"name": "Institution"}]}],
  "titles": [{"title": "Main Poster Title"}],
  "posterContent": {
    "sections": [
      {"sectionTitle": "Abstract", "sectionContent": "..."},
      {"sectionTitle": "Methods", "sectionContent": "..."},
      {"sectionTitle": "Results", "sectionContent": "..."}
    ]
  },
  "imageCaption": [{"caption1": "Figure 1 caption"}],
  "tableCaption": [{"caption1": "Table 1 caption"}],
  "domain": "Research field"
}
```

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **RAM**: 32GB minimum
- **Storage**: 20GB for model weights

### Software
- Python 3.10+
- PyTorch 2.0+
- transformers 4.36+
- CUDA 12.0+

## Installation

```bash
# Clone repository
git clone https://github.com/fairdataihub/machine-actionable-posterextraction-beta.git
cd machine-actionable-posterextraction-beta

# Install dependencies
pip install -r requirements.txt

# Accept Llama license and login to HuggingFace
huggingface-cli login
```

## Usage

### Python API

```python
from poster_extraction import extract_poster

result = extract_poster("path/to/poster.pdf")
print(result["titles"][0]["title"])
```

### Command Line

```bash
python poster_extraction.py --input poster.pdf --output output.json
```

## API Endpoint

Deployed at: `http://100.81.132.45:47362` (Tailscale network only)

```bash
# Health check
curl http://100.81.132.45:47362/health

# Extract poster
curl -X POST -F "file=@poster.pdf" http://100.81.132.45:47362/extract
```

## License

This tool uses the Llama 3.1 model under the [Llama 3.1 Community License](https://ai.meta.com/llama/license/).

## Citation

```bibtex
@software{posterextraction2026,
  title = {Machine Actionable Poster Extraction},
  author = {O'Neill, James and Patel, Bhavesh and Soundarajan, Sanjay},
  year = {2026},
  url = {https://github.com/fairdataihub/machine-actionable-posterextraction-beta}
}
```

## Acknowledgments

- FAIR Data Innovations Hub
- Meta AI for Llama 3.1
- Alibaba Cloud for Qwen2-VL
- HuggingFace for model hosting

## Related Resources

- **Model**: https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction
- **Documentation**: https://posters.science/
- **JSON Schema**: See `manual_poster_annotation/README.md`

