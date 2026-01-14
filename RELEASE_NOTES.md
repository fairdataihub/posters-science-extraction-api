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

| Metric | v1.0.0 |
|--------|--------|
| **Passing Rate** | 9/10 (90%) |
| **Word Capture** | 0.954 |
| **ROUGE-L** | 0.885 |
| **Number Capture** | 0.929 |
| **Field Proportion** | 0.996 |

### Passing Criteria

All metrics must meet these thresholds:
- Word Capture ≥ 0.75
- ROUGE-L ≥ 0.75
- Number Capture ≥ 0.75
- Field Proportion: 0.3 - 2.5

### Per-Poster Results

| Poster ID | Word | ROUGE-L | Numbers | Fields | Sections | Status |
|-----------|------|---------|---------|--------|----------|--------|
| 10890106 | 0.98 | 0.85 | 1.00 | 0.88 | 17/13 | ✅ PASS |
| 15963941 | 0.98 | 0.93 | 1.00 | 0.84 | 5/5 | ✅ PASS |
| 16083265 | 0.90 | 0.90 | 0.82 | 0.92 | 14/15 | ✅ PASS |
| 17268692 | 1.00 | 0.83 | 1.00 | 1.76 | 14/10 | ✅ PASS |
| 42 | 0.99 | 0.88 | 1.00 | 0.77 | 12/15 | ✅ PASS |
| 4737132 | 0.89 | 0.74 | 0.91 | 1.10 | 12/10 | ❌ FAIL |
| 5128504 | 0.99 | 1.00 | 1.00 | 1.05 | 12/8 | ✅ PASS |
| 6724771 | 0.89 | 0.95 | 0.85 | 0.96 | 9/6 | ✅ PASS |
| 8228476 | 0.94 | 0.87 | 0.89 | 0.90 | 16/7 | ✅ PASS |
| 8228568 | 0.99 | 0.91 | 0.82 | 0.78 | 6/9 | ✅ PASS |

### Notes on Failed Poster

Poster 4737132 (image poster processed via Qwen Vision OCR) narrowly failed due to:
- ROUGE-L: 0.74 (threshold: 0.75)

The failure is marginal and reflects the inherent challenge of OCR-based text extraction from image posters compared to PDF extraction via pdfalto.

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

