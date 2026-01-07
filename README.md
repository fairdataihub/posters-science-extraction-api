# FAIR Data Innovations Hub - Mchine-Actionable Poster Extraction Beta

Automated extraction of structured metadata from scientific poster PDFs and images using Large Language Models.

## Overview

This pipeline converts scientific posters (PDF and image formats) into structured JSON following the [posters-science JSON schema](https://github.com/fairdataihub/posters-science-schema). The system achieves **100% compliance** (10/10 posters) on validation metrics with a ≥0.75 threshold across all measures.

## Models Used

This pipeline leverages the following Large Language Models:

| Model                     | Provider | Parameters | Purpose                                        |
| ------------------------- | -------- | ---------- | ---------------------------------------------- |
| **Llama 3.1 8B Instruct** | Meta AI  | 8B         | JSON structuring and text-to-schema conversion |
| **Qwen2-VL-7B-Instruct**  | Alibaba  | 7B         | Vision-language OCR for image posters          |

### Meta Llama 3.1 8B Instruct (via Ollama)

The core JSON structuring is performed by [Meta's Llama 3.1 8B Instruct](https://ollama.ai/library/llama3.1) served through [Ollama](https://ollama.ai), selected for:

The core JSON structuring is performed by [Llama 3.1 8B Poster Extraction](https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction), selected for:
- Strong instruction-following capabilities for structured output generation
- 128K context window supporting full poster text processing
- Efficient inference on consumer GPUs (16GB+ VRAM)
- Simplified deployment via Ollama without HuggingFace authentication

The pipeline uses the Q8 quantized variant (`llama3.1:8b-instruct-q8_0`) for optimal balance between quality and speed.

### Qwen2-VL-7B-Instruct

Image-based posters (JPG/PNG) are processed using [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), a vision-language model that provides:

- Direct pixel-to-text extraction without traditional OCR preprocessing
- Multi-language support for international poster content
- Layout-aware text recognition preserving reading order

## Architecture

### Pipeline Overview

```mermaid
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Input Poster   │────▶│  Raw Text       │────▶│  Structured     │
│  (PDF/Image)    │     │  Extraction     │     │  JSON Output    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                    ┌─────────┴─────────┐    ┌────────┴────────┐
                    │                   │    │                 │
               [PDF Files]        [Image Files]    [Ollama]
                    │                   │         Llama 3.1 8B
               [pdfalto]         [Qwen2-VL-7B]   Section-aware
               XML Layout        Vision OCR      JSON Generation
```

### Stage 1: Raw Text Extraction

The pipeline automatically selects the extraction method based on input file type:

#### PDF Files → pdfalto

- Converts PDF to ALTO XML format preserving layout structure
- Extracts text blocks with spatial coordinates
- Maintains reading order through XML hierarchy analysis
- Handles multi-column layouts and complex poster designs

#### Image Files → Qwen2-VL-7B-Instruct

- Loads image directly into vision-language model
- Generates text transcription via multimodal inference
- Prompt: "Extract ALL text from this scientific poster image exactly as written"
- Outputs raw text preserving section headers and content

### Stage 2: JSON Structuring (Ollama + Llama 3.1 8B)

Raw text is converted to structured JSON using Meta's Llama 3.1 8B Instruct via Ollama:

#### Text Preprocessing

- Removes/replaces problematic quote characters that cause JSON parsing issues
- Normalizes smart quotes and curly quotes to prevent double-quote artifacts

#### Primary Prompt Strategy

#### Primary Prompt Strategy

- Section-aware extraction identifying: Abstract, Introduction, Methods, Results, Key Findings, Discussion, Conclusions, References, Contact
- Explicit disambiguation between semantically similar sections (e.g., "Key Findings" vs "References")
- Verbatim text preservation instructions to maintain scientific accuracy

#### Adaptive Fallback Mechanism

1. Initial generation with 18,000 output tokens
2. If truncation detected → retry with 24,000 tokens
3. If still truncating → switch to condensed prompt format (saves input tokens for output)

## Evaluation Metrics

The pipeline is validated against manually annotated reference JSONs using four complementary metrics:

| Metric                   | Description                                                       | Threshold | Rationale                             |
| ------------------------ | ----------------------------------------------------------------- | --------- | ------------------------------------- |
| **Word Capture (w)**     | Proportion of reference vocabulary present in extracted text      | ≥0.75     | Measures lexical completeness         |
| **ROUGE-L (r)**          | Longest common subsequence similarity with section-aware matching | ≥0.75     | Captures sequential text preservation |
| **Number Capture (n)**   | Proportion of numeric values preserved                            | ≥0.75     | Validates quantitative data integrity |
| **Field Proportion (f)** | Ratio of extracted to reference JSON structural elements          | 0.30–2.50 | Accommodates layout variability       |

### Metric Implementation

#### Text Normalization

- Unicode normalization (NFKD) for character standardization
- Whitespace consolidation and trimming
- Quotation mark and dash character unification across encoding variants

#### Section-Aware ROUGE-L

- Computes pairwise similarity between extracted and reference sections
- Returns maximum of global document score and section-averaged score
- Accounts for structural reorganization in poster layouts

#### Field Proportion Range

- Extended acceptance range (0.30–2.50) accommodates inherent variability in poster organization
- Some posters contain nested subsections; others use flat structures
- Metric validates structural completeness without penalizing format differences

#### Number Capture Filtering

- Excludes DOI components and publication years from reference sections
- Focuses on scientifically meaningful numeric content (measurements, statistics, counts)

## Validation Results

**Production Release**: 10/10 (100%) passing

| Poster ID | Word | ROUGE-L | Numbers | Fields | OCR Method  |
| --------- | ---- | ------- | ------- | ------ | ----------- |
| 10890106  | 0.97 | 0.81    | 0.96    | 0.90   | pdfalto     |
| 15963941  | 0.97 | 0.90    | 0.97    | 0.95   | pdfalto     |
| 16083265  | 0.98 | 0.89    | 1.00    | 0.96   | pdfalto     |
| 17268692  | 1.00 | 0.87    | 0.94    | 1.91   | pdfalto     |
| 42        | 0.99 | 0.89    | 0.97    | 0.76   | pdfalto     |
| 4737132   | 0.94 | 0.84    | 0.95    | 1.32   | qwen_vision |
| 5128504   | 0.99 | 0.99    | 0.97    | 1.16   | pdfalto     |
| 6724771   | 0.91 | 0.95    | 0.82    | 1.05   | pdfalto     |
| 8228476   | 0.95 | 0.90    | 0.89    | 0.86   | pdfalto     |
| 8228568   | 0.99 | 0.82    | 0.91    | 0.96   | pdfalto     |

**Aggregate Performance**: w=0.969, r=0.887, n=0.936, f=1.083

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fairdataihub/posters-science-posterextraction-beta.git
cd posters-science-posterextraction-beta
```

### 2. Install Ollama (Required)

Ollama is used to serve the Llama 3.1 8B model locally.

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.ai/download

1. Create a HuggingFace account at https://huggingface.co
2. Accept the Llama 3.1 license at https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set the environment variable:

**Start Ollama server (if not running as service):**
```bash
ollama serve
```

### 3. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 4. Install pdfalto (Required)

`pdfalto` is required for PDF text extraction with layout preservation.

**Option A: Build from source**
```bash
git clone https://github.com/kermitt2/pdfalto.git
cd pdfalto
mkdir build && cd build
cmake ..
make
# Binary will be at: pdfalto/build/pdfalto
```

**Option B: Download pre-built binary**
- Check releases at https://github.com/kermitt2/pdfalto/releases

**Configure the path (one of the following):**

```bash
# Option 1: Set environment variable
export PDFALTO_PATH="/path/to/pdfalto/build/pdfalto"

# Option 2: Add to system PATH
sudo cp /path/to/pdfalto/build/pdfalto /usr/local/bin/

# Option 3: Place in auto-discovered location
cp /path/to/pdfalto/build/pdfalto ~/Downloads/pdfalto
```

The pipeline automatically searches these locations:
- `PDFALTO_PATH` environment variable
- System PATH (`which pdfalto`)
- `/usr/local/bin/pdfalto`
- `/usr/bin/pdfalto`
- `~/Downloads/pdfalto`

## Usage

### Basic Usage

```bash
python poster_extraction.py \
    --annotation-dir "./posters" \
    --output-dir "./output"
```

### With Environment Variables

```bash
# Specify GPU device
CUDA_VISIBLE_DEVICES=0 python poster_extraction.py --annotation-dir ./posters

# Custom pdfalto location
PDFALTO_PATH=/opt/pdfalto/pdfalto python poster_extraction.py --annotation-dir ./posters

# Full example
PDFALTO_PATH="/usr/local/bin/pdfalto" \
CUDA_VISIBLE_DEVICES=0 \
python poster_extraction.py \
    --annotation-dir "./manual_poster_annotation" \
    --output-dir "./extraction_output"
```

### Command Line Arguments

| Argument           | Description                             | Default  |
| ------------------ | --------------------------------------- | -------- |
| `--annotation-dir` | Directory containing poster PDFs/images | Required |
| `--output-dir`     | Directory for extracted JSON outputs    | Required |

## System Requirements

### Hardware

- CUDA-capable GPU with ≥16GB VRAM (tested on NVIDIA RTX 4090)
- Sufficient system RAM for model loading (~32GB recommended)

### Software

- Python 3.10+
- CUDA 11.8+ with compatible drivers
- Ollama (latest version)
- Linux, macOS, or Windows with WSL2

### Python Dependencies

```bash
transformers>=4.40.0
torch>=2.0.0
rouge-score
qwen-vl-utils
accelerate
Pillow
numpy
ollama>=0.2.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### External Tools

- `pdfalto` - PDF layout analysis tool (compiled binary required)
  - Installation: https://github.com/kermitt2/pdfalto

## Output Structure

```bash
output/
├── {poster_id}_raw.txt        # Extracted raw text from OCR
├── {poster_id}_extracted.json # Structured JSON per schema
└── results.json               # Evaluation metrics summary
```

### JSON Schema

Output JSONs conform to the posters-science schema:

```json
{
  "creators": [
    {
      "name": "LastName, FirstName",
      "affiliation": [{ "name": "Institution" }]
    }
  ],
  "titles": [{ "title": "Poster Title" }],
  "posterContent": {
    "posterTitle": "Poster Title",
    "sections": [
      { "sectionTitle": "Abstract", "sectionContent": "..." },
      { "sectionTitle": "Methods", "sectionContent": "..." }
    ]
  },
  "imageCaption": [{ "caption1": "Figure 1 description" }],
  "tableCaption": [{ "caption1": "Table 1 description" }]
}
```

## Directory Structure

```bash
posters-science-posterextraction-beta/
├── README.md
├── poster_extraction.py       # Main extraction pipeline
├── requirements.txt           # Python dependencies
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

### Text Preprocessing

Before JSON structuring, the raw text undergoes preprocessing to handle characters that cause parsing issues:

- Removes curly/smart quotes that Ollama tends to include literally in output
- Normalizes quote characters to prevent double-quote JSON artifacts
- This preprocessing step is specific to the Ollama integration

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

### JSON Repair

The pipeline includes repair functions to handle common LLM output issues:

- Double-quote artifacts at value boundaries
- Unescaped quotes in scientific notation
- Trailing commas in arrays/objects
- Unicode encoding errors
- Truncated JSON completion

## License

MIT License

## Citation

Part of the [FAIR Data Innovations Hub](https://fairdataihub.org/) posters-science project.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
