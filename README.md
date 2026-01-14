# FAIR Data Innovations Hub - Machine-Actionable Poster Extraction Beta

Automated extraction of structured metadata from scientific poster PDFs and images using Large Language Models.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
  - [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
  - [Development Mode](#development-mode-hot-reload)
- [Architecture](#architecture)
  - [Pipeline Overview](#pipeline-overview)
  - [Models Used](#models-used)
  - [Stage 1: Raw Text Extraction](#stage-1-raw-text-extraction)
  - [Stage 2: JSON Structuring](#stage-2-json-structuring-transformers--llama-31-8b)
- [Evaluation](#evaluation)
  - [Metrics](#metrics)
  - [Validation Results](#validation-results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This pipeline converts scientific posters (PDF and image formats) into structured JSON following the [posters-science JSON schema](https://github.com/fairdataihub/posters-science-schema). The system achieves **90% compliance** (9/10 posters) on validation metrics with a ≥0.75 threshold across all measures.

## System Requirements

### Hardware

- CUDA-capable GPU with ≥24GB VRAM (both models loaded simultaneously)
  - Or ≥16GB VRAM if processing PDFs and images separately
- Sufficient system RAM for model loading (~32GB recommended)

### Software

- Python 3.10+
- CUDA 11.8+ with compatible drivers
- Linux, macOS, or Windows with WSL2

### External Tools

- `pdfalto` - PDF layout analysis tool (compiled binary required)
  - Installation instructions below

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fairdataihub/posters-science-posterextraction-beta.git
cd posters-science-posterextraction-beta
```

### 2. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Python Dependencies

```
transformers>=4.40.0
torch>=2.0.0
rouge-score
qwen-vl-utils
accelerate
Pillow
numpy
flask>=2.3.0
flask-cors>=4.0.0
jsonschema>=4.20.0
```

### 3. Install pdfalto (Required for PDF processing)

`pdfalto` is required for PDF text extraction with layout preservation.

#### Option A: Build with Docker (Linux/macOS/Windows)
> **Windows users**: Use Docker deployment or WSL2. Native Windows builds are not supported.

The easiest cross-platform method is to build using Docker. This produces a Linux binary suitable for use in Docker containers or WSL2.

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/kermitt2/pdfalto.git
cd pdfalto

# Create a build Dockerfile
cat > Dockerfile.build << 'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /pdfalto
COPY . .
RUN cmake . && make -j$(nproc)
EOF
```

Build and extract the binary:

```bash
# Build the builder image
docker build -f Dockerfile.build -t pdfalto-builder .

# Create a container and copy the compiled binary out
container=$(docker create pdfalto-builder)
docker cp "${container}":/pdfalto/pdfalto ./pdfalto
docker rm "${container}"

# Place in the repo executables folder
mkdir -p <this_repo>/executables
mv ./pdfalto <this_repo>/executables/pdfalto
chmod +x <this_repo>/executables/pdfalto
```

#### Option B: Build from source (Linux/macOS only)

Requires `cmake` and a C++ compiler (gcc/clang) installed on your system.

```bash
git clone --recurse-submodules https://github.com/kermitt2/pdfalto.git
cd pdfalto
cmake .
make -j$(nproc)
# Binary will be at: ./pdfalto
```

#### Configure the path (Linux/macOS only)

After building from source, configure pdfalto using one of these methods:

```bash
# Option 1: Set environment variable (recommended)
export PDFALTO_PATH="/path/to/pdfalto"

# Option 2: Add to system PATH
sudo cp /path/to/pdfalto /usr/local/bin/

# Option 3: Place in auto-discovered location
cp /path/to/pdfalto ~/Downloads/pdfalto
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

### Command Line Arguments

| Argument           | Description                             | Default  |
| ------------------ | --------------------------------------- | -------- |
| `--annotation-dir` | Directory containing poster PDFs/images | Required |
| `--output-dir`     | Directory for extracted JSON outputs    | Required |

### With Environment Variables

```bash
# Specify GPU device
CUDA_VISIBLE_DEVICES=0 python poster_extraction.py --annotation-dir ./posters

# Custom pdfalto location
PDFALTO_PATH=/opt/pdfalto/pdfalto python poster_extraction.py --annotation-dir ./posters

# Full example
HF_TOKEN="your_token" \
PDFALTO_PATH="/usr/local/bin/pdfalto" \
CUDA_VISIBLE_DEVICES=0 \
python poster_extraction.py \
    --annotation-dir "./manual_poster_annotation" \
    --output-dir "./extraction_output"
```

### API Usage

The pipeline includes a Flask API for web integration:

```bash
# Start the API server
python api.py

# Or via Docker
docker-compose up
```

#### Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/extract` | POST | Extract JSON from uploaded poster |

#### Example Request

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@poster.pdf"
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker compose build

# Run with GPU support
docker compose up

# For production deployment
docker compose -f docker-compose-prod.yml up -d
```

### Environment Variables

Create a `.env` file in the project root:

```bash
CUDA_VISIBLE_DEVICES=0
GPU_COUNT=1
RESTART_POLICY=unless-stopped
```

### Development Mode (Hot Reload)

For active development, use the dev configuration which mounts source files as volumes. This allows you to edit code locally and apply changes with a quick restart instead of a full rebuild.

```bash
# First time setup (builds image, downloads models ~16GB each)
docker compose -f docker-compose.dev.yml up --build

# After editing .py files, restart to apply changes
docker compose -f docker-compose.dev.yml restart

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Stop the container
docker compose -f docker-compose.dev.yml down
```

The dev configuration:

- Mounts `poster_extraction.py` and `api.py` as volumes for live editing
- Persists model cache in a named volume (survives container rebuilds)
- Uses a separate container name (`poster-extraction-dev`) to avoid conflicts

**Note:** If you change the model IDs (`JSON_MODEL_ID` or `VISION_MODEL_ID` in `poster_extraction.py`), you need to clear the cached models:

```bash
# Clear model cache and rebuild
docker compose -f docker-compose.dev.yml down -v
docker compose -f docker-compose.dev.yml up --build
```

The `-v` flag removes the named volumes, forcing a fresh model download.

See [DOCKER.md](DOCKER.md) for detailed deployment instructions.

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
               [PDF Files]        [Image Files]    [Transformers]
                    │                   │         Llama 3.1 8B
               [pdfalto]         [Qwen2-VL-7B]   Section-aware
               XML Layout        Vision OCR      JSON Generation
```

### Models Used

This pipeline leverages the following Large Language Models via HuggingFace transformers:

| Model                     | Provider | Parameters | Purpose                                        |
| ------------------------- | -------- | ---------- | ---------------------------------------------- |
| **Llama 3.1 8B Poster Extraction** | Meta AI / FAIR Data Hub  | 8B         | JSON structuring and text-to-schema conversion |
| **Qwen2-VL-7B-Instruct**  | Alibaba  | 7B         | Vision-language OCR for image posters          |

#### Llama 3.1 8B Poster Extraction

The core JSON structuring is performed by [Llama 3.1 8B Poster Extraction](https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction), a fine-tuned version of Meta's Llama 3.1 8B Instruct optimized for scientific poster metadata extraction. Key features:

- Strong instruction-following capabilities for structured output generation
- 128K context window supporting full poster text processing
- Efficient inference on consumer GPUs (16GB+ VRAM)
- Loaded via HuggingFace transformers for seamless integration

#### Qwen2-VL-7B-Instruct

Image-based posters (JPG/PNG) are processed using [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), a vision-language model that provides:

- Direct pixel-to-text extraction without traditional OCR preprocessing
- Multi-language support for international poster content
- Layout-aware text recognition preserving reading order

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

### Stage 2: JSON Structuring (Transformers + Llama 3.1 8B)

Raw text is converted to structured JSON using Llama 3.1 8B Poster Extraction via HuggingFace transformers:

#### Primary Prompt Strategy

- Section-aware extraction identifying: Abstract, Introduction, Methods, Results, Key Findings, Discussion, Conclusions, References, Contact
- Explicit disambiguation between semantically similar sections (e.g., "Key Findings" vs "References")
- Verbatim text preservation instructions to maintain scientific accuracy

#### Adaptive Fallback Mechanism

1. Initial generation with 18,000 output tokens
2. If truncation detected → retry with 24,000 tokens
3. If still truncating → switch to condensed prompt format (saves input tokens for output)

### JSON Repair

The pipeline includes repair functions to handle common LLM output issues:

- Unescaped quotes in scientific notation
- Trailing commas in arrays/objects
- Unicode encoding errors
- Truncated JSON completion

## Evaluation

### Metrics

The pipeline is validated against manually annotated reference JSONs using four complementary metrics:

| Metric                   | Description                                                       | Threshold | Rationale                             |
| ------------------------ | ----------------------------------------------------------------- | --------- | ------------------------------------- |
| **Word Capture (w)**     | Proportion of reference vocabulary present in extracted text      | ≥0.75     | Measures lexical completeness         |
| **ROUGE-L (r)**          | Longest common subsequence similarity with section-aware matching | ≥0.75     | Captures sequential text preservation |
| **Number Capture (n)**   | Proportion of numeric values preserved                            | ≥0.75     | Validates quantitative data integrity |
| **Field Proportion (f)** | Ratio of extracted to reference JSON structural elements          | 0.30–2.50 | Accommodates layout variability       |

### Metric Implementation Details

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

### Validation Results

**Production Release**: 9/10 (90%) passing

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
| 8228568   | 0.99 | 0.75    | 0.91    | 0.96   | pdfalto     |

**Aggregate Performance**: w=0.969, r=0.879, n=0.938, f=1.083

## Project Structure

### Directory Layout

```bash
posters-science-posterextraction-beta/
├── README.md
├── poster_extraction.py       # Main extraction pipeline
├── api.py                     # Flask API server
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container build instructions
├── docker-compose.yml         # Docker orchestration
├── DOCKER.md                  # Docker deployment guide
├── manual_poster_annotation/  # Reference posters and ground truth JSONs
│   ├── {poster_id}/
│   │   ├── {poster_id}.pdf    # Source poster
│   │   └── {poster_id}_sub-json.json  # Reference annotation
└── example_output/            # Sample extraction results
```

### Output Structure

```bash
output/
├── {poster_id}_raw.txt        # Extracted raw text from OCR
├── {poster_id}_extracted.json # Structured JSON per schema
└── results.json               # Evaluation metrics summary
```

### JSON Schema

Output JSONs conform to the [posters-science schema](https://posters.science/schemas/)

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

## License

MIT License

## Citation

Part of the [FAIR Data Innovations Hub](https://fairdataihub.org/) posters-science project.
