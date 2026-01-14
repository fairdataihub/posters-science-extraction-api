#!/usr/bin/env python3
"""
Poster Science - Scientific Poster Metadata Extraction Pipeline

Extracts structured metadata from scientific poster PDFs and images
into JSON format conforming to the posters-science schema.

Models:
- Llama 3.1 8B Poster Extraction: JSON structuring (via HuggingFace transformers)
- Qwen2-VL-7B-Instruct: Vision OCR for images

Requirements:
- pdfalto: PDF layout analysis tool (https://github.com/kermitt2/pdfalto)
- CUDA-capable GPU with ≥16GB VRAM (models loaded one at a time)

Environment Variables:
- PDFALTO_PATH: Path to pdfalto binary (required for PDF processing)
- CUDA_VISIBLE_DEVICES: GPU device(s) to use (default: 0)
- HF_TOKEN: HuggingFace token for gated models
"""

import os
import shutil
import torch
import fitz  # PyMuPDF
import json
import re
import time
import argparse
import subprocess
import tempfile
import gc
import unicodedata
import numpy as np
from pathlib import Path
from datetime import datetime

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from rouge_score import rouge_scorer


# GPU visibility can be controlled via CUDA_VISIBLE_DEVICES environment variable
# If not set, all available GPUs will be used with device_map="auto"

# Model configuration - HuggingFace transformers
JSON_MODEL_ID = "jimnoneill/Llama-3.1-8B-Poster-Extraction"
VISION_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Find pdfalto: check environment variable, then known paths, then PATH
PDFALTO_PATH = os.environ.get("PDFALTO_PATH")
if not PDFALTO_PATH:
    # Check known project locations first
    known_paths = [
        "/home/joneill/vaults/jmind/calmi2/poster_science/pdfalto/pdfalto",
        Path.home() / "Downloads" / "pdfalto",
        "/usr/local/bin/pdfalto",
        "/usr/bin/pdfalto",
    ]
    for p in known_paths:
        if Path(p).exists():
            PDFALTO_PATH = str(p)
            break
    # Finally check PATH
    if not PDFALTO_PATH:
        pdfalto_in_path = shutil.which("pdfalto")
        if pdfalto_in_path:
            PDFALTO_PATH = pdfalto_in_path

MAX_JSON_TOKENS = 18000
MAX_RETRY_TOKENS = 24000


def log(msg: str):
    """
    Simple timestamped logger used across CLI and API entrypoints.

    Using print() keeps behaviour the same whether the module is run
    from the command line or imported from the Flask API.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def free_gpu():
    """Best-effort GPU memory cleanup helper."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_best_gpu(min_memory_gb: int = 16):
    """
    Get the GPU with most available memory for model loading.
    Returns device string like 'cuda:0' or 'cpu' if no GPU available.
    Works automatically across any multi-GPU configuration.
    Uses torch.cuda.mem_get_info() to get actual free memory (not just this process).
    
    Args:
        min_memory_gb: Minimum free memory required in GB (default: 16 for 8B models)
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"
    
    # Find GPU with most actual free memory (accounts for all processes)
    best_gpu = 0
    max_free = 0
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        log(f"   GPU {i}: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        if free_mem > max_free:
            max_free = free_mem
            best_gpu = i
    
    max_free_gb = max_free / (1024**3)
    if max_free_gb < min_memory_gb:
        log(f"   ⚠ WARNING: Best GPU has only {max_free_gb:.1f}GB free, model needs ~{min_memory_gb}GB")
        log(f"   ⚠ Other processes may be using GPU memory. Consider waiting or killing them.")
    
    log(f"   Selected GPU {best_gpu} with {max_free_gb:.1f}GB free")
    return f"cuda:{best_gpu}"


# ============================
# QWEN2-VL OCR FOR IMAGES
# ============================

_vision_model = None
_vision_processor = None


def load_vision_model():
    global _vision_model, _vision_processor
    if _vision_model is None:
        device = get_best_gpu()
        log(f"Loading {VISION_MODEL_ID} for image OCR on {device}...")
        _vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VISION_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        _vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        log(f"   ✓ Vision model loaded on {device}")
    return _vision_model, _vision_processor


def unload_vision_model():
    global _vision_model, _vision_processor
    if _vision_model is not None:
        del _vision_model
        _vision_model = None
    if _vision_processor is not None:
        del _vision_processor
        _vision_processor = None
    free_gpu()
    log("   ✓ Vision model unloaded, GPU memory cleared")


def extract_text_with_qwen_vision(image_path: str) -> str:
    """
    Use Qwen2-VL-7B for high-quality image OCR.

    This is called for image posters (JPG/PNG). It loads the vision
    model on first use, resizes the image if needed, and sends a single
    instruction to transcribe all visible text.
    """
    log(f"Starting vision OCR on image: {image_path}")
    model, processor = load_vision_model()

    image = Image.open(image_path).convert("RGB")
    max_size = 1280
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize(
            (int(image.size[0] * ratio), int(image.size[1] * ratio)),
            Image.Resampling.LANCZOS,
        )

    prompt = """Transcribe ALL visible text from this scientific poster exactly as written.

Include:
- Title and subtitle
- Author names and affiliations
- All section headers and content
- Algorithm/method descriptions
- Figure and table captions
- Numbers, statistics, equations
- References and URLs

Rules:
- Output the raw text ONLY
- Do NOT add explanations or interpretations
- Do NOT translate any text
- Preserve the original language
- Include all bullet points and lists"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    ).to(model.device)

    # Time the vision generation step so we can see how long OCR takes
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=4000, do_sample=False)
    vision_elapsed = time.time() - t0
    log(f"   Vision OCR generate() finished in {vision_elapsed:.2f} seconds")

    response = processor.batch_decode(output, skip_special_tokens=True)[0]

    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    log(f"   Completed vision OCR for: {image_path}")
    return response


# ============================
# PDF TEXT EXTRACTION
# ============================


def extract_text_with_pdfalto(pdf_path: str) -> str:
    """
    Extract text from a PDF using pdfalto (preferred for layout-aware text).

    Returns:
        Extracted text as a single string, or None on error.
    """
    log(f"Attempting text extraction with pdfalto for: {pdf_path}")
    if PDFALTO_PATH is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "output.xml")
            t0 = time.time()
            result = subprocess.run(
                [PDFALTO_PATH, "-noImage", "-readingOrder", pdf_path, xml_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            elapsed = time.time() - t0
            if result.returncode != 0:
                raise RuntimeError(
                    f"pdfalto returned code {result.returncode}: {result.stderr}"
                )
            if not os.path.exists(xml_path):
                raise RuntimeError("pdfalto did not produce output XML")
            text = parse_alto_xml(xml_path)
            if text is None:
                log(f"pdfalto XML parsing failed for: {pdf_path}")
            else:
                log(
                    f"pdfalto extracted {len(text)} characters from {pdf_path} "
                    f"in {elapsed:.2f} seconds"
                )
            return text
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pdfalto timeout processing {pdf_path}")
    except Exception as e:
        raise RuntimeError(f"pdfalto error: {e}")


def parse_alto_xml(xml_path: str) -> str:
    from xml.etree import ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        text_blocks = root.findall(
            ".//{http://www.loc.gov/standards/alto/ns-v3#}TextBlock"
        )
        if not text_blocks:
            text_blocks = root.findall(".//TextBlock")

        lines = []
        for block in text_blocks:
            strings = block.findall(
                ".//{http://www.loc.gov/standards/alto/ns-v3#}String"
            )
            if not strings:
                strings = block.findall(".//String")

            words = [s.get("CONTENT", "") for s in strings if s.get("CONTENT")]
            if words:
                lines.append(" ".join(words))

        return "\n".join(lines)
    except Exception:
        return None


def extract_text_with_pymupdf(pdf_path: str) -> str:
    """
    Fallback text extraction using PyMuPDF when pdfalto is unavailable
    or fails to return enough content.
    """
    log(f"Attempting text extraction with PyMuPDF for: {pdf_path}")
    t0 = time.time()
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    elapsed = time.time() - t0
    log(
        f"PyMuPDF extracted {len(text)} characters from {pdf_path} "
        f"in {elapsed:.2f} seconds"
    )
    return text.strip()


def get_raw_text(
    poster_path: str, poster_id: str = None, output_dir: str = None
) -> tuple:
    """
    Get raw text from a poster file.

    This handles both images (via Qwen2-VL) and PDFs (via pdfalto with
    PyMuPDF fallback). When an output directory and poster_id are
    provided, image OCR results may be cached as *_raw.txt files.

    Returns:
        (text, source) where source indicates which extractor was used.
    """
    log(f"Starting raw text extraction for: {poster_path}")
    ext = Path(poster_path).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        # Check cache if output_dir is provided (check both .md and legacy .txt)
        if output_dir and poster_id:
            cache_file = Path(output_dir) / f"{poster_id}_raw.md"
            # Also check legacy .txt format for backwards compatibility
            if not cache_file.exists():
                cache_file = Path(output_dir) / f"{poster_id}_raw.txt"
            if cache_file.exists():
                with open(cache_file) as f:
                    text = f.read()
                if len(text) > 500:
                    log(
                        f"Using cached OCR text for image poster {poster_id} "
                        f"({len(text)} characters)"
                    )
                    return text, "qwen_vision_cached"

        text = extract_text_with_qwen_vision(poster_path)
        log(f"Image OCR produced {len(text)} characters for {poster_path}")
        return text, "qwen_vision"

    if ext == ".pdf":
        text = extract_text_with_pdfalto(poster_path)
        if text and len(text) > 500:
            log(f"Using pdfalto output for {poster_path} " f"({len(text)} characters)")
            return text, "pdfalto"
        # Either pdfalto failed or produced too little text; fall back
        text = extract_text_with_pymupdf(poster_path)
        log(f"Using PyMuPDF fallback for {poster_path} " f"({len(text)} characters)")
        return text, "pymupdf"

    return "", "unknown"


# ============================
# JSON MODEL (TRANSFORMERS)
# ============================

_json_model = None
_json_tokenizer = None


def load_json_model():
    """
    Load the Llama 3.1 8B model for JSON structuring via HuggingFace transformers.

    Automatically uses 8-bit quantization if GPU memory is limited (<16GB free).
    
    Returns:
        (model, tokenizer) tuple for use with generate().
    """
    global _json_model, _json_tokenizer
    if _json_model is None:
        device = get_best_gpu()
        
        # Check available memory
        if device != "cpu":
            gpu_id = int(device.split(":")[1])
            free_mem, _ = torch.cuda.mem_get_info(gpu_id)
            free_gb = free_mem / (1024**3)
        else:
            free_gb = 32  # Assume enough RAM
        
        log(f"Loading {JSON_MODEL_ID} for JSON structuring on {device}...")
        _json_tokenizer = AutoTokenizer.from_pretrained(JSON_MODEL_ID)
        
        if free_gb < 16 and device != "cpu":
            # Use 8-bit quantization for limited memory
            log(f"   Using 8-bit quantization (only {free_gb:.1f}GB free)")
            _json_model = AutoModelForCausalLM.from_pretrained(
                JSON_MODEL_ID,
                load_in_8bit=True,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        else:
            _json_model = AutoModelForCausalLM.from_pretrained(
                JSON_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map=device,
                low_cpu_mem_usage=True,
            )
        log(f"   ✓ JSON model loaded on {device}")
    return _json_model, _json_tokenizer


def unload_json_model():
    """Unload the JSON model to free GPU memory."""
    global _json_model, _json_tokenizer
    if _json_model is not None:
        del _json_model
        _json_model = None
    if _json_tokenizer is not None:
        del _json_tokenizer
        _json_tokenizer = None
    free_gpu()
    log("   ✓ JSON model unloaded, GPU memory cleared")


def generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """
    Generate a response using the Llama 3.1 model via transformers.

    Uses the model's chat template for proper formatting.
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    log(
        f"Calling model.generate() with max_new_tokens={max_tokens} "
        f"and input length={inputs['input_ids'].shape[1]} tokens"
    )
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    log(f"   model.generate() completed in {elapsed:.2f} seconds")

    # Decode only the new tokens (skip the input prompt)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


# PRIMARY PROMPT (best quality - 90% pass rate)
EXTRACTION_PROMPT = """Convert this scientific poster text to JSON format.

CRITICAL RULES FOR SECTIONS:
1. Create a SEPARATE section for EACH distinct topic/header found in the poster
2. Common section headers include:
   - Abstract, Introduction, Background
   - Methods, Methodology, Materials and Methods
   - Results, Key Findings, Findings (SEPARATE sections if both exist!)
   - Discussion, Conclusions, Discussion/Conclusions
   - References (MUST contain numbered citations like "1. Author..." NOT findings!)
   - Acknowledgements, Contact, Funding
3. Each section must have its OWN "sectionTitle" and "sectionContent"
4. Do NOT merge multiple topics into one section
5. Copy ALL text EXACTLY - do not paraphrase or summarize
6. IMPORTANT: "Key Findings" and "References" are DIFFERENT sections:
   - Key Findings = bullet points about discoveries/results
   - References = numbered bibliography citations with author names and years

JSON SCHEMA:
{{
  "creators": [
    {{"name": "LastName, FirstName", "affiliation": [{{"name": "Institution Name"}}]}}
  ],
  "titles": [{{"title": "Main Poster Title"}}],
  "posterContent": {{
    "sections": [
      {{"sectionTitle": "First Section Header", "sectionContent": "Complete verbatim text of first section"}},
      {{"sectionTitle": "Second Section Header", "sectionContent": "Complete verbatim text of second section"}},
      {{"sectionTitle": "Third Section Header", "sectionContent": "Complete verbatim text..."}},
      ...continue for ALL sections found in the poster...
    ]
  }},
  "imageCaption": [{{"caption1": "Figure 1 caption text"}}],
  "tableCaption": [{{"caption1": "Table 1 caption text"}}]
}}

EXAMPLE - A poster with 8 sections (Abstract, Intro, Methods, Results, Key Findings, Discussion, References, Contact) should produce 8 section objects, not fewer.

POSTER TEXT TO CONVERT:
{raw_text}

OUTPUT VALID JSON ONLY:"""

# FALLBACK PROMPT (shorter - for truncation cases)
FALLBACK_PROMPT = """Convert poster text to JSON. RULES:
1. SEPARATE section for EACH header (Abstract, Intro, Methods, Results, Key Findings, Discussion, Conclusions, References, Contact)
2. Key Findings ≠ References. References = numbered citations with authors/years
3. Copy ALL text EXACTLY verbatim

{{
  "creators": [{{"name": "LastName, FirstName", "affiliation": [{{"name": "Institution"}}]}}],
  "titles": [{{"title": "Poster Title"}}],
  "posterContent": {{
    "sections": [{{"sectionTitle": "Header", "sectionContent": "verbatim text"}}]
  }},
  "imageCaption": [{{"caption1": "Figure caption"}}],
  "tableCaption": [{{"caption1": "Table caption"}}]
}}

TEXT:
{raw_text}

JSON:"""


def is_truncated(json_str: str) -> bool:
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")

    if open_braces > 0 or open_brackets > 0:
        return True

    if json_str.rstrip().endswith((",", ":", '"')):
        return True

    return False


def extract_json_with_retry(raw_text: str, model, tokenizer) -> dict:
    """
    Send raw poster text to the LLM and robustly parse the JSON response.

    This function:
      1. Calls the model with a full prompt
      2. Retries with more tokens if truncation is detected
      3. Falls back to a shorter prompt if needed
      4. Runs several repair passes to make the JSON parseable
    """
    # Try primary prompt first
    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)

    log("Starting primary JSON extraction with full prompt")
    response = generate(model, tokenizer, prompt, MAX_JSON_TOKENS)
    result = robust_json_parse(response)
    if "error" in result:
        log(f"Primary JSON parse reported error: {result['error']}")
    else:
        log("Primary JSON parse succeeded")

    # If truncation/error, retry with more tokens
    if "error" in result or is_truncated(result.get("raw", "")):
        log(
            f"Truncation or error detected, retrying JSON extraction "
            f"with max_tokens={MAX_RETRY_TOKENS}"
        )
        response = generate(model, tokenizer, prompt, MAX_RETRY_TOKENS)
        result = robust_json_parse(response)
        if "error" in result:
            log(f"Retry JSON parse reported error: {result['error']}")
        else:
            log("Retry JSON parse succeeded")

    # If still failing, try FALLBACK shorter prompt (saves input tokens for output)
    if "error" in result or is_truncated(result.get("raw", "")):
        log("Still seeing truncation or errors, using FALLBACK shorter prompt")
        fallback_prompt = FALLBACK_PROMPT.format(raw_text=raw_text)
        response = generate(model, tokenizer, fallback_prompt, MAX_RETRY_TOKENS)
        result = robust_json_parse(response)
        if "error" in result:
            log(f"Fallback JSON parse reported error: {result['error']}")
        else:
            log("Fallback JSON parse succeeded")

    # Comprehensive post-processing for schema compliance
    result = postprocess_json(result)

    return result


def remove_empty_sections(generated: dict) -> dict:
    """
    Remove sections with empty or whitespace-only content.
    
    This prevents hallucinated sections where the model creates a section
    header (e.g., "Discussion") but has no content to fill it.
    """
    def has_content(section):
        """Check if a section has non-empty content."""
        if not isinstance(section, dict):
            return False
        content = section.get("sectionContent", "")
        # Handle both string and list content
        if isinstance(content, str):
            return bool(content.strip())
        elif isinstance(content, list):
            return len(content) > 0
        return bool(content)
    
    if "posterContent" in generated and isinstance(generated["posterContent"], dict):
        sections = generated["posterContent"].get("sections", [])
        if isinstance(sections, list):
            original_count = len(sections)
            filtered = [s for s in sections if has_content(s)]
            if len(filtered) < original_count:
                log(f"   Removed {original_count - len(filtered)} empty section(s)")
            generated["posterContent"]["sections"] = filtered
    return generated


# ============================
# POST-PROCESSING
# ============================

SCHEMA_URL = "https://posters.science/schema/v0.1/poster_schema.json"


def is_empty_or_self_referential(section: dict) -> bool:
    """Check if a section has no meaningful content."""
    if not isinstance(section, dict):
        return True
    
    title = section.get("sectionTitle", "").strip()
    content = section.get("sectionContent", "")
    
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    content = content.strip() if isinstance(content, str) else ""
    
    if not content:
        return True
    if content == title:
        return True
    if len(content) < 5 and not any(c.isalpha() for c in content):
        return True
    
    return False


def is_figure_section(section: dict) -> bool:
    """Check if this section is actually a figure caption misclassified as section."""
    title = section.get("sectionTitle", "").strip()
    content = section.get("sectionContent", "")
    
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    content = content.strip() if isinstance(content, str) else ""
    
    if re.match(r'^Figure\s+\d+', title, re.IGNORECASE):
        return True
    if re.match(r'^Figure\s+\d+\.', content, re.IGNORECASE):
        return True
    
    return False


def is_table_section(section: dict) -> bool:
    """Check if this section is actually a table caption misclassified as section."""
    title = section.get("sectionTitle", "").strip()
    content = section.get("sectionContent", "")
    
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    content = content.strip() if isinstance(content, str) else ""
    
    if re.match(r'^Table\s+\d+', title, re.IGNORECASE):
        return True
    if re.match(r'^Table\s+\d+\.', content, re.IGNORECASE):
        return True
    
    return False


def detect_table_data_in_content(content: str) -> bool:
    """Detect if section content contains raw table data."""
    table_indicators = [
        r'Reason\s+Participant\s*\(n=\d+\)',
        r'(?:Column\s+\d+|Row\s+\d+)',
        r'\|\s*\w+\s*\|',
        r'(?:\w+\s+){2,}\n(?:\w+\s+){2,}',
    ]
    for pattern in table_indicators:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def clean_table_data_from_content(content: str, title: str) -> str:
    """Remove raw table data from section content, keeping only summary text."""
    if "Results" in title or "Findings" in title:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20 and not re.match(r'^(Reason|Participant|Example)', first_sentence):
                return first_sentence
    return content


def normalize_captions(captions: list) -> list:
    """Normalize caption keys and deduplicate (including substring matches)."""
    normalized = []
    
    for caption in captions:
        if not isinstance(caption, dict):
            continue
        
        caption_text = ""
        for key in sorted(caption.keys()):
            val = caption.get(key, "")
            if val and isinstance(val, str) and val.strip():
                caption_text = val.strip()
                break
        
        if not caption_text:
            continue
        
        is_duplicate = False
        for existing in normalized:
            existing_text = existing.get("caption1", "")
            if caption_text == existing_text:
                is_duplicate = True
                break
            if caption_text in existing_text:
                is_duplicate = True
                break
            if existing_text in caption_text:
                existing["caption1"] = caption_text
                is_duplicate = True
                break
        
        if not is_duplicate:
            normalized.append({"caption1": caption_text, "caption2": ""})
    
    return normalized


def remove_duplicate_sections(sections: list) -> list:
    """Remove sections with duplicate content."""
    seen_content = {}
    unique = []
    
    for section in sections:
        if not isinstance(section, dict):
            continue
        
        content = section.get("sectionContent", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        
        content_key = content.strip().lower()[:200]
        
        if content_key and content_key in seen_content:
            log(f"   Removing duplicate section: '{section.get('sectionTitle')}' (same as '{seen_content[content_key]}')")
            continue
        
        if content_key:
            seen_content[content_key] = section.get("sectionTitle", "")
        unique.append(section)
    
    return unique


def remove_caption_text_from_sections(sections: list, captions: list, caption_type: str) -> list:
    """Remove figure/table caption text that's embedded in section content."""
    if not captions:
        return sections
    
    caption_texts = []
    for cap in captions:
        if isinstance(cap, dict):
            text = cap.get("caption1", "") or cap.get("caption2", "")
            if text:
                caption_texts.append(text.strip())
    
    if not caption_texts:
        return sections
    
    updated_sections = []
    for section in sections:
        if not isinstance(section, dict):
            updated_sections.append(section)
            continue
        
        title = section.get("sectionTitle", "")
        content = section.get("sectionContent", "")
        
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        
        if not content:
            updated_sections.append(section)
            continue
        
        original_content = content
        content_modified = False
        
        for caption_text in caption_texts:
            match_key = caption_text[:100] if len(caption_text) > 100 else caption_text
            
            if match_key in content:
                if caption_text in content:
                    content = content.replace(caption_text, "").strip()
                    content_modified = True
                elif match_key in content:
                    idx = content.find(match_key)
                    end_idx = len(content)
                    for end_char in ['\n\n', '\n', '. ', '.']:
                        pos = content.find(end_char, idx + len(match_key))
                        if pos != -1 and pos < end_idx:
                            end_idx = pos + len(end_char)
                            break
                    content = (content[:idx] + content[end_idx:]).strip()
                    content_modified = True
        
        if content_modified:
            content = re.sub(r'^\s*[A-Z]\s+[A-Z](?:\s+[A-Z])*\s*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            content = content.strip()
            
            if len(content) < len(original_content):
                log(f"   Removed {caption_type} caption text from section: '{title}' ({len(original_content)} -> {len(content)} chars)")
        
        if content:
            updated_sections.append({"sectionTitle": title, "sectionContent": content})
        else:
            log(f"   Section '{title}' became empty after caption removal, removing section")
    
    return updated_sections


def postprocess_json(data: dict) -> dict:
    """
    Comprehensive post-processing for extracted JSON.
    
    Cleans up extraction artifacts:
    1. Removes empty/self-referential sections
    2. Moves figure/table sections to caption arrays
    3. Removes duplicate sections
    4. Cleans table data from Results sections
    5. Deduplicates captions
    6. Removes caption text embedded in sections
    7. Adds schema URL
    """
    result = data.copy()
    
    # Add schema if missing
    if "$schema" not in result:
        result["$schema"] = SCHEMA_URL
    
    # Ensure required fields exist
    if "imageCaption" not in result:
        result["imageCaption"] = []
    if "tableCaption" not in result:
        result["tableCaption"] = []
    
    # Process sections
    if "posterContent" in result and isinstance(result["posterContent"], dict):
        sections = result["posterContent"].get("sections", [])
        
        if isinstance(sections, list):
            cleaned_sections = []
            extracted_figures = []
            extracted_tables = []
            
            for section in sections:
                if not isinstance(section, dict):
                    continue
                
                title = section.get("sectionTitle", "").strip()
                content = section.get("sectionContent", "")
                
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                content = content.strip() if isinstance(content, str) else ""
                
                # Skip empty/self-referential sections
                if is_empty_or_self_referential(section):
                    log(f"   Removing empty/self-referential section: '{title}'")
                    continue
                
                # Extract figure sections to imageCaption
                if is_figure_section(section):
                    caption_text = content if content.startswith("Figure") else f"{title}. {content}"
                    extracted_figures.append({"caption1": caption_text, "caption2": ""})
                    log(f"   Moving figure section to imageCaption: '{title}'")
                    continue
                
                # Extract table sections to tableCaption  
                if is_table_section(section):
                    caption_text = content if content.startswith("Table") else f"{title}. {content}"
                    extracted_tables.append({"caption1": caption_text, "caption2": ""})
                    log(f"   Moving table section to tableCaption: '{title}'")
                    continue
                
                # Clean table data from Results/Findings sections
                if detect_table_data_in_content(content):
                    original_len = len(content)
                    content = clean_table_data_from_content(content, title)
                    if len(content) < original_len:
                        log(f"   Cleaned table data from section: '{title}' ({original_len} -> {len(content)} chars)")
                        section = {"sectionTitle": title, "sectionContent": content}
                
                cleaned_sections.append(section)
            
            # Remove duplicates
            cleaned_sections = remove_duplicate_sections(cleaned_sections)
            
            # Merge extracted captions with existing
            existing_figures = result.get("imageCaption", [])
            existing_tables = result.get("tableCaption", [])
            
            all_figures = existing_figures + extracted_figures
            all_tables = existing_tables + extracted_tables
            
            # Normalize and deduplicate captions
            result["imageCaption"] = normalize_captions(all_figures)
            result["tableCaption"] = normalize_captions(all_tables)
            
            # Remove caption text from section content where it duplicates the caption arrays
            cleaned_sections = remove_caption_text_from_sections(
                cleaned_sections, result["imageCaption"], "Figure"
            )
            cleaned_sections = remove_caption_text_from_sections(
                cleaned_sections, result["tableCaption"], "Table"
            )
            
            result["posterContent"]["sections"] = cleaned_sections
    
    return result


# ============================
# JSON PARSING
# ============================


def robust_json_parse(response: str) -> dict:
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        start_marker = response.find("```json")
        end_marker = response.find("```", start_marker + 7)
        if end_marker > start_marker:
            response = response[start_marker + 7 : end_marker]
    elif "```" in response:
        # Find content between first ``` and next ```
        start_marker = response.find("```")
        end_marker = response.find("```", start_marker + 3)
        if end_marker > start_marker:
            response = response[start_marker + 3 : end_marker]

    response = response.strip()

    start = response.find("{")
    if start == -1:
        return {"error": "No JSON found", "raw": response[:3000]}

    json_str = response[start:]

    # First, fix known problematic patterns (do this BEFORE extracting object)
    json_str = repair_unescaped_quotes(json_str)

    # Try to extract complete JSON object
    extracted = extract_first_json_object(json_str)
    if extracted:
        json_str = extracted

    # Try direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Apply repairs in order of likelihood to fix
    repair_funcs = [
        repair_unescaped_quotes,  # Fix quote escaping first
        repair_trailing_commas,
        repair_unicode,
        repair_truncation,
        repair_all,
    ]

    for repair_func in repair_funcs:
        try:
            repaired = repair_func(json_str)
            return json.loads(repaired)
        except Exception:
            continue

    # Last resort: aggressive repair
    try:
        repaired = repair_all(repair_unescaped_quotes(json_str))
        return json.loads(repaired)
    except Exception:
        pass

    return {"error": "JSON parse failed", "raw": json_str[:3000]}


def extract_first_json_object(s: str) -> str:
    if not s or s[0] != "{":
        return ""

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]

    return s


def repair_unescaped_quotes(s: str) -> str:
    """Fix quotes that appear after / which are not properly escaped."""
    # Pattern: something/" where " is likely part of content (e.g., units like pc/")
    # These should be escaped as /" -> /\"
    s = re.sub(
        r'(\d+\s*(?:pc|km|m|cm|mm|Hz|kHz|MHz|GHz|s|ms|ns|arcsec|arcmin|deg))/"',
        r'\1/\\"',
        s,
    )
    # Also handle parenthetical unit patterns like (53 pc/")
    s = re.sub(r'\((\d+\.?\d*\s*\w+)/"\)', r'(\1/\\")', s)
    return s


def repair_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def repair_unicode(s: str) -> str:
    s = re.sub(r"\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])", "", s)
    s = re.sub(r"[\x00-\x1f]", " ", s)
    return s


def repair_truncation(s: str) -> str:
    s = repair_trailing_commas(s)

    in_string = False
    escape = False
    open_braces = 0
    open_brackets = 0

    for c in s:
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            open_braces += 1
        elif c == "}":
            open_braces -= 1
        elif c == "[":
            open_brackets += 1
        elif c == "]":
            open_brackets -= 1

    if in_string:
        s = s.rstrip()
        if not s.endswith('"'):
            s += '"'
        # If we were in a string value (like sectionContent), need to close the object too
        # Check if we were inside a section by looking for recent sectionContent key
        if '"sectionContent":' in s[-1000:] or "sectionContent" in s[-500:]:
            open_braces += 1  # Account for the section object

    s = s.rstrip()
    while s and s[-1] not in '{}[]"0123456789truefalsenull':
        s = s[:-1].rstrip()
    if s.endswith(","):
        s = s[:-1]

    s += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    return s


def repair_all(s: str) -> str:
    s = repair_unescaped_quotes(s)  # Fix quote escaping first
    s = repair_unicode(s)
    s = repair_trailing_commas(s)
    s = repair_truncation(s)
    return s


# ============================
# METRICS
# ============================


def get_all_text(d) -> str:
    if isinstance(d, dict):
        return " ".join(get_all_text(v) for v in d.values())
    elif isinstance(d, list):
        return " ".join(get_all_text(item) for item in d)
    elif isinstance(d, str):
        return d
    return ""


def get_section_texts(d) -> list:
    sections = []
    if isinstance(d, dict):
        if "posterContent" in d and isinstance(d["posterContent"], dict):
            for section in d["posterContent"].get("sections", []):
                if isinstance(section, dict):
                    content = section.get("sectionContent", "")
                    if content:
                        sections.append(content)
        for key in ["imageCaption", "tableCaption"]:
            if key in d:
                for caption in d[key]:
                    if isinstance(caption, dict):
                        for v in caption.values():
                            if v:
                                sections.append(str(v))
    return sections


def normalize_text(text) -> str:
    # Handle non-string inputs
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)

    space_chars = [
        "\xa0",
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
    ]
    for space in space_chars:
        text = text.replace(space, " ")

    single_quotes = [""", """, "‛", "′", "‹", "›", "‚", "‟"]
    for quote in single_quotes:
        text = text.replace(quote, "'")

    double_quotes = ['"', '"', "„", "‟", "«", "»", "〝", "〞", "〟", "＂"]
    for quote in double_quotes:
        text = text.replace(quote, '"')

    hyphens_and_dashes = ["‐", "‑", "‒", "–", "—", "―", "−"]
    for dash in hyphens_and_dashes:
        text = text.replace(dash, "-")

    superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    subscripts = "₀₁₂₃₄₅₆₇₈₉"
    normal_numbers = "0123456789"
    for super_, sub_, normal in zip(superscripts, subscripts, normal_numbers):
        text = text.replace(super_, normal).replace(sub_, normal)

    text = unicodedata.normalize("NFKD", text)

    return text


def strip_to_alphanumeric(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", text)).strip().lower()


def calculate_forgiving_rouge(
    gen_text: str, ref_text: str, gen_sections: list, ref_sections: list
) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    global_score = scorer.score(ref_text, gen_text)["rougeL"].fmeasure

    if ref_sections and gen_sections:
        section_scores = []
        for ref_sec in ref_sections:
            ref_sec_norm = normalize_text(ref_sec)
            best_match = 0.0
            for gen_sec in gen_sections:
                gen_sec_norm = normalize_text(gen_sec)
                score = scorer.score(ref_sec_norm, gen_sec_norm)["rougeL"].fmeasure
                best_match = max(best_match, score)
            section_scores.append(best_match)

        section_avg = np.mean(section_scores) if section_scores else 0.0
    else:
        section_avg = 0.0

    return max(global_score, section_avg)


def calculate_forgiving_number_capture(gen_text: str, ref_text: str) -> float:
    gen_nums = set(re.findall(r"\d+\.?\d*", gen_text))
    ref_nums = set(re.findall(r"\d+\.?\d*", ref_text))

    def is_likely_content_number(n):
        try:
            val = float(n)
            if "." in n:
                return True
            if val < 1000:
                return True
            if 2010 <= val <= 2030:
                return False
            return val < 100000
        except Exception:
            return True

    ref_content_nums = {n for n in ref_nums if is_likely_content_number(n)}
    gen_content_nums = {n for n in gen_nums if is_likely_content_number(n)}

    if not ref_content_nums:
        return 1.0

    return len(gen_content_nums & ref_content_nums) / len(ref_content_nums)


def calculate_metrics(generated: dict, reference: dict) -> dict:
    gen_text = normalize_text(get_all_text(generated))
    ref_text = normalize_text(get_all_text(reference))

    gen_sections = get_section_texts(generated)
    ref_sections = get_section_texts(reference)

    gen_alpha = strip_to_alphanumeric(gen_text)
    ref_alpha = strip_to_alphanumeric(ref_text)

    gen_words = set(gen_alpha.split())
    ref_words = set(ref_alpha.split())
    word_capture = len(gen_words & ref_words) / max(len(ref_words), 1)

    rouge_l = calculate_forgiving_rouge(gen_text, ref_text, gen_sections, ref_sections)
    number_capture = calculate_forgiving_number_capture(gen_text, ref_text)

    def count_fields(d, depth=0):
        if depth > 10:
            return 0
        count = 0
        if isinstance(d, dict):
            for v in d.values():
                if v and v != [] and v != {}:
                    count += 1 + count_fields(v, depth + 1)
        elif isinstance(d, list):
            for item in d:
                count += count_fields(item, depth + 1)
        return count

    gen_fields = count_fields(generated)
    ref_fields = count_fields(reference)
    field_proportion = gen_fields / max(ref_fields, 1)

    return {
        "word_capture": float(word_capture),
        "rouge_l": float(rouge_l),
        "number_capture": float(number_capture),
        "field_proportion": float(field_proportion),
        "gen_sections": len(gen_sections),
        "ref_sections": len(ref_sections),
    }


def passes(m: dict) -> bool:
    return (
        m["word_capture"] >= 0.75
        and m["rouge_l"] >= 0.75
        and m["number_capture"] >= 0.75
        and 0.3 <= m["field_proportion"] <= 2.5
    )


def find_pairs(annotation_dir: str):
    pairs = []
    for subdir in Path(annotation_dir).iterdir():
        if not subdir.is_dir():
            continue
        json_files = list(subdir.glob("*_sub-json.json"))
        poster_files = (
            list(subdir.glob("*.pdf"))
            + list(subdir.glob("*.jpg"))
            + list(subdir.glob("*.png"))
        )
        if json_files and poster_files:
            pairs.append((str(poster_files[0]), str(json_files[0]), subdir.name))
    return sorted(pairs, key=lambda x: x[2])


def process_poster_file(poster_path: str) -> dict:
    """
    Process a single poster file and return the extracted JSON.

    Args:
        poster_path: Path to the poster file (PDF, JPG, PNG)

    Returns:
        Dictionary containing the extracted JSON structure
    """
    log(f"Processing single poster file: {poster_path}")

    # Extract raw text
    t_extract_start = time.time()
    raw_text, source = get_raw_text(poster_path)
    t_extract_elapsed = time.time() - t_extract_start

    if not raw_text or source == "unknown":
        return {
            "error": "Failed to extract text from file. Unsupported format or extraction failed."
        }

    log(
        f"Extracted {len(raw_text)} characters from {poster_path} "
        f"using source={source} in {t_extract_elapsed:.2f} seconds"
    )

    # IMPORTANT: Unload vision model BEFORE loading JSON model to free GPU memory
    # This ensures only one large model is loaded at a time
    ext = Path(poster_path).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        unload_vision_model()

    # Load JSON model (vision model is now unloaded)
    model, tokenizer = load_json_model()

    # Convert to JSON
    try:
        t_json_start = time.time()
        generated = extract_json_with_retry(raw_text, model, tokenizer)
        t_json_elapsed = time.time() - t_json_start
        if "error" in generated:
            log(
                f"JSON extraction finished with error after "
                f"{t_json_elapsed:.2f} seconds"
            )
        else:
            log(
                f"JSON extraction finished successfully in "
                f"{t_json_elapsed:.2f} seconds"
            )

        # Unload JSON model to free GPU memory for next request
        unload_json_model()

        log("Finished processing poster, returning JSON result")
        return generated
    except Exception as e:
        log(f"ERROR processing poster: {e}")
        import traceback

        traceback.print_exc()
        # Still unload models on error to free GPU memory
        unload_json_model()
        return {"error": str(e)}


def run(annotation_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(annotation_dir)

    # High-level summary before starting the full evaluation run
    log("=" * 60)
    log("POSTER EXTRACTION PIPELINE (Transformers)")
    log("=" * 60)
    log(f"JSON Model: {JSON_MODEL_ID}")
    log(f"Vision Model: {VISION_MODEL_ID}")
    log(f"Total posters to process: {len(pairs)}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")

    image_posters = [
        p for p in pairs if Path(p[0]).suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    pdf_posters = [p for p in pairs if Path(p[0]).suffix.lower() == ".pdf"]

    log(f"Image posters (Qwen Vision): {[p[2] for p in image_posters]}")
    log(f"PDF posters (pdfalto): {len(pdf_posters)}")

    # Phase 1: Extract raw text
    log("\n" + "=" * 40)
    log("PHASE 1: Raw Text Extraction")
    log("=" * 40)

    raw_texts = {}

    if image_posters:
        for poster_path, ref_json_path, poster_id in image_posters:
            log(f"[Vision OCR] {poster_id}")
            t0 = time.time()
            text, source = get_raw_text(poster_path, poster_id, output_dir)
            raw_texts[poster_id] = (text, source)
            log(f"   {len(text)} chars ({source}) [{time.time() - t0:.1f}s]")

            with open(f"{output_dir}/{poster_id}_raw.md", "w") as f:
                f.write(text)

        unload_vision_model()

    for poster_path, ref_json_path, poster_id in pdf_posters:
        text, source = get_raw_text(poster_path, poster_id, output_dir)
        raw_texts[poster_id] = (text, source)
        log(f"   {poster_id}: {len(text)} chars ({source})")

        with open(f"{output_dir}/{poster_id}_raw.md", "w") as f:
            f.write(text)

    # Phase 2: JSON structuring
    log("\n" + "=" * 40)
    log("PHASE 2: JSON Structuring (Transformers)")
    log("=" * 40)

    model, tokenizer = load_json_model()
    results = []

    for i, (poster_path, ref_json_path, poster_id) in enumerate(pairs, 1):
        log(f"\n[{i}/{len(pairs)}] {poster_id}")

        with open(ref_json_path) as f:
            reference = json.load(f)

        raw_text, source = raw_texts[poster_id]

        t0 = time.time()
        try:
            generated = extract_json_with_retry(raw_text, model, tokenizer)

            with open(f"{output_dir}/{poster_id}_extracted.json", "w") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)

            m = calculate_metrics(generated, reference)
            p = passes(m)

            is_error = "error" in generated
            err_str = " [ERR]" if is_error else ""
            sec_str = f" (sec:{m['gen_sections']}/{m['ref_sections']})"

            log(
                f"   w={m['word_capture']:.2f} r={m['rouge_l']:.2f} n={m['number_capture']:.2f} f={m['field_proportion']:.2f} {'✅' if p else '❌'}{err_str}{sec_str} [{time.time() - t0:.1f}s]"
            )

            results.append(
                {
                    "poster_id": poster_id,
                    "metrics": m,
                    "passes": bool(p),
                    "error": bool(is_error),
                    "source": source,
                }
            )

        except Exception as e:
            log(f"   ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append({"poster_id": poster_id, "error": str(e), "passes": False})

        torch.cuda.empty_cache()

    # Final cleanup - unload all models before exit
    unload_json_model()
    unload_vision_model()
    log("   ✓ All models unloaded, GPU memory freed")

    # Summary
    log("\n" + "=" * 60)
    successful = [r for r in results if "metrics" in r]
    if successful:
        avg = {
            k: float(np.mean([r["metrics"][k] for r in successful]))
            for k in ["word_capture", "rouge_l", "number_capture", "field_proportion"]
        }
        passed = sum(1 for r in successful if r["passes"])
        errors = sum(1 for r in successful if r.get("error"))

        log(
            f"AVG: w={avg['word_capture']:.3f} r={avg['rouge_l']:.3f} n={avg['number_capture']:.3f} f={avg['field_proportion']:.3f}"
        )
        log(
            f"PASSING: {passed}/{len(successful)} ({100 * passed / len(successful):.0f}%)"
        )
        if errors:
            log(f"ERRORS: {errors}/{len(successful)}")
        log("")
        for r in successful:
            s = "✅" if r["passes"] else "❌"
            e = " [ERR]" if r.get("error") else ""
            m = r["metrics"]
            src = r.get("source", "?")
            log(
                f"  {s} {r['poster_id']}: w={m['word_capture']:.2f} r={m['rouge_l']:.2f} n={m['number_capture']:.2f} f={m['field_proportion']:.2f} sec:{m['gen_sections']}/{m['ref_sections']}{e} [{src}]"
            )

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--output-dir", default="./llama_v24_output")
    args = parser.parse_args()
    run(args.annotation_dir, args.output_dir)
