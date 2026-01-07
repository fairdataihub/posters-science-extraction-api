#!/usr/bin/env python3
"""
Poster Science - Scientific Poster Metadata Extraction Pipeline

Extracts structured metadata from scientific poster PDFs and images
into JSON format conforming to the posters-science schema.

Models (Ollama):
- Llama 3.1 8B Instruct (Q8_0): JSON structuring
- Qwen3-VL 4B Instruct (Q8_0): Vision OCR for images

Requirements:
- pdfalto: PDF layout analysis tool (https://github.com/kermitt2/pdfalto)
- Ollama: Local LLM server (https://ollama.ai)
- CUDA-capable GPU with ≥8GB VRAM

Environment Variables:
- PDFALTO_PATH: Path to pdfalto binary (required for PDF processing)
- CUDA_VISIBLE_DEVICES: GPU device(s) to use (default: 0)
"""

import os
import shutil
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

import ollama
from rouge_score import rouge_scorer

# Ollama models
OLLAMA_JSON_MODEL = "llama3.1:8b-instruct-q8_0"
OLLAMA_VISION_MODEL = "qwen3-vl:4b-instruct-q8_0"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Find pdfalto: check environment variable, then project path, then PATH
PDFALTO_PATH = os.environ.get("PDFALTO_PATH")
if not PDFALTO_PATH:
    # Check project-specific path
    project_pdfalto = Path(__file__).parent.parent.parent / "pdfalto" / "pdfalto"
    if project_pdfalto.exists():
        PDFALTO_PATH = str(project_pdfalto)
    else:
        # Check if pdfalto is in PATH
        pdfalto_in_path = shutil.which("pdfalto")
        if pdfalto_in_path:
            PDFALTO_PATH = pdfalto_in_path
        else:
            PDFALTO_PATH = None

MAX_JSON_TOKENS = 18000
MAX_RETRY_TOKENS = 24000


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================
# VISION OCR FOR IMAGES
# ============================

_vision_model_ready = False


def load_vision_model():
    """Ensure Ollama vision model is available."""
    global _vision_model_ready
    if not _vision_model_ready:
        log(f"Checking Ollama vision model: {OLLAMA_VISION_MODEL}")
        try:
            ollama.show(OLLAMA_VISION_MODEL)
            log(f"   ✓ Vision model ready: {OLLAMA_VISION_MODEL}")
        except ollama.ResponseError:
            log(f"   Pulling {OLLAMA_VISION_MODEL}...")
            ollama.pull(OLLAMA_VISION_MODEL)
            log(f"   ✓ Vision model pulled: {OLLAMA_VISION_MODEL}")
        _vision_model_ready = True


def unload_vision_model():
    """Unload Ollama vision model to free GPU memory."""
    global _vision_model_ready
    if _vision_model_ready:
        try:
            subprocess.run(
                ["ollama", "stop", OLLAMA_VISION_MODEL],
                capture_output=True,
                timeout=30
            )
            time.sleep(3)
            log(f"   ✓ Vision model unloaded: {OLLAMA_VISION_MODEL}")
        except Exception as e:
            log(f"   ⚠️ Could not unload vision model: {e}")
        _vision_model_ready = False


def _deduplicate_lines(text: str) -> str:
    """Remove duplicate lines (case-insensitive, whitespace-normalized)."""
    lines = text.split('\n')
    seen = set()
    result = []
    for line in lines:
        norm = re.sub(r'\s+', ' ', line).strip().lower()
        if norm and norm not in seen:
            result.append(line)
            seen.add(norm)
        elif not norm:
            result.append(line)
    return '\n'.join(result)


def extract_text_with_vision(image_path: str) -> str:
    """Use Ollama vision model for image OCR."""
    load_vision_model()
    
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
- Include all bullet points and lists
- Do NOT repeat any content"""

    response = ollama.chat(
        model=OLLAMA_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path],
        }],
        think=False,
        options={
            "num_ctx": 32768,
            "num_predict": 4000,
            "temperature": 0,
            "repeat_penalty": 1.3,
        },
    )
    
    text = response.message.content if hasattr(response.message, 'content') else response['message']['content']
    return _deduplicate_lines(text)


# ============================
# PDF TEXT EXTRACTION
# ============================


def extract_text_with_pdfalto(pdf_path: str) -> str:
    """Extract text from PDF using pdfalto."""
    if PDFALTO_PATH is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "output.xml")
            result = subprocess.run(
                [PDFALTO_PATH, "-noImage", "-readingOrder", pdf_path, xml_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pdfalto returned code {result.returncode}: {result.stderr}")
            if not os.path.exists(xml_path):
                raise RuntimeError("pdfalto did not produce output XML")
            return parse_alto_xml(xml_path)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pdfalto timeout processing {pdf_path}")
    except Exception as e:
        raise RuntimeError(f"pdfalto error: {e}")


def parse_alto_xml(xml_path: str) -> str:
    """Parse ALTO XML output from pdfalto."""
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
    """Fallback PDF extraction using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text.strip()


def get_raw_text(
    poster_path: str, poster_id: str = None, output_dir: str = None
) -> tuple:
    """Get raw text from poster."""
    ext = Path(poster_path).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        # Check cache if output_dir is provided
        if output_dir and poster_id:
            cache_file = Path(output_dir) / f"{poster_id}_raw.txt"
            if cache_file.exists():
                with open(cache_file) as f:
                    text = f.read()
                if len(text) > 500:
                    return text, "vision_cached"

        text = extract_text_with_vision(poster_path)
        return text, "vision"

    if ext == ".pdf":
        text = extract_text_with_pdfalto(poster_path)
        if text and len(text) > 500:
            return text, "pdfalto"
        return extract_text_with_pymupdf(poster_path), "pymupdf"

    return "", "unknown"


# ============================
# JSON MODEL (Ollama)
# ============================

_json_model_ready = False


def load_json_model():
    """Ensure Ollama JSON model is available."""
    global _json_model_ready
    if not _json_model_ready:
        log(f"Checking Ollama JSON model: {OLLAMA_JSON_MODEL}")
        try:
            ollama.show(OLLAMA_JSON_MODEL)
            log(f"   ✓ Ollama model ready: {OLLAMA_JSON_MODEL}")
        except ollama.ResponseError:
            log(f"   Pulling {OLLAMA_JSON_MODEL}...")
            ollama.pull(OLLAMA_JSON_MODEL)
            log(f"   ✓ Model pulled: {OLLAMA_JSON_MODEL}")
        _json_model_ready = True
    return None, None  # Return tuple for API compatibility


def generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate JSON using Ollama."""
    response = ollama.chat(
        model=OLLAMA_JSON_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": 32768,
            "num_predict": max_tokens,
            "temperature": 0,
        },
    )
    return response.message.content if hasattr(response.message, 'content') else response['message']['content']


# PRIMARY PROMPT
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
    """Check if JSON appears to be truncated."""
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")

    if open_braces > 0 or open_brackets > 0:
        return True

    if json_str.rstrip().endswith((",", ":", '"')):
        return True

    return False


def extract_json_with_retry(raw_text: str, model, tokenizer) -> dict:
    """Extract JSON with retry logic for truncation."""
    prompt = EXTRACTION_PROMPT.format(raw_text=raw_text)

    response = generate(model, tokenizer, prompt, MAX_JSON_TOKENS)
    result = robust_json_parse(response)

    if "error" in result or is_truncated(result.get("raw", "")):
        log(f"   ⚠️ Truncation/error detected, retrying with {MAX_RETRY_TOKENS} tokens...")
        response = generate(model, tokenizer, prompt, MAX_RETRY_TOKENS)
        result = robust_json_parse(response)

    if "error" in result or is_truncated(result.get("raw", "")):
        log("   ⚠️ Still truncating, trying FALLBACK shorter prompt...")
        fallback_prompt = FALLBACK_PROMPT.format(raw_text=raw_text)
        response = generate(model, tokenizer, fallback_prompt, MAX_RETRY_TOKENS)
        result = robust_json_parse(response)

    return result


# ============================
# JSON PARSING
# ============================


def robust_json_parse(response: str) -> dict:
    """Parse JSON with robust error handling and repair."""
    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    start = response.find("{")
    if start == -1:
        return {"error": "No JSON found", "raw": response[:3000]}

    json_str = response[start:]

    json_str = repair_unescaped_quotes(json_str)

    extracted = extract_first_json_object(json_str)
    if extracted:
        json_str = extracted

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    repair_funcs = [
        repair_unescaped_quotes,
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

    try:
        repaired = repair_all(repair_unescaped_quotes(json_str))
        return json.loads(repaired)
    except Exception:
        pass

    return {"error": "JSON parse failed", "raw": json_str[:3000]}


def extract_first_json_object(s: str) -> str:
    """Extract the first complete JSON object from a string."""
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
    s = re.sub(
        r'(\d+\s*(?:pc|km|m|cm|mm|Hz|kHz|MHz|GHz|s|ms|ns|arcsec|arcmin|deg))/"',
        r'\1/\\"',
        s,
    )
    s = re.sub(r'\((\d+\.?\d*\s*\w+)/"\)', r'(\1/\\")', s)
    return s


def repair_trailing_commas(s: str) -> str:
    """Remove trailing commas before closing brackets."""
    return re.sub(r",\s*([}\]])", r"\1", s)


def repair_unicode(s: str) -> str:
    """Fix unicode escape sequences."""
    s = re.sub(r"\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])", "", s)
    s = re.sub(r"[\x00-\x1f]", " ", s)
    return s


def repair_truncation(s: str) -> str:
    """Attempt to close truncated JSON."""
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
        if '"sectionContent":' in s[-1000:] or "sectionContent" in s[-500:]:
            open_braces += 1

    s = s.rstrip()
    while s and s[-1] not in '{}[]"0123456789truefalsenull':
        s = s[:-1].rstrip()
    if s.endswith(","):
        s = s[:-1]

    s += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    return s


def repair_all(s: str) -> str:
    """Apply all repair functions."""
    s = repair_unescaped_quotes(s)
    s = repair_unicode(s)
    s = repair_trailing_commas(s)
    s = repair_truncation(s)
    return s


# ============================
# METRICS
# ============================


def get_all_text(d) -> str:
    """Extract all text from a nested structure."""
    if isinstance(d, dict):
        return " ".join(get_all_text(v) for v in d.values())
    elif isinstance(d, list):
        return " ".join(get_all_text(item) for item in d)
    elif isinstance(d, str):
        return d
    return ""


def get_section_texts(d) -> list:
    """Extract section texts from JSON."""
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
    """Normalize text for comparison."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)

    space_chars = [
        "\xa0", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",
        "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
        "\u202f", "\u205f", "\u3000",
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
    """Strip to alphanumeric characters only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", "", text)).strip().lower()


def calculate_forgiving_rouge(
    gen_text: str, ref_text: str, gen_sections: list, ref_sections: list
) -> float:
    """Calculate ROUGE-L with section-level matching."""
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
    """Calculate number capture rate."""
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
    """Calculate all metrics comparing generated to reference."""
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
    """Check if metrics pass thresholds."""
    return (
        m["word_capture"] >= 0.75
        and m["rouge_l"] >= 0.75
        and m["number_capture"] >= 0.75
        and 0.3 <= m["field_proportion"] <= 2.5
    )


def find_pairs(annotation_dir: str):
    """Find poster/annotation pairs in directory."""
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
    log(f"Processing poster: {poster_path}")

    raw_text, source = get_raw_text(poster_path)

    if not raw_text or source == "unknown":
        return {
            "error": f"Failed to extract text from file. Unsupported format or extraction failed."
        }

    log(f"Extracted {len(raw_text)} characters using {source}")

    model, tokenizer = load_json_model()

    try:
        generated = extract_json_with_retry(raw_text, model, tokenizer)

        ext = Path(poster_path).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            unload_vision_model()

        return generated
    except Exception as e:
        log(f"ERROR processing poster: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run(annotation_dir: str, output_dir: str):
    """Run extraction pipeline on all posters."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(annotation_dir)

    log("=" * 60)
    log("POSTER EXTRACTION PIPELINE (Ollama)")
    log("=" * 60)
    log(f"Posters: {len(pairs)}")
    log(f"JSON Model: {OLLAMA_JSON_MODEL}")
    log(f"Vision Model: {OLLAMA_VISION_MODEL}")

    image_posters = [
        p for p in pairs if Path(p[0]).suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    pdf_posters = [p for p in pairs if Path(p[0]).suffix.lower() == ".pdf"]

    log(f"Image posters: {[p[2] for p in image_posters]}")
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

            with open(f"{output_dir}/{poster_id}_raw.txt", "w") as f:
                f.write(text)

        unload_vision_model()

    for poster_path, ref_json_path, poster_id in pdf_posters:
        text, source = get_raw_text(poster_path, poster_id, output_dir)
        raw_texts[poster_id] = (text, source)
        log(f"   {poster_id}: {len(text)} chars ({source})")

        with open(f"{output_dir}/{poster_id}_raw.txt", "w") as f:
            f.write(text)

    # Phase 2: JSON structuring
    log("\n" + "=" * 40)
    log("PHASE 2: JSON Structuring")
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

        gc.collect()

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
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()
    run(args.annotation_dir, args.output_dir)
