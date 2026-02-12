# FAIR Pipeline Error Codes

Every error surfaced by the posters.science extraction pipeline carries a
**FAIR-PP** code — a structured identifier that makes failures *Findable*,
the cause *Accessible*, the format *Interoperable* across services, and the
fix *Reusable* in automation.

## Format

```
FAIR-PP<category><sequence>
         │         │
         │         └── Numeric sequence within category (0-9)
         └──────────── Category digit (1-5)
```

| Prefix | Category | Description |
|--------|----------|-------------|
| `FAIR-PP1x` | **Sentry** | PosterSentry classification gate |
| `FAIR-PP2x` | **Extraction** | Raw text extraction (pdfalto / OCR) |
| `FAIR-PP3x` | **LLM** | JSON structuring via Llama 3.1 8B |
| `FAIR-PP4x` | **Validation** | Schema validation & field checks |
| `FAIR-PP5x` | **Infrastructure** | GPU, storage, database resources |

---

## Error Code Reference

### FAIR-PP1x — PosterSentry Classification Gate

These errors fire **before** the expensive LLM extraction step.  The
lightweight, CPU-only [PosterSentry](https://huggingface.co/fairdataihub/poster-sentry)
model screens every incoming document.

| Code | Title | HTTP | Description |
|------|-------|------|-------------|
| `FAIR-PP10` | Not a Scientific Poster | 422 | The document was classified as a non-poster (paper, proceedings, newsletter, abstract book, etc.). Only single-page scientific posters are accepted. |
| `FAIR-PP11` | Low Classification Confidence | 422 | PosterSentry could not confidently determine the document type. Confidence fell below the configurable threshold (default 0.65). |
| `FAIR-PP12` | Classification Failed | 500 | An error occurred during classification — the file may be corrupted, password-protected, or in an unexpected format. |
| `FAIR-PP13` | Sentry Model Unavailable | 503 | The PosterSentry model could not be loaded. Transient issue — retry later. |

**Environment knobs:**

| Variable | Default | Effect |
|----------|---------|--------|
| `SENTRY_ENABLED` | `true` | Set `false` to bypass all sentry checks. |
| `SENTRY_CONFIDENCE_THRESHOLD` | `0.65` | Minimum confidence to accept a poster. |
| `SENTRY_ALLOW_ON_ERROR` | `true` | When `true`, model-load or classify errors let the document through instead of rejecting it. Useful during rollout. |

---

### FAIR-PP2x — Text Extraction

| Code | Title | HTTP | Description |
|------|-------|------|-------------|
| `FAIR-PP20` | No Text Extracted | 422 | Neither pdfalto nor PyMuPDF (nor Qwen vision for images) could extract usable text from the file. |
| `FAIR-PP21` | Unsupported File Format | 415 | The file extension is not PDF, JPG/JPEG, or PNG. |
| `FAIR-PP22` | PDF Layout Analysis Failed | 500 | pdfalto crashed or timed out. A PyMuPDF fallback may still produce results at reduced quality. |

---

### FAIR-PP3x — JSON Structuring (LLM)

| Code | Title | HTTP | Description |
|------|-------|------|-------------|
| `FAIR-PP30` | JSON Structuring Failed | 500 | The LLM was unable to produce valid JSON after primary, retry, and fallback prompt attempts. |
| `FAIR-PP31` | Truncated LLM Output | 500 | The generated JSON was truncated — the poster may exceed the model's context window. |
| `FAIR-PP32` | Extraction Model Unavailable | 503 | The Llama-3.1-8B-Poster-Extraction model could not be loaded (missing weights or insufficient GPU memory). |

---

### FAIR-PP4x — Validation / Schema

| Code | Title | HTTP | Description |
|------|-------|------|-------------|
| `FAIR-PP40` | Schema Validation Failed | 422 | The extracted JSON does not conform to the [poster-json-schema](https://github.com/fairdataihub/poster-json-schema). |
| `FAIR-PP41` | No Creators Extracted | 422 | The `creators` array is empty — author names could not be identified. |
| `FAIR-PP42` | No Title Extracted | 422 | The `titles` array is empty — the poster title could not be identified. |

---

### FAIR-PP5x — Infrastructure / Resources

| Code | Title | HTTP | Description |
|------|-------|------|-------------|
| `FAIR-PP50` | GPU Unavailable | 503 | No CUDA-capable GPU detected. The pipeline needs at least 16 GB VRAM. |
| `FAIR-PP51` | File Download Failed | 502 | Downloading the poster file from Bunny storage failed (deleted file or storage outage). |
| `FAIR-PP52` | Database Error | 503 | A PostgreSQL operation failed while processing the extraction job. |

---

## JSON Error Response Shape

When an error is recorded in the `ExtractionJob.error` column (or returned via
a future HTTP error response), it follows this format:

```json
{
  "error_code": "FAIR-PP10",
  "error": "Not a Scientific Poster",
  "message": "PosterSentry classified this document as a non-poster …",
  "category": "sentry",
  "detail": "Classified as non-poster with confidence 0.123. File: upload.pdf"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | `string` | The `FAIR-PPxx` code. |
| `error` | `string` | Short human-readable title. |
| `message` | `string` | Longer explanation (same for every instance of this code). |
| `category` | `string` | One of: `sentry`, `extraction`, `llm`, `validation`, `infrastructure`. |
| `detail` | `string?` | Instance-specific context (file name, confidence score, stack trace excerpt, etc.). |

---

## Pipeline Flow with PosterSentry Gate

```
                         ┌────────────────────┐
  PDF uploaded            │   Bunny Storage    │
  to platform  ────────► │   (file stored)    │
                         └────────┬───────────┘
                                  │
                         ┌────────▼───────────┐
                         │  ExtractionJob      │
                         │  status: pending    │
                         └────────┬───────────┘
                                  │
                         ┌────────▼───────────┐
                         │  Download file      │  FAIR-PP51 on failure
                         └────────┬───────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      PosterSentry         │  CPU-only (~300 docs/s)
                    │   "Is this a poster?"     │
                    └─────┬───────────┬─────────┘
                          │           │
                     YES  │           │  NO / LOW CONFIDENCE
                          │           │
                          │    ┌──────▼──────────┐
                          │    │  FAIR-PP10 /     │
                          │    │  FAIR-PP11       │
                          │    │  Job → failed    │
                          │    └─────────────────┘
                          │
                 ┌────────▼───────────┐
                 │  extract_poster()  │  GPU  (Llama 3.1 8B)
                 │  Raw text → JSON   │  FAIR-PP20/30 on failure
                 └────────┬───────────┘
                          │
                 ┌────────▼───────────┐
                 │  validate_and_fix  │  FAIR-PP40/41/42
                 └────────┬───────────┘
                          │
                 ┌────────▼───────────┐
                 │  PosterMetadata    │  FAIR-PP52 on DB error
                 │  saved to DB       │
                 └────────────────────┘
```

---

## Adding New Error Codes

1. Pick the right category prefix (`FAIR-PP1x`–`FAIR-PP5x`).
2. Choose the next unused sequence number.
3. Add a `FAIRError` constant in `error_codes.py`.
4. Update this document.

All `FAIRError` instances are auto-collected via `ALL_ERRORS` / `ALL_CODES`
in `error_codes.py` — no registration step needed.

---

## See Also

- [PosterSentry model card](https://huggingface.co/fairdataihub/poster-sentry)
- [poster-json-schema](https://github.com/fairdataihub/poster-json-schema)
- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)
