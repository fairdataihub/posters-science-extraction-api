# FAIR Pipeline Error Codes

Every code surfaced by the posters.science extraction pipeline carries a
**FAIR-PP** identifier — making failures *Findable*, the cause *Accessible*,
the format *Interoperable* across services, and the fix *Reusable* in
automation.

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

## Severity

Each code has a **severity**:

| Severity | Meaning |
|----------|---------|
| `error` | **Blocks** the pipeline — the job fails and no metadata is saved. |
| `warning` | **Noted** but the document continues through extraction. Warnings are attached to the final result in `_pipeline_warnings`. |

---

## Error Code Reference

### FAIR-PP1x — PosterSentry Classification Gate

These fire **before** the expensive LLM extraction step. The lightweight,
CPU-only [PosterSentry](https://huggingface.co/fairdataihub/poster-sentry)
model screens every incoming document.

| Code | Title | Severity | HTTP | Description |
|------|-------|----------|------|-------------|
| `FAIR-PP10` | Not a Scientific Poster | **error** | 422 | The document was classified as a non-poster (confidence below warn floor, default 0.50). Hard reject. |
| `FAIR-PP11` | Low Classification Confidence | **warning** | 200 | Confidence is between the warn floor (0.50) and the threshold (0.65). Borderline — document proceeds with a warning. |
| `FAIR-PP12` | Classification Failed | **error** | 500 | An error occurred during classification — file may be corrupted or unsupported. |
| `FAIR-PP13` | Sentry Model Unavailable | **error** | 503 | PosterSentry model could not be loaded. Transient — retry later. |

**Decision tree:**

```
confidence ≥ THRESHOLD (0.65)          →  clean pass
WARN_FLOOR (0.50) ≤ conf < THRESHOLD  →  pass + FAIR-PP11 warning
confidence < WARN_FLOOR  or !is_poster →  FAIR-PP10 error (reject)
```

**Environment knobs:**

| Variable | Default | Effect |
|----------|---------|--------|
| `SENTRY_ENABLED` | `true` | Set `false` to bypass all sentry checks. |
| `SENTRY_CONFIDENCE_THRESHOLD` | `0.65` | Minimum confidence for a clean pass (no warning). |
| `SENTRY_WARN_FLOOR` | `0.50` | Below this, the document is hard-rejected. Between floor and threshold, a warning is attached. |
| `SENTRY_ALLOW_ON_ERROR` | `true` | When `true`, model-load or classify errors let the document through with a warning instead of rejecting. |

---

### FAIR-PP2x — Text Extraction

| Code | Title | Severity | HTTP | Description |
|------|-------|----------|------|-------------|
| `FAIR-PP20` | No Text Extracted | **error** | 422 | No usable text from pdfalto, PyMuPDF, or Qwen vision. |
| `FAIR-PP21` | Unsupported File Format | **error** | 415 | File extension is not PDF, JPG/JPEG, or PNG. |
| `FAIR-PP22` | PDF Layout Analysis Fallback | **warning** | 200 | pdfalto failed; PyMuPDF fallback was used. Extraction quality may be reduced. |

---

### FAIR-PP3x — JSON Structuring (LLM)

| Code | Title | Severity | HTTP | Description |
|------|-------|----------|------|-------------|
| `FAIR-PP30` | JSON Structuring Failed | **error** | 500 | LLM could not produce valid JSON after primary + retry + fallback prompts. |
| `FAIR-PP31` | Truncated LLM Output | **warning** | 200 | JSON was truncated but partial results were saved. Some sections may be missing. |
| `FAIR-PP32` | Extraction Model Unavailable | **error** | 503 | Llama model could not be loaded (missing weights or insufficient GPU). |

---

### FAIR-PP4x — Validation / Schema

| Code | Title | Severity | HTTP | Description |
|------|-------|----------|------|-------------|
| `FAIR-PP40` | Schema Validation Issues | **warning** | 200 | Extracted JSON has schema deviations. Partial results were saved. |
| `FAIR-PP41` | No Creators Extracted | **warning** | 200 | The `creators` array is empty — author names not found. |
| `FAIR-PP42` | No Title Extracted | **warning** | 200 | The `titles` array is empty — poster title not found. |

---

### FAIR-PP5x — Infrastructure / Resources

| Code | Title | Severity | HTTP | Description |
|------|-------|----------|------|-------------|
| `FAIR-PP50` | GPU Unavailable | **error** | 503 | No CUDA GPU detected. Pipeline needs ≥16 GB VRAM. |
| `FAIR-PP51` | File Download Failed | **error** | 502 | Bunny storage download failed (file deleted or outage). |
| `FAIR-PP52` | Database Error | **error** | 503 | PostgreSQL operation failed. |

---

## Severity Summary

| Severity | Codes |
|----------|-------|
| **error** (blocks pipeline) | `FAIR-PP10`, `FAIR-PP12`, `FAIR-PP13`, `FAIR-PP20`, `FAIR-PP21`, `FAIR-PP30`, `FAIR-PP32`, `FAIR-PP50`, `FAIR-PP51`, `FAIR-PP52` |
| **warning** (continues with flag) | `FAIR-PP11`, `FAIR-PP22`, `FAIR-PP31`, `FAIR-PP40`, `FAIR-PP41`, `FAIR-PP42` |

---

## JSON Response Shapes

### Error (job fails)

Recorded in `ExtractionJob.error`:

```json
{
  "error_code": "FAIR-PP10",
  "severity": "error",
  "error": "Not a Scientific Poster",
  "message": "PosterSentry classified this document as a non-poster …",
  "category": "sentry",
  "detail": "Classified as non-poster (is_poster=False, confidence=0.012). File: upload.pdf"
}
```

### Warning (job succeeds, flag attached)

Attached to the extraction result in `_pipeline_warnings`:

```json
{
  "_pipeline_warnings": [
    {
      "error_code": "FAIR-PP11",
      "severity": "warning",
      "error": "Low Classification Confidence",
      "message": "PosterSentry could not confidently determine …",
      "category": "sentry",
      "detail": "Poster confidence 0.583 is below threshold 0.65 (warn floor 0.50). File: borderline.pdf"
    }
  ]
}
```

### Shared fields

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | `string` | The `FAIR-PPxx` code. |
| `severity` | `string` | `"error"` or `"warning"`. |
| `error` | `string` | Short human-readable title. |
| `message` | `string` | Longer explanation (same for every instance of this code). |
| `category` | `string` | One of: `sentry`, `extraction`, `llm`, `validation`, `infrastructure`. |
| `detail` | `string?` | Instance-specific context (file name, confidence score, etc.). |

---

## Pipeline Flow

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
                    └──┬──────────┬──────────┬──┘
                       │          │          │
                  conf ≥ 0.65     │     conf < 0.50
                  (clean pass)    │     or !is_poster
                       │          │          │
                       │   0.50 ≤ conf       │
                       │     < 0.65          │
                       │  (⚠ FAIR-PP11)      │
                       │   pass w/ warning   │
                       │          │   ┌──────▼──────────┐
                       ├──────────┘   │  FAIR-PP10      │
                       │              │  Job → failed    │
                       │              └─────────────────┘
                       │
              ┌────────▼───────────┐
              │  extract_poster()  │  GPU  (Llama 3.1 8B)
              │  Raw text → JSON   │  ⚠ FAIR-PP22/31  errors: PP20/30
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  validate_and_fix  │  ⚠ FAIR-PP40/41/42
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  PosterMetadata    │  FAIR-PP52 on DB error
              │  saved to DB       │
              │  + _pipeline_warnings
              └────────────────────┘
```

---

## Adding New Error Codes

1. Pick the right category prefix (`FAIR-PP1x`–`FAIR-PP5x`).
2. Choose the next unused sequence number.
3. Add a `FAIRError` constant in `error_codes.py` with the correct `severity`.
4. Update this document.

All `FAIRError` instances are auto-collected via `ALL_ERRORS` / `ALL_CODES`
in `error_codes.py` — no registration step needed.

---

## See Also

- [PosterSentry model card](https://huggingface.co/fairdataihub/poster-sentry)
- [poster-json-schema](https://github.com/fairdataihub/poster-json-schema)
- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)
