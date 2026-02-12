"""
FAIR Error Codes for the posters.science extraction pipeline.

Each error code follows the format: FAIR-PPXX

    FAIR  — Findable, Accessible, Interoperable, Reusable (the FAIR principles)
    PP    — Poster Pipeline
    XX    — Numeric code within the category

Categories:
    FAIR-PP1x  — Sentry / classification gate errors
    FAIR-PP2x  — Text extraction errors
    FAIR-PP3x  — JSON structuring / LLM errors
    FAIR-PP4x  — Validation / schema errors
    FAIR-PP5x  — Infrastructure / resource errors

See docs/FAIR_ERROR_CODES.md for full documentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FAIRError:
    """Immutable descriptor for a pipeline error code."""

    code: str
    title: str
    message: str
    http_status: int = 400
    category: str = "general"

    def to_dict(self, detail: Optional[str] = None) -> dict:
        """Serialize for JSON API responses."""
        d = {
            "error_code": self.code,
            "error": self.title,
            "message": self.message,
            "category": self.category,
        }
        if detail:
            d["detail"] = detail
        return d


# ---------------------------------------------------------------------------
# FAIR-PP1x  —  PosterSentry / classification gate
# ---------------------------------------------------------------------------

SENTRY_NOT_A_POSTER = FAIRError(
    code="FAIR-PP10",
    title="Not a Scientific Poster",
    message=(
        "PosterSentry classified this document as a non-poster "
        "(e.g. paper, proceedings, newsletter, abstract book). "
        "Only scientific posters are accepted by the extraction pipeline."
    ),
    http_status=422,
    category="sentry",
)

SENTRY_LOW_CONFIDENCE = FAIRError(
    code="FAIR-PP11",
    title="Low Classification Confidence",
    message=(
        "PosterSentry could not confidently determine whether this document "
        "is a scientific poster. The confidence score fell below the required "
        "threshold. Please verify the file is a single-page scientific poster."
    ),
    http_status=422,
    category="sentry",
)

SENTRY_CLASSIFICATION_FAILED = FAIRError(
    code="FAIR-PP12",
    title="Classification Failed",
    message=(
        "PosterSentry encountered an error while classifying the document. "
        "The file may be corrupted, password-protected, or in an unsupported format."
    ),
    http_status=500,
    category="sentry",
)

SENTRY_MODEL_UNAVAILABLE = FAIRError(
    code="FAIR-PP13",
    title="Sentry Model Unavailable",
    message=(
        "The PosterSentry classification model could not be loaded. "
        "This is a transient infrastructure issue — please retry later."
    ),
    http_status=503,
    category="sentry",
)

# ---------------------------------------------------------------------------
# FAIR-PP2x  —  Text extraction
# ---------------------------------------------------------------------------

EXTRACTION_NO_TEXT = FAIRError(
    code="FAIR-PP20",
    title="No Text Extracted",
    message=(
        "The pipeline could not extract any text from this file. "
        "The poster may be a pure image without embedded text, or the "
        "file may be corrupted."
    ),
    http_status=422,
    category="extraction",
)

EXTRACTION_UNSUPPORTED_FORMAT = FAIRError(
    code="FAIR-PP21",
    title="Unsupported File Format",
    message=(
        "The uploaded file format is not supported. "
        "Accepted formats: PDF, JPG/JPEG, PNG."
    ),
    http_status=415,
    category="extraction",
)

EXTRACTION_PDFALTO_FAILED = FAIRError(
    code="FAIR-PP22",
    title="PDF Layout Analysis Failed",
    message=(
        "The pdfalto layout analysis tool failed to process this PDF. "
        "Falling back to basic text extraction is possible but may "
        "reduce extraction quality."
    ),
    http_status=500,
    category="extraction",
)

# ---------------------------------------------------------------------------
# FAIR-PP3x  —  JSON structuring / LLM
# ---------------------------------------------------------------------------

LLM_JSON_PARSE_FAILED = FAIRError(
    code="FAIR-PP30",
    title="JSON Structuring Failed",
    message=(
        "The LLM was unable to produce valid JSON output after multiple "
        "attempts including retry and fallback prompts."
    ),
    http_status=500,
    category="llm",
)

LLM_TRUNCATED_OUTPUT = FAIRError(
    code="FAIR-PP31",
    title="Truncated LLM Output",
    message=(
        "The LLM output was truncated — the poster may contain more text "
        "than the model can process in a single pass."
    ),
    http_status=500,
    category="llm",
)

LLM_MODEL_UNAVAILABLE = FAIRError(
    code="FAIR-PP32",
    title="Extraction Model Unavailable",
    message=(
        "The Llama-3.1-8B-Poster-Extraction model could not be loaded. "
        "GPU memory may be insufficient or the model weights are missing."
    ),
    http_status=503,
    category="llm",
)

# ---------------------------------------------------------------------------
# FAIR-PP4x  —  Validation / schema
# ---------------------------------------------------------------------------

VALIDATION_SCHEMA_FAILED = FAIRError(
    code="FAIR-PP40",
    title="Schema Validation Failed",
    message=(
        "The extracted JSON does not conform to the poster-json-schema. "
        "Required fields may be missing or malformed."
    ),
    http_status=422,
    category="validation",
)

VALIDATION_EMPTY_CREATORS = FAIRError(
    code="FAIR-PP41",
    title="No Creators Extracted",
    message=(
        "The extraction produced an empty creators array. "
        "Author information could not be identified in the poster."
    ),
    http_status=422,
    category="validation",
)

VALIDATION_EMPTY_TITLE = FAIRError(
    code="FAIR-PP42",
    title="No Title Extracted",
    message=(
        "The extraction produced an empty titles array. "
        "The poster title could not be identified."
    ),
    http_status=422,
    category="validation",
)

# ---------------------------------------------------------------------------
# FAIR-PP5x  —  Infrastructure / resource
# ---------------------------------------------------------------------------

INFRA_GPU_UNAVAILABLE = FAIRError(
    code="FAIR-PP50",
    title="GPU Unavailable",
    message=(
        "No CUDA-capable GPU is available for model inference. "
        "The extraction pipeline requires a GPU with at least 16 GB VRAM."
    ),
    http_status=503,
    category="infrastructure",
)

INFRA_DOWNLOAD_FAILED = FAIRError(
    code="FAIR-PP51",
    title="File Download Failed",
    message=(
        "Could not download the poster file from storage. "
        "The file may have been deleted or the storage service is down."
    ),
    http_status=502,
    category="infrastructure",
)

INFRA_DATABASE_ERROR = FAIRError(
    code="FAIR-PP52",
    title="Database Error",
    message=(
        "A database operation failed while processing the extraction job."
    ),
    http_status=503,
    category="infrastructure",
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

# Collect all FAIRError instances defined in this module
ALL_ERRORS: dict[str, FAIRError] = {
    name: obj
    for name, obj in globals().items()
    if isinstance(obj, FAIRError)
}

ALL_CODES: dict[str, FAIRError] = {
    err.code: err for err in ALL_ERRORS.values()
}


def get_error(code: str) -> Optional[FAIRError]:
    """Look up a FAIRError by its code string (e.g. 'FAIR-PP10')."""
    return ALL_CODES.get(code)
