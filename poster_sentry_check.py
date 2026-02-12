"""
PosterSentry integration for the extraction API.

Wraps the fairdataihub/poster-sentry model to gate the extraction
pipeline.  Documents are sorted into three buckets:

    confidence ≥ THRESHOLD            → ACCEPTED  (clean pass)
    WARN_FLOOR ≤ confidence < THRESH  → ACCEPTED with WARNING  (borderline)
    confidence < WARN_FLOOR           → REJECTED with ERROR    (not a poster)

Usage in the pipeline:
    from poster_sentry_check import sentry_gate

    result, error, warnings = sentry_gate("/tmp/upload.pdf")
    if error is not None:
        mark_job_failed(conn, job_id, ...)
        return
    # `warnings` is a PipelineWarnings accumulator — pass it downstream

Model reference:
    https://huggingface.co/fairdataihub/poster-sentry
"""

import logging
import os
import time
from typing import Optional, Tuple

from error_codes import (
    FAIRError,
    PipelineWarnings,
    SENTRY_NOT_A_POSTER,
    SENTRY_LOW_CONFIDENCE,
    SENTRY_CLASSIFICATION_FAILED,
    SENTRY_MODEL_UNAVAILABLE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum confidence for a clean pass (no warning).
SENTRY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SENTRY_CONFIDENCE_THRESHOLD", "0.65")
)

# Floor below which the document is hard-rejected.  Anything between
# WARN_FLOOR and THRESHOLD gets a warning but proceeds through extraction.
SENTRY_WARN_FLOOR = float(
    os.environ.get("SENTRY_WARN_FLOOR", "0.50")
)

# When True, a failed or unavailable sentry check lets the document
# through instead of rejecting it.  Useful during rollout / testing.
SENTRY_ALLOW_ON_ERROR = os.environ.get(
    "SENTRY_ALLOW_ON_ERROR", "true"
).lower() in ("1", "true", "yes")

# Set to "false" to completely disable sentry checks (bypass).
SENTRY_ENABLED = os.environ.get(
    "SENTRY_ENABLED", "true"
).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Singleton model holder
# ---------------------------------------------------------------------------

_sentry_instance = None
_sentry_load_error: Optional[str] = None


def _log(msg: str) -> None:
    """Timestamped log matching poster_extraction.py convention."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [sentry] {msg}", flush=True)


def load_sentry():
    """
    Lazily load and cache the PosterSentry model.

    Returns the PosterSentry instance, or None if loading fails.
    """
    global _sentry_instance, _sentry_load_error

    if _sentry_instance is not None:
        return _sentry_instance

    if _sentry_load_error is not None:
        # Already failed once this process — don't retry every request.
        return None

    try:
        from poster_sentry import PosterSentry

        _log("Loading PosterSentry model from HuggingFace …")
        t0 = time.time()

        sentry = PosterSentry()
        sentry.initialize()

        elapsed = time.time() - t0
        _log(f"PosterSentry loaded in {elapsed:.2f}s")
        _sentry_instance = sentry
        return sentry

    except ImportError:
        _sentry_load_error = (
            "poster_sentry package not installed. "
            "Install with: pip install git+https://github.com/fairdataihub/poster-repo-qc.git"
        )
        _log(f"WARN: {_sentry_load_error}")
        return None

    except Exception as e:
        _sentry_load_error = f"Failed to initialize PosterSentry: {e}"
        _log(f"ERROR: {_sentry_load_error}")
        return None


def reset_sentry() -> None:
    """Reset the cached model so the next call to load_sentry() retries."""
    global _sentry_instance, _sentry_load_error
    _sentry_instance = None
    _sentry_load_error = None


def is_sentry_available() -> bool:
    """Check if PosterSentry can be loaded (used by /health)."""
    if not SENTRY_ENABLED:
        return True  # disabled = not an error
    sentry = load_sentry()
    return sentry is not None


def get_sentry_status() -> dict:
    """Return a status dict for the /health endpoint."""
    if not SENTRY_ENABLED:
        return {"status": "disabled"}
    sentry = load_sentry()
    if sentry is not None:
        return {
            "status": "ok",
            "confidence_threshold": SENTRY_CONFIDENCE_THRESHOLD,
            "warn_floor": SENTRY_WARN_FLOOR,
        }
    return {"status": "unavailable", "error": _sentry_load_error or "unknown"}


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------


def sentry_gate(
    file_path: str,
    warnings: Optional[PipelineWarnings] = None,
) -> Tuple[Optional[dict], Optional[FAIRError], PipelineWarnings]:
    """
    Classify a file with PosterSentry before extraction.

    Args:
        file_path: Local path to the downloaded poster file.
        warnings:  Optional existing PipelineWarnings accumulator.
                   A new one is created if not provided.

    Returns:
        (result_dict, error_or_None, warnings)

        - Hard reject (not a poster / infrastructure failure):
              (None, FAIRError, warnings)
        - Borderline poster (confidence between warn_floor and threshold):
              (classification_result, None, warnings)   # warning appended
        - Clean pass:
              (classification_result, None, warnings)
    """
    if warnings is None:
        warnings = PipelineWarnings()

    if not SENTRY_ENABLED:
        _log("PosterSentry is disabled — skipping classification.")
        return ({"is_poster": True, "confidence": 1.0, "skipped": True}, None, warnings)

    sentry = load_sentry()

    if sentry is None:
        detail = _sentry_load_error or "PosterSentry model could not be loaded"
        _log(f"Sentry unavailable: {detail}")
        if SENTRY_ALLOW_ON_ERROR:
            _log("SENTRY_ALLOW_ON_ERROR=true — allowing document through.")
            warnings.add(SENTRY_MODEL_UNAVAILABLE, detail)
            return ({"is_poster": True, "confidence": 0.0, "skipped": True}, None, warnings)
        return (None, SENTRY_MODEL_UNAVAILABLE, warnings)

    # Run classification
    try:
        _log(f"Classifying: {file_path}")
        t0 = time.time()
        result = sentry.classify(file_path)
        elapsed = time.time() - t0
        _log(
            f"Classification complete in {elapsed:.2f}s — "
            f"is_poster={result.get('is_poster')}, "
            f"confidence={result.get('confidence', 0):.3f}"
        )
    except Exception as e:
        detail = f"PosterSentry.classify() raised: {e}"
        _log(f"ERROR: {detail}")
        if SENTRY_ALLOW_ON_ERROR:
            _log("SENTRY_ALLOW_ON_ERROR=true — allowing document through.")
            warnings.add(SENTRY_CLASSIFICATION_FAILED, detail)
            return ({"is_poster": True, "confidence": 0.0, "skipped": True}, None, warnings)
        return (None, SENTRY_CLASSIFICATION_FAILED, warnings)

    is_poster = result.get("is_poster", False)
    confidence = result.get("confidence", 0.0)

    # --- Decision tree -------------------------------------------------------
    #
    #   confidence ≥ THRESHOLD              → clean pass
    #   WARN_FLOOR ≤ confidence < THRESHOLD → pass with FAIR-PP11 warning
    #   confidence < WARN_FLOOR  (or not a poster) → hard reject FAIR-PP10
    #
    # -------------------------------------------------------------------------

    if not is_poster or confidence < SENTRY_WARN_FLOOR:
        # Hard reject — this is very likely not a poster.
        detail = (
            f"Classified as non-poster (is_poster={is_poster}, "
            f"confidence={confidence:.3f}). "
            f"File: {os.path.basename(file_path)}"
        )
        _log(f"REJECTED: {detail}")
        return (None, SENTRY_NOT_A_POSTER, warnings)

    if confidence < SENTRY_CONFIDENCE_THRESHOLD:
        # Borderline — let it through but flag it.
        detail = (
            f"Poster confidence {confidence:.3f} is below threshold "
            f"{SENTRY_CONFIDENCE_THRESHOLD:.2f} (warn floor "
            f"{SENTRY_WARN_FLOOR:.2f}). "
            f"File: {os.path.basename(file_path)}"
        )
        _log(f"WARNING (borderline): {detail}")
        warnings.add(SENTRY_LOW_CONFIDENCE, detail)
        return (result, None, warnings)

    # Clean pass
    _log(f"ACCEPTED: confidence={confidence:.3f}")
    return (result, None, warnings)
