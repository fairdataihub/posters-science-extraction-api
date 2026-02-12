"""
PosterSentry integration for the extraction API.

Wraps the fairdataihub/poster-sentry model to gate the extraction
pipeline: only scientific posters are allowed through.

Usage in the pipeline:
    from poster_sentry_check import sentry_gate

    ok, err = sentry_gate("/tmp/upload.pdf")
    if err is not None:
        mark_job_failed(conn, job_id, err.message)
        return

Model reference:
    https://huggingface.co/fairdataihub/poster-sentry
"""

import logging
import os
import time
from typing import Optional, Tuple

from error_codes import (
    FAIRError,
    SENTRY_NOT_A_POSTER,
    SENTRY_LOW_CONFIDENCE,
    SENTRY_CLASSIFICATION_FAILED,
    SENTRY_MODEL_UNAVAILABLE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum confidence to accept a document as a poster.
# The model's decision boundary is 0.5; we require a higher bar to reduce
# false positives entering the expensive extraction pipeline.
SENTRY_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SENTRY_CONFIDENCE_THRESHOLD", "0.65")
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
        return {"status": "ok", "confidence_threshold": SENTRY_CONFIDENCE_THRESHOLD}
    return {"status": "unavailable", "error": _sentry_load_error or "unknown"}


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------


def sentry_gate(
    file_path: str,
) -> Tuple[Optional[dict], Optional[FAIRError], Optional[str]]:
    """
    Classify a file with PosterSentry before extraction.

    Args:
        file_path: Local path to the downloaded poster file.

    Returns:
        (result_dict, error, detail)
        - On success (poster accepted): (classification_result, None, None)
        - On rejection / error:         (None, FAIRError, detail_string)

    The caller decides what to do with errors based on SENTRY_ALLOW_ON_ERROR.
    """
    if not SENTRY_ENABLED:
        _log("PosterSentry is disabled — skipping classification.")
        return ({"is_poster": True, "confidence": 1.0, "skipped": True}, None, None)

    sentry = load_sentry()

    if sentry is None:
        detail = _sentry_load_error or "PosterSentry model could not be loaded"
        _log(f"Sentry unavailable: {detail}")
        if SENTRY_ALLOW_ON_ERROR:
            _log("SENTRY_ALLOW_ON_ERROR=true — allowing document through.")
            return ({"is_poster": True, "confidence": 0.0, "skipped": True}, None, None)
        return (None, SENTRY_MODEL_UNAVAILABLE, detail)

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
            return ({"is_poster": True, "confidence": 0.0, "skipped": True}, None, None)
        return (None, SENTRY_CLASSIFICATION_FAILED, detail)

    is_poster = result.get("is_poster", False)
    confidence = result.get("confidence", 0.0)

    # Decision: is this a poster?
    if not is_poster:
        detail = (
            f"Classified as non-poster with confidence {confidence:.3f}. "
            f"File: {os.path.basename(file_path)}"
        )
        _log(f"REJECTED: {detail}")
        return (None, SENTRY_NOT_A_POSTER, detail)

    if confidence < SENTRY_CONFIDENCE_THRESHOLD:
        detail = (
            f"Poster confidence {confidence:.3f} is below threshold "
            f"{SENTRY_CONFIDENCE_THRESHOLD:.2f}. "
            f"File: {os.path.basename(file_path)}"
        )
        _log(f"REJECTED (low confidence): {detail}")
        return (None, SENTRY_LOW_CONFIDENCE, detail)

    # Accepted
    _log(f"ACCEPTED: confidence={confidence:.3f}")
    return (result, None, None)
