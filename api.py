#!/usr/bin/env python3
"""
Flask API server for poster extraction.

Polls the database for new ExtractionJob records. When one is found,
downloads the file from Bunny storage, runs extraction, and writes
results to PosterMetadata. No file upload endpoint; the frontend
uploads files to Bunny and creates jobs in the database.
"""

import re
import threading
import tempfile
import os

import config
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from poster2json.extract import log, load_json_model
from job_worker import (
    run_worker_loop,
    generate_and_upload_thumbnail,
    download_from_bunny,
    worker_wake_event,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Lock to prevent concurrent model usage (GPU memory is limited)
# Shared between Flask and the background job worker
_extraction_lock = threading.Lock()
_worker_start_lock = threading.Lock()
_worker_started = False


@app.route("/", methods=["GET"])
def root():
    """Health check endpoint."""
    print("[status] api: GET /")
    return jsonify({"status": "ok", "service": "Poster Extraction API", "version": "1.0.0"})


@app.route("/health", methods=["GET"])
@app.route("/up", methods=["GET"])
def health():
    """Health check endpoint including model status."""
    print("[status] api: GET /health or /up")
    checks = {"api": "ok"}

    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            checks["cuda"] = "ok"
            checks["gpu"] = torch.cuda.get_device_name(0)
            print("[status] api: health check cuda=ok")
        else:
            checks["cuda"] = "unavailable"
            print("[status] api: health check cuda=unavailable")

        # Try loading the JSON model (will be cached after first load)
        try:
            load_json_model()
            checks["json_model"] = "ok"
            print("[status] api: health check json_model=ok")
        except Exception as e:
            checks["json_model"] = f"error: {str(e)}"
            print(f"[status] api: health check json_model error: {e}")

        # Determine overall status
        if checks.get("cuda") == "ok" and checks.get("json_model") == "ok":
            status = "healthy"
            http_status = 200
            print("[status] api: health status=healthy")
        else:
            status = "degraded"
            http_status = 200  # Still return 200 if API is running
            print("[status] api: health status=degraded")
    except Exception as e:
        checks["error"] = str(e)
        status = "unhealthy"
        http_status = 503
        print(f"[status] api: health status=unhealthy error={e}")

    return jsonify({"status": status, "checks": checks}), http_status


@app.route("/thumbnails/generate", methods=["POST"])
def thumbnails_generate():
    """
    Generate and upload a thumbnail for a poster file (PDF or image) already in Bunny storage.

    Body (JSON):
        {
          "file_path": "posters/<env>/<uid>[/version-<sequence>-<id>]/filename.pdf"
          // required; also accepts "pdf_path" for backwards compatibility
        }

    Returns:
        { "thumbnail_path": "thumbnails/<env>/<uid>/image.jpeg" }

    Deliberately touches no database. A poster id on its own cannot say which
    environment's database it belongs to, and the staging and production id
    spaces overlap, so the caller writes the returned URL to its own database.
    """
    # posters/<env>/<uid>[/version-<sequence>-<id>]/<filename>.<ext>
    # <env>      — lowercase letters only (e.g. "p", "staging", "production")
    # <uid>      — alphanumeric + hyphens/underscores, 8-40 chars
    # <filename> — no path traversal; must use a supported poster extension
    _FILE_PATH_RE = re.compile(
        r"^posters/[a-z]+/[a-zA-Z0-9_-]{8,40}/"
        r"(?:version-\d+-[a-zA-Z0-9_-]{8,40}/)?"
        r"[^/]+\.(pdf|jpg|jpeg|png)$"
    )

    print("[status] api: POST /thumbnails/generate")
    body = request.get_json(silent=True) or {}
    file_path = (body.get("file_path") or body.get("pdf_path") or "").strip()
    if not file_path:
        return jsonify({"error": "file_path is required"}), 400
    if not _FILE_PATH_RE.match(file_path):
        message = (
            "file_path must match posters/<env>/<uid>"
            "[/version-<sequence>-<id>]/<filename>."
            "(pdf|jpg|jpeg|png)"
        )
        return jsonify({"error": message}), 400

    suffix = os.path.splitext(file_path)[-1].lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    try:
        download_from_bunny(file_path, tmp_path)
        thumbnail_path = generate_and_upload_thumbnail(tmp_path, file_path)
    except Exception as e:
        print(f"[status] api: thumbnail generation error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not thumbnail_path:
        return jsonify({"error": "Could not derive thumbnail path from file_path"}), 400

    print(f"[status] api: thumbnail generated at {thumbnail_path}")
    return jsonify({"thumbnail_path": thumbnail_path}), 200


@app.route("/jobs/check", methods=["POST"])
def jobs_check():
    """
    Wake the background job worker so a freshly submitted job is claimed without
    waiting out the poll interval.

    Returns immediately rather than extracting inline: Flask runs single
    threaded, so doing the work here blocks every other request (/health
    included) for the length of an extraction. The worker polls every configured
    database, so waking it covers all environments.
    """
    print("[status] api: POST /jobs/check")
    worker_wake_event.set()
    return "", 204


def _start_worker(db_urls: list):
    """Run the job worker loop in a daemon thread, polling db_urls sequentially."""
    global _worker_started
    with _worker_start_lock:
        if _worker_started:
            return
        _worker_started = True
    labels = [label for label, _ in db_urls]
    print(f"[status] api: starting background job worker thread (dbs={labels})")
    t = threading.Thread(
        target=run_worker_loop,
        kwargs={"extraction_lock": _extraction_lock, "db_urls": db_urls},
        daemon=True,
        name="job-worker",
    )
    t.start()
    log(f"Background job worker thread started (dbs={labels})")
    print(f"[status] api: background job worker thread started (dbs={labels})")


def _compute_db_targets() -> list:
    """Databases the worker polls. Both serve real users and are drained
    round-robin, so the order only decides the first pick of each pass."""
    db_urls = []
    if prod_db_url := config.get_env("PRODUCTION_DATABASE_URL"):
        db_urls.append(("production", prod_db_url))
        log("Production database polling enabled")
    db_urls.append(("staging", config.get_env("STAGING_DATABASE_URL")))
    return db_urls


def init_background_worker() -> None:
    """Start background worker unless explicitly disabled by env var."""
    enabled = (config.get_env("ENABLE_BACKGROUND_WORKER") or "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        log("Background worker disabled via ENABLE_BACKGROUND_WORKER")
        return
    _start_worker(_compute_db_targets())


# Start worker on import so WSGI/gunicorn entrypoints also process jobs.
init_background_worker()


if __name__ == "__main__":
    print("[status] api: __main__ starting")
    port = int(config.get_env("PORT") or 8000)
    host = config.get_env("HOST") or "0.0.0.0"
    print(f"[status] api: host={host} port={port}")

    log(f"Starting Poster Extraction API on {host}:{port}")
    init_background_worker()
    # threaded=False so only one request at a time; worker runs in separate thread
    print(f"[status] api: running Flask app.run(host={host}, port={port})")
    app.run(host=host, port=port, debug=False, threaded=False)
