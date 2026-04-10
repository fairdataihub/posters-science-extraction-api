#!/usr/bin/env python3
"""
Flask API server for poster extraction.

Polls the database for new ExtractionJob records. When one is found,
downloads the file from Bunny storage, runs extraction, and writes
results to PosterMetadata. No file upload endpoint; the frontend
uploads files to Bunny and creates jobs in the database.
"""

import threading

import config
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from poster2json.extract import log, load_json_model
from job_worker import run_worker_loop, run_one_cycle, generate_and_upload_thumbnail

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
    Generate and upload a thumbnail for a poster PDF already in Bunny storage.

    Body (JSON):
        { "pdf_path": "posters/<env>/<uid>/filename.pdf" }

    Returns:
        { "thumbnail_path": "thumbnails/<env>/<uid>/image.jpeg" }
    """
    print("[status] api: POST /thumbnails/generate")
    body = request.get_json(silent=True) or {}
    pdf_path = (body.get("pdf_path") or "").strip()
    if not pdf_path:
        return jsonify({"error": "pdf_path is required"}), 400

    import tempfile
    import os
    from job_worker import download_from_bunny

    suffix = os.path.splitext(pdf_path)[-1].lower() or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    try:
        download_from_bunny(pdf_path, tmp_path)
        thumbnail_path = generate_and_upload_thumbnail(tmp_path, pdf_path)
    except Exception as e:
        print(f"[status] api: thumbnail generation error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not thumbnail_path:
        return jsonify({"error": "Could not derive thumbnail path from pdf_path"}), 400

    print(f"[status] api: thumbnail generated at {thumbnail_path}")
    return jsonify({"thumbnail_path": thumbnail_path}), 200


@app.route("/jobs/check", methods=["POST"])
def jobs_check():
    """
    Trigger one cycle of the job worker: claim and process one uncompleted
    (pending) job if available. Call after submitting a job to start processing
    without waiting for the next poll interval.
    """
    run_one_cycle(_extraction_lock)
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
    db_urls = [("staging", None)]
    if prod_db_url := config.get_env("PRODUCTION_DATABASE_URL"):
        db_urls.append(("production", prod_db_url))
        log("Production database polling enabled")
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
