#!/usr/bin/env python3
"""
Flask API server for poster extraction.

Polls the database for new ExtractionJob records. When one is found,
downloads the file from Bunny storage, runs extraction, and writes
results to PosterMetadata. No file upload endpoint; the frontend
uploads files to Bunny and creates jobs in the database.
"""

import json
import re
import threading
import tempfile
import os

import config
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from poster2json.extract import log, load_json_model
from feature_flags import features_for_environment
from poster_matcher import match_rows_to_posters
from sheet_mapper import map_sheet
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
            # Health checks must not load a second model while image OCR or
            # extraction owns the GPU. Busy is a healthy worker state.
            if _extraction_lock.acquire(blocking=False):
                try:
                    load_json_model()
                    checks["json_model"] = "ok"
                finally:
                    _extraction_lock.release()
            else:
                checks["json_model"] = "busy"
            print(f"[status] api: health check json_model={checks['json_model']}")
        except Exception as e:
            checks["json_model"] = f"error: {str(e)}"
            print(f"[status] api: health check json_model error: {e}")

        # Determine overall status
        if checks.get("cuda") == "ok" and checks.get("json_model") in {"ok", "busy"}:
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


# Organizer sheets are small; anything bigger is not a poster list.
_SHEET_MAX_BYTES = 10 * 1024 * 1024
_SHEET_EXTENSIONS = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv")


@app.route("/sheets/map", methods=["POST"])
def sheets_map():
    """
    Map an organizer's poster spreadsheet onto our bulk metadata format.

    multipart/form-data:
        sheet       the spreadsheet (.xlsx, .xls, .csv, .tsv). May be sent more
                    than once for an export split over files (EasyChair
                    submission.csv plus author.csv); 10 MB in total.
        file_names  JSON array of the poster file names uploaded to the batch
        mapping     optional JSON object {column header: field or null}, the
                    user's confirmed choices from the confirm screen

    Returns the detected template, the column mapping, the file column, the
    rows in our format (sheet name -> rows keyed by fillKey, linked by
    file_name), unmatched rows and files, and warnings. See docs/SHEET_MAPPING.md.

    Touches no database and no GPU model, like /thumbnails/generate.
    """
    print("[status] api: POST /sheets/map")
    uploads = [u for u in request.files.getlist("sheet") if u and u.filename]
    if not uploads:
        return jsonify({"error": "sheet is required"}), 400
    if not all(u.filename.lower().endswith(_SHEET_EXTENSIONS) for u in uploads):
        return jsonify({"error": "sheet must be .xlsx, .xls, .csv or .tsv"}), 400
    files = []
    total = 0
    for u in uploads:
        data = u.read(_SHEET_MAX_BYTES + 1)
        total += len(data)
        if total > _SHEET_MAX_BYTES:
            return jsonify({"error": "sheet is larger than 10 MB"}), 413
        files.append((data, u.filename))

    try:
        file_names = json.loads(request.form.get("file_names") or "[]")
        overrides = json.loads(request.form.get("mapping") or "null")
    except json.JSONDecodeError:
        return jsonify({"error": "file_names and mapping must be JSON"}), 400
    if not isinstance(file_names, list) or not all(isinstance(n, str) for n in file_names):
        return jsonify({"error": "file_names must be a JSON array of strings"}), 400
    if overrides is not None and not isinstance(overrides, dict):
        return jsonify({"error": "mapping must be a JSON object"}), 400

    try:
        result = map_sheet(files[0][0], files[0][1], file_names, overrides, extra_files=files[1:])
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        print(f"[status] api: sheet mapping error: {e}")
        return jsonify({"error": "Could not read this spreadsheet"}), 422
    return jsonify(result), 200


@app.route("/sheets/match", methods=["POST"])
def sheets_match():
    """
    Match sheet rows that have no file name to extracted posters, by title,
    with the authors as a tiebreaker. Called once posters have extracted.

    JSON body:
        rows     [{row_key, title, authors: [names]}]  (match_rows from /sheets/map)
        posters  [{file_name, title, authors: [names]}] (what extraction read)

    Returns matches with method and confidence ("high" is automatic, "review"
    needs the organizer's check), plus unmatched rows and files. Stateless.
    """
    print("[status] api: POST /sheets/match")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    rows, posters = body.get("rows"), body.get("posters")
    if not isinstance(rows, list) or not isinstance(posters, list):
        return jsonify({"error": "rows and posters must be arrays"}), 400
    if len(rows) > 2000 or len(posters) > 2000:
        return jsonify({"error": "at most 2000 rows and 2000 posters"}), 413
    if not all(isinstance(x, dict) for x in rows + posters):
        return jsonify({"error": "rows and posters must be arrays of objects"}), 400
    return jsonify(match_rows_to_posters(rows, posters)), 200


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
    """Bind databases to trusted environments; validate flags before starting."""
    db_urls = []
    if prod_db_url := config.get_env("PRODUCTION_DATABASE_URL"):
        db_urls.append(("production", prod_db_url))
        log("Production database polling enabled")
    if staging_db_url := config.get_env("STAGING_DATABASE_URL"):
        db_urls.append(("staging", staging_db_url))
    if len({url for _, url in db_urls}) != len(db_urls):
        raise ValueError("Staging and production must use distinct database targets")
    for environment, _ in db_urls:
        features_for_environment(environment)
    return db_urls


def init_background_worker() -> None:
    """Start background worker unless explicitly disabled by env var."""
    enabled = (config.get_env("ENABLE_BACKGROUND_WORKER") or "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        log("Background worker disabled via ENABLE_BACKGROUND_WORKER")
        return
    if int(config.get_env("SINGLE_JOBS_PER_BULK") or "3") < 1:
        raise ValueError("SINGLE_JOBS_PER_BULK must be at least 1")
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
