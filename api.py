#!/usr/bin/env python3
"""
Flask API server for poster extraction.

Polls the database for new ExtractionJob records. When one is found,
downloads the file from Bunny storage, runs extraction, and writes
results to PosterMetadata. No file upload endpoint; the frontend
uploads files to Bunny and creates jobs in the database.
"""

import os
import threading

import config
import torch
from flask import Flask, jsonify
from flask_cors import CORS

from poster2json.extract import log, load_json_model
from job_worker import run_worker_loop, run_one_cycle
from poster_sentry_check import get_sentry_status

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Lock to prevent concurrent model usage (GPU memory is limited)
# Shared between Flask and the background job worker
_extraction_lock = threading.Lock()


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

        # Check PosterSentry model
        sentry_status = get_sentry_status()
        checks["poster_sentry"] = sentry_status.get("status", "unknown")
        if sentry_status.get("confidence_threshold"):
            checks["sentry_threshold"] = sentry_status["confidence_threshold"]
        print(f"[status] api: health check poster_sentry={checks['poster_sentry']}")

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


@app.route("/jobs/check", methods=["POST"])
def jobs_check():
    """
    Trigger one cycle of the job worker: claim and process one uncompleted
    (pending) job if available. Call after submitting a job to start processing
    without waiting for the next poll interval.
    """
    run_one_cycle(_extraction_lock)
    return "", 204


def _start_worker():
    """Run the job worker loop in a daemon thread."""
    print("[status] api: starting background job worker thread")
    t = threading.Thread(target=run_worker_loop, args=(_extraction_lock,), daemon=True)
    t.start()
    log("Background job worker thread started")
    print("[status] api: background job worker thread started")


if __name__ == "__main__":
    print("[status] api: __main__ starting")
    port = int(config.get_env("PORT") or 8000)
    host = config.get_env("HOST") or "0.0.0.0"
    print(f"[status] api: host={host} port={port}")

    log(f"Starting Poster Extraction API on {host}:{port}")
    _start_worker()
    # threaded=False so only one request at a time; worker runs in separate thread
    print(f"[status] api: running Flask app.run(host={host}, port={port})")
    app.run(host=host, port=port, debug=False, threaded=False)
