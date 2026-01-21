#!/usr/bin/env python3
"""
Flask API server for poster extraction.
Accepts file uploads (PDF, images) and returns extracted JSON.
"""

import os
import tempfile
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from poster_extraction import process_poster_file, log, load_json_model
from validation import validate_and_fix_extraction

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024


def allowed_file(filename):
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def root():
    """Health check endpoint."""
    return jsonify(
        {"status": "ok", "service": "Poster Extraction API", "version": "1.0.0"}
    )


@app.route("/health", methods=["GET"])
@app.route("/up", methods=["GET"])
def health():
    """Health check endpoint including model status."""
    checks = {"api": "ok"}

    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            checks["cuda"] = "ok"
            checks["gpu"] = torch.cuda.get_device_name(0)
        else:
            checks["cuda"] = "unavailable"

        # Try loading the JSON model (will be cached after first load)
        try:
            load_json_model()
            checks["json_model"] = "ok"
        except Exception as e:
            checks["json_model"] = f"error: {str(e)}"

        # Determine overall status
        if checks.get("cuda") == "ok" and checks.get("json_model") == "ok":
            status = "healthy"
            http_status = 200
        else:
            status = "degraded"
            http_status = 200  # Still return 200 if API is running
    except Exception as e:
        checks["error"] = str(e)
        status = "unhealthy"
        http_status = 503

    return jsonify({"status": status, "checks": checks}), http_status


@app.route("/extract", methods=["POST"])
def extract_poster():
    """
    Extract structured JSON from a scientific poster.

    Accepts:
    - PDF files (.pdf)
    - Image files (.jpg, .jpeg, .png)

    Returns:
    - JSON object with extracted poster data
    """
    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Check if file was selected
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "error": f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                }
            ),
            400,
        )

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return (
            jsonify(
                {
                    "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
                }
            ),
            400,
        )

    # Save uploaded file to temporary location
    file_ext = Path(file.filename).suffix.lower()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    tmp_file_path = tmp_file.name

    try:
        # Write uploaded content to temp file
        file.seek(0)  # Reset to beginning
        content = file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(content)

        log(f"Received file: {file.filename} ({file_size} bytes)")

        # Process the poster
        result = process_poster_file(tmp_file_path)

        # Check for errors in result
        if "error" in result:
            return jsonify(result), 500

        # Validate and auto-fix the extraction result
        result, validation_warnings = validate_and_fix_extraction(result)
        if validation_warnings:
            result["validation_warnings"] = validation_warnings

        return jsonify(result)

    except Exception as e:
        log(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Error processing poster: {str(e)}"}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    log(f"Starting Poster Extraction API on {host}:{port}")
    app.run(host=host, port=port, debug=False)
