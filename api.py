#!/usr/bin/env python3
"""
Flask API server for poster extraction.
Accepts file uploads (PDF, images) and returns extracted JSON.
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from poster_extraction import process_poster_file, log, ensure_ollama_available

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
def health():
    """Health check endpoint including Ollama status."""
    checks = {"api": "ok"}

    try:
        # Fast, single-attempt Ollama health check to avoid long blocking retries
        ensure_ollama_available(max_retries=5, retry_delay=1)
        checks["ollama"] = "ok"
        status = "healthy"
        http_status = 200
    except Exception as e:
        checks["ollama"] = f"error: {str(e)}"
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
