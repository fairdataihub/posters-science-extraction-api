#!/usr/bin/env python3
"""
Job worker for poster extraction.

Polls the database for new ExtractionJob records, downloads the file from
Bunny storage, runs extraction via the poster2json library, and writes results
to PosterMetadata.
"""

import os
import tempfile
import json
import time
from pathlib import Path
from typing import Optional
import contextlib
import pymupdf

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

import config
from poster2json import extract_poster
from poster2json.extract import log
from validation import validate_and_fix_extraction


# --- Config from config.py --------------------------------------------------


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = config.get_env(key) or default
    print(f"[status] {key}: {'set' if val else 'unset'}")
    return val


DATABASE_URL = _env("STAGING_DATABASE_URL")
BUNNY_PRIVATE_STORAGE = _env("BUNNY_PRIVATE_STORAGE")
BUNNY_PRIVATE_STORAGE_KEY = _env("BUNNY_PRIVATE_STORAGE_KEY")
POLL_INTERVAL_SECONDS = int(_env("POLL_INTERVAL_SECONDS") or "30")
STUCK_PROCESSING_MINUTES = int(_env("STUCK_PROCESSING_MINUTES") or "5")
EXTRACTION_LOCK_TIMEOUT_SECONDS = int(_env("EXTRACTION_LOCK_TIMEOUT_SECONDS") or "300")

THUMBNAIL_MAX_WIDTH = 1200  # px - enough for display, keeps file size small
THUMBNAIL_JPEG_QUALITY = 75  # 1-100;


# --- Bunny: download file ---------------------------------------------------


def download_from_bunny(file_path: str, dest_path: str) -> None:
    """
    Download a file from Bunny storage to a local path.

    file_path: Path within the storage zone (e.g. "posters/abc123/file.pdf").
    dest_path: Local path to write the file.
    """
    print(f"[status] download_from_bunny: starting file_path={file_path} -> dest_path={dest_path}")
    path = file_path.lstrip("/")
    url = f"{BUNNY_PRIVATE_STORAGE}/{path}"
    headers = {
        "AccessKey": f"{BUNNY_PRIVATE_STORAGE_KEY}",
        "Content-Type": "application/octet-stream",
    }
    total_bytes = 0
    with requests.get(url, headers=headers, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                total_bytes += len(chunk)
    print(f"[status] download_from_bunny: done, wrote {total_bytes} bytes")


def upload_to_bunny(file_path: str, data: bytes, content_type: str = "image/jpeg") -> None:
    """
    Upload bytes to Bunny storage at the given path.

    file_path: Path within the storage zone (e.g. "thumbnails/<env>/abc123/uuid.jpg").
    data: Raw bytes to upload.
    """
    print(f"[status] upload_to_bunny: uploading {len(data)} bytes to {file_path}")
    path = file_path.lstrip("/")
    url = f"{BUNNY_PRIVATE_STORAGE}/{path}"
    headers = {
        "AccessKey": f"{BUNNY_PRIVATE_STORAGE_KEY}",
        "Content-Type": content_type,
    }
    resp = requests.put(url, headers=headers, data=data, timeout=60)
    resp.raise_for_status()
    print(f"[status] upload_to_bunny: done, status={resp.status_code}")


def generate_thumbnail_bytes(pdf_path: str) -> bytes:
    """
    Render the first page of a PDF as a compressed JPEG and return the raw bytes.

    The pixmap is scaled so the width does not exceed _THUMBNAIL_MAX_WIDTH, then
    encoded at _THUMBNAIL_JPEG_QUALITY to keep the file size small.
    """
    print(f"[status] generate_thumbnail_bytes: opening {pdf_path}")
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[0]
        # Compute scale so width <= _THUMBNAIL_MAX_WIDTH (height scales proportionally)
        natural_width = page.rect.width  # points, at 72 dpi
        scale = min(1.0, THUMBNAIL_MAX_WIDTH / natural_width)
        matrix = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        jpeg_bytes = pix.tobytes("jpeg", jpg_quality=THUMBNAIL_JPEG_QUALITY)
    finally:
        doc.close()
    print(
        f"[status] generate_thumbnail_bytes: rendered {len(jpeg_bytes)} bytes (scale={scale:.2f}, quality={THUMBNAIL_JPEG_QUALITY})"
    )
    return jpeg_bytes


def generate_and_upload_thumbnail(pdf_path: str, file_path: str) -> Optional[str]:
    """
    Generate a JPEG thumbnail from the first page of a PDF and upload it to Bunny.

    Derives the thumbnail storage path from the original file path:
        posters/<environment>/<uid>/filename.pdf  →  thumbnails/<environment>/<uid>/<uuid>.jpg

    Returns the thumbnail storage path on success, None on failure.
    """
    # Strip leading slash and split into parts
    parts = file_path.lstrip("/").split("/")
    # Need at least: ["posters", "<environment>", "<uid>", "filename.pdf"]
    if len(parts) < 4 or parts[0] != "posters":
        print(
            f"[status] generate_and_upload_thumbnail: unexpected file_path format '{file_path}', skipping"
        )
        return None

    # Take everything between the first segment ("posters") and the filename
    sub_path = "/".join(parts[1:-1])  # e.g. "<environment>/rpfcack1xzysi9p8yuzrt0ma"
    thumbnail_path = f"thumbnails/{sub_path}/image.jpeg"

    jpeg_bytes = generate_thumbnail_bytes(pdf_path)
    upload_to_bunny(thumbnail_path, jpeg_bytes)
    print(f"[status] generate_and_upload_thumbnail: uploaded thumbnail to {thumbnail_path}")

    full_thumbnail_path = f"{BUNNY_PRIVATE_STORAGE}/{thumbnail_path}"
    print(f"[status] generate_and_upload_thumbnail: full thumbnail URL {full_thumbnail_path}")

    return full_thumbnail_path


# --- DB: claim job, update status, save metadata ---------------------------ß-


def get_conn(db_url: Optional[str] = None):
    """Return a new DB connection (caller must close)."""
    url = db_url or DATABASE_URL
    print("[status] get_conn: opening database connection")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(url)
    print("[status] get_conn: connected")
    return conn


def claim_next_job(conn) -> Optional[dict]:
    """
    Claim the next pending ExtractionJob (status -> processing).
    Returns the job row as dict or None if no job available.
    """
    print("[status] claim_next_job: querying for pending job")
    job_id = None
    row = None
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Select one pending job and lock it
        cur.execute(
            """
            SELECT id, "posterId", "fileName", "filePath"
            FROM "ExtractionJob"
            WHERE completed = false AND status = 'pending-extraction'
            ORDER BY created
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        """
        )
        row = cur.fetchone()
        if not row:
            print("[status] claim_next_job: no pending job")
            return None
        job_id = row["id"]
        print(f"[status] claim_next_job: claiming job id={job_id}")
        cur.execute(
            """
            UPDATE "ExtractionJob"
            SET status = 'processing', updated = now()
            WHERE id = %s
        """,
            (job_id,),
        )
    conn.commit()
    print(f"[status] claim_next_job: claimed job {job_id}")
    return dict(row)


def fail_stuck_processing_jobs(conn, stuck_minutes: int = STUCK_PROCESSING_MINUTES) -> int:
    """
    Mark ExtractionJobs stuck in 'processing' for longer than stuck_minutes as failed.
    Returns the number of jobs updated.
    """
    error_msg = (
        f"Job stuck in processing for over {stuck_minutes} minutes; marked failed by cleanup"
    )
    print(
        f"[status] fail_stuck_processing_jobs: checking for processing jobs older than {stuck_minutes} min"
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE "ExtractionJob"
            SET status = 'failed', error = %s, completed = true, updated = now()
            WHERE completed = false AND status = 'processing'
              AND updated < now() - (%s::text || ' minutes')::interval
            """,
            (error_msg, stuck_minutes),
        )
        updated = cur.rowcount
        conn.commit()
    if updated:
        print(f"[status] fail_stuck_processing_jobs: marked {updated} stuck job(s) as failed")
    return updated


def mark_job_failed(conn, job_id: str, error: str) -> None:
    print(f"[status] mark_job_failed: job_id={job_id} error={error[:200]}...")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE "ExtractionJob"
            SET status = 'failed', error = %s, completed = true, updated = now()
            WHERE id = %s
        """,
            (error[:10000], job_id),
        )
        conn.commit()
    print(f"[status] mark_job_failed: job {job_id} marked failed")


def mark_job_completed(conn, job_id: str) -> None:
    print(f"[status] mark_job_completed: job_id={job_id}")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE "ExtractionJob"
            SET status = 'completed', completed = true, updated = now()
            WHERE id = %s
        """,
            (job_id,),
        )
        conn.commit()
    print(f"[status] mark_job_completed: job {job_id} marked completed")


def _extraction_to_metadata_row(extraction: dict) -> dict:
    """
    Map extraction result (and validation defaults) to PosterMetadata columns.
    Strips internal keys and renames extraction fields to DB column names.

    Handles field name mapping between poster2json v0.1.3+ output (which uses
    canonical schema names like ``content``, ``researchField``) and the DB
    columns (which use Prisma names like ``posterContent``, ``domain``).
    """
    print("[status] _extraction_to_metadata_row: mapping extraction to metadata row")
    # Keys we never persist
    skip = {"_validation", "validation_warnings", "error", "raw"}
    row = {k: v for k, v in extraction.items() if k not in skip and not k.startswith("_")}

    # Ensure JSON-serializable and Prisma-compatible types
    out = {}
    for k, v in row.items():
        if v is None and k in (
            "publicationYear",
            "doi",
            "language",
            "version",
            "domain",
            "researchField",
        ):
            out[k] = None
        elif k in ("subjects",) and isinstance(v, list):
            # PosterMetadata String[] columns; keep as list for psycopg2 array binding
            out[k] = v
        elif isinstance(v, (dict, list)) and not isinstance(v, str):
            out[k] = json.dumps(v)
        else:
            out[k] = v

    # --- Field name mapping: schema → DB column names ---

    # Bug 1 fix: poster2json v0.1.3+ outputs "content", DB column is "posterContent"
    if "content" in extraction:
        content_val = extraction["content"]
        out["posterContent"] = (
            json.dumps(content_val) if isinstance(content_val, (dict, list)) else content_val
        )
    elif "posterContent" in extraction:
        content_val = extraction["posterContent"]
        out["posterContent"] = (
            json.dumps(content_val) if isinstance(content_val, (dict, list)) else content_val
        )
    out.pop("content", None)

    # Bug 2 fix: poster2json v0.1.3+ outputs "researchField", DB column is "domain"
    if "researchField" in extraction:
        out["domain"] = extraction["researchField"]
    elif "domain" in extraction:
        out["domain"] = extraction["domain"]
    out.pop("researchField", None)

    # Bug 6 fix: extract license SPDX identifier from rightsList array
    if "rightsList" in extraction and isinstance(extraction["rightsList"], list):
        for r in extraction["rightsList"]:
            if isinstance(r, dict) and r.get("rightsIdentifier"):
                out["license"] = r["rightsIdentifier"]
                break
    out.pop("rightsList", None)

    # Bug 7 fix: publisher may be {"name": "..."} object — extract plain string
    if isinstance(out.get("publisher"), str):
        # Already a string (possibly json.dumps'd from the generic loop)
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            parsed = json.loads(out["publisher"])
            if isinstance(parsed, dict):
                name = (parsed.get("name") or "").strip()
                out["publisher"] = name or None
    elif isinstance(extraction.get("publisher"), dict):
        name = (extraction["publisher"].get("name") or "").strip()
        out["publisher"] = name or None

    # if extraction has sizes or formats, convert to single string for DB
    if "sizes" in extraction and isinstance(extraction["sizes"], list):
        if len(extraction["sizes"]) > 1:
            out["size"] = json.dumps(extraction["sizes"])
        elif len(extraction["sizes"]) == 1:
            out["size"] = extraction["sizes"][0]
        else:
            out["size"] = None

    if "formats" in extraction and isinstance(extraction["formats"], list):
        if len(extraction["formats"]) > 1:
            out["format"] = json.dumps(extraction["formats"])
        elif len(extraction["formats"]) == 1:
            out["format"] = extraction["formats"][0]
        else:
            out["format"] = None

    # Bug 4 fix: flatten subjects from [{subject: "kw"}] → ["kw"]
    if "subjects" in extraction and isinstance(extraction["subjects"], list):
        flat = []
        for s in extraction["subjects"]:
            if isinstance(s, dict) and "subject" in s:
                flat.append(s["subject"])
            elif isinstance(s, str):
                flat.append(s)
        out["subjects"] = flat
    elif "subjects" not in out:
        out["subjects"] = []

    # Bug 3 fix: spread conference object into DB columns using full schema key names
    conference = extraction.get("conference")
    if isinstance(conference, dict):
        out["conferenceName"] = conference.get("conferenceName")
        out["conferenceLocation"] = conference.get("conferenceLocation")
        out["conferenceUri"] = conference.get("conferenceUri")
        out["conferenceIdentifier"] = conference.get("conferenceIdentifier")
        out["conferenceIdentifierType"] = conference.get("conferenceIdentifierType")
        out["conferenceStartDate"] = conference.get("conferenceStartDate")
        out["conferenceEndDate"] = conference.get("conferenceEndDate")
        out["conferenceAcronym"] = conference.get("conferenceAcronym")
        out["conferenceSeries"] = conference.get("conferenceSeries")
        # conferenceYear needs int conversion for DB Integer column
        cy = conference.get("conferenceYear")
        if cy is not None:
            try:
                out["conferenceYear"] = int(cy)
            except (ValueError, TypeError):
                out["conferenceYear"] = None
    out.pop("conference", None)

    # Clean up schema-only keys that have no DB column (prevent stale json.dumps values)
    for key in [
        "titles",
        "descriptions",
        "dates",
        "types",
        "ethicsApprovals",
        "alternateIdentifiers",
        "$schema",
        "prefix",
        "suffix",
        "sizes",
        "formats",
    ]:
        out.pop(key, None)

    return out


# PosterMetadata columns we can fill from extraction (match Prisma schema)
_POSTER_METADATA_COLUMNS = [
    "posterId",
    "doi",
    "identifiers",
    "creators",
    "publisher",
    "publicationYear",
    "subjects",
    "language",
    "relatedIdentifiers",
    "size",
    "format",
    "version",
    "license",
    "fundingReferences",
    "conferenceName",
    "conferenceLocation",
    "conferenceUri",
    "conferenceIdentifier",
    "conferenceIdentifierType",
    "conferenceYear",
    "conferenceStartDate",
    "conferenceEndDate",
    "conferenceAcronym",
    "conferenceSeries",
    "posterContent",
    "tableCaptions",
    "imageCaptions",
    "domain",
]

# Static upsert SQL: column names are fixed (no composition). Values are bound via %s.
# "created" and "updated" are set via now() so they are never null.
_POSTER_METADATA_UPSERT_SQL = """
    INSERT INTO "PosterMetadata" (
        "posterId", "doi", "identifiers", "creators", "publisher", "publicationYear", "subjects",
        "language", "relatedIdentifiers", "size", "format", "version", "license",
        "fundingReferences", "conferenceName", "conferenceLocation", "conferenceUri",
        "conferenceIdentifier", "conferenceIdentifierType", "conferenceYear", "conferenceStartDate",
        "conferenceEndDate", "conferenceAcronym", "conferenceSeries", "posterContent",
        "tableCaptions", "imageCaptions", "domain",
        "created", "updated"
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        now(), now()
    )
    ON CONFLICT ("posterId") DO UPDATE SET
        "doi" = EXCLUDED."doi",
        "identifiers" = EXCLUDED."identifiers",
        "creators" = EXCLUDED."creators",
        "publisher" = EXCLUDED."publisher",
        "publicationYear" = EXCLUDED."publicationYear",
        "subjects" = EXCLUDED."subjects",
        "language" = EXCLUDED."language",
        "relatedIdentifiers" = EXCLUDED."relatedIdentifiers",
        "size" = EXCLUDED."size",
        "format" = EXCLUDED."format",
        "version" = EXCLUDED."version",
        "license" = EXCLUDED."license",
        "fundingReferences" = EXCLUDED."fundingReferences",
        "conferenceName" = EXCLUDED."conferenceName",
        "conferenceLocation" = EXCLUDED."conferenceLocation",
        "conferenceUri" = EXCLUDED."conferenceUri",
        "conferenceIdentifier" = EXCLUDED."conferenceIdentifier",
        "conferenceIdentifierType" = EXCLUDED."conferenceIdentifierType",
        "conferenceYear" = EXCLUDED."conferenceYear",
        "conferenceStartDate" = EXCLUDED."conferenceStartDate",
        "conferenceEndDate" = EXCLUDED."conferenceEndDate",
        "conferenceAcronym" = EXCLUDED."conferenceAcronym",
        "conferenceSeries" = EXCLUDED."conferenceSeries",
        "posterContent" = EXCLUDED."posterContent",
        "tableCaptions" = EXCLUDED."tableCaptions",
        "imageCaptions" = EXCLUDED."imageCaptions",
        "domain" = EXCLUDED."domain",
        "updated" = now()
"""


def _title_and_description_from_extraction(extraction: dict) -> tuple[str, str]:
    """
    Derive poster title and description from extraction result.
    Returns (title, description) as non-empty strings when possible; otherwise ("", "").
    """
    title = ""
    if extraction.get("titles") and isinstance(extraction["titles"], list):
        for t in extraction["titles"]:
            if isinstance(t, dict) and t.get("title"):
                title = (t.get("title") or "").strip()
                break

    description = ""
    if extraction.get("descriptions") and isinstance(extraction["descriptions"], list):
        for d in extraction["descriptions"]:
            if (
                isinstance(d, dict)
                and d.get("description")
                and (d.get("descriptionType") or "").strip() == "Abstract"
            ):
                description = (d.get("description") or "").strip()
                break
        if not description:
            for d in extraction["descriptions"]:
                if isinstance(d, dict) and d.get("description"):
                    description = (d.get("description") or "").strip()
                    break
    # Check both new ("content") and old ("posterContent") field names
    poster_content = extraction.get("content") or extraction.get("posterContent")
    if not description and poster_content and isinstance(poster_content, dict):
        sections = poster_content.get("sections") or []
        for s in sections:
            if isinstance(s, dict):
                st = (s.get("sectionTitle") or "").strip()
                if st and "abstract" in st.lower():
                    description = (s.get("sectionContent") or "").strip()
                    break

    return (title or "", description or "")


def update_poster_title_description(conn, poster_id: int, title: str, description: str) -> None:
    """
    Update Poster.title and Poster.description for the given poster.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE "Poster"
            SET title = %s, description = %s, updated = now()
            WHERE id = %s
            """,
            (title, description, poster_id),
        )
        conn.commit()
    print(f"[status] update_poster_title_description: poster_id={poster_id}")


def update_poster_image_url(conn, poster_id: int, image_url: str) -> None:
    """
    Update Poster.imageUrl for the given poster.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE "Poster"
            SET "imageUrl" = %s, updated = now()
            WHERE id = %s
            """,
            (image_url, poster_id),
        )
        conn.commit()
    print(f"[status] update_poster_image_url: poster_id={poster_id} imageUrl={image_url}")


def save_poster_metadata(conn, poster_id: int, extraction: dict) -> None:
    """
    Upsert PosterMetadata for the given poster from validated extraction result.
    Uses a static SQL string and bound parameters only (no string composition).
    Also updates Poster.title and Poster.description from extraction.
    """
    print(f"[status] save_poster_metadata: poster_id={poster_id}")
    row = _extraction_to_metadata_row(extraction)
    row["posterId"] = poster_id

    # Build values tuple in fixed column order (all columns; missing keys → None)
    values = tuple(row.get(c) for c in _POSTER_METADATA_COLUMNS)

    with conn.cursor() as cur:
        cur.execute(_POSTER_METADATA_UPSERT_SQL, values)
        conn.commit()

    title, description = _title_and_description_from_extraction(extraction)
    update_poster_title_description(conn, poster_id, title, description)

    print(f"[status] save_poster_metadata: done for poster_id={poster_id}")


# --- Worker loop ------------------------------------------------------------


def run_one_cycle(extraction_lock, db_url: Optional[str] = None) -> bool:
    """
    Poll DB for one job, claim it, download from Bunny, extract, save metadata.
    Uses extraction_lock to serialize with any other extractors (e.g. Flask).
    db_url overrides DATABASE_URL when polling a second database (e.g. production).
    Returns True if a job was processed, False if no job found.
    """
    print("[status] run_one_cycle: starting")
    url = db_url or DATABASE_URL
    if not url or not BUNNY_PRIVATE_STORAGE or not BUNNY_PRIVATE_STORAGE_KEY:
        log("Job worker: DATABASE_URL, BUNNY_PRIVATE_STORAGE, BUNNY_PRIVATE_STORAGE_KEY required")
        print("[status] run_one_cycle: missing env, aborting")
        return False

    conn = get_conn(url)
    try:
        fail_stuck_processing_jobs(conn)
        job = claim_next_job(conn)
        if not job:
            print("[status] run_one_cycle: no job, cycle done")
            return False

        job_id = job["id"]
        poster_id = job["posterId"]
        file_path = job["filePath"]
        file_name = job["fileName"]

        log(f"Job worker: claimed job {job_id} (posterId={poster_id}, file={file_name})")
        print(
            f"[status] run_one_cycle: processing job_id={job_id} poster_id={poster_id} file={file_name}"
        )

        suffix = Path(file_name).suffix.lower() or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()
        print(f"[status] run_one_cycle: temp file {tmp_path}")

        try:
            download_from_bunny(file_path, tmp_path)
            log(f"Job worker: downloaded to {tmp_path}")

            print("[status] run_one_cycle: acquiring extraction lock")
            acquired = extraction_lock.acquire(timeout=EXTRACTION_LOCK_TIMEOUT_SECONDS)
            if not acquired:
                mark_job_failed(conn, job_id, "Could not acquire extraction lock")
                return True

            try:
                print(f"[status] run_one_cycle: calling extract_poster({tmp_path})")
                result = extract_poster(tmp_path)
                print("[status] run_one_cycle: extract_poster returned")
            finally:
                extraction_lock.release()
                print("[status] run_one_cycle: released extraction lock")

            if "error" in result:
                mark_job_failed(conn, job_id, result["error"])
                log(f"Job worker: job {job_id} failed: {result['error']}")
                return True

            print("[status] run_one_cycle: validating extraction")
            result, _ = validate_and_fix_extraction(result)
            try:
                save_poster_metadata(conn, poster_id, result)
                mark_job_completed(conn, job_id)
                log(f"Job worker: job {job_id} completed, PosterMetadata updated")
                print(f"[status] run_one_cycle: job {job_id} completed successfully")
            except Exception as e:
                mark_job_failed(conn, job_id, f"Failed to save metadata: {e}")
                log(f"Job worker: job {job_id} failed to save metadata: {e}")
                return True

            try:
                thumbnail_path = generate_and_upload_thumbnail(tmp_path, file_path)
                if thumbnail_path:
                    log(f"Job worker: thumbnail uploaded to {thumbnail_path}")
                    update_poster_image_url(conn, poster_id, thumbnail_path)
            except Exception as e:
                log(f"Job worker: thumbnail generation failed (non-fatal): {e}")
                print(f"[status] run_one_cycle: thumbnail generation failed: {e}")

            return True

        except Exception as e:
            print(f"[status] run_one_cycle: exception {e}")
            mark_job_failed(conn, job_id, str(e))
            log(f"Job worker: job {job_id} failed: {e}")
            import traceback

            traceback.print_exc()
            return True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print(f"[status] run_one_cycle: removed temp file {tmp_path}")

    finally:
        conn.close()
        print("[status] run_one_cycle: closed DB connection")


def run_worker_loop(
    extraction_lock, poll_interval: int = POLL_INTERVAL_SECONDS, db_urls: Optional[list] = None
) -> None:
    """Run the poll loop forever. Use in a background thread.
    db_urls is a list of (label, db_url) pairs to poll sequentially each cycle.
    Defaults to [("default", None)] which uses DATABASE_URL.
    """
    targets = db_urls or [("default", None)]
    log(
        f"Job worker: starting poll loop (interval={poll_interval}s, dbs={[l for l, _ in targets]})"
    )
    print(
        f"[status] run_worker_loop: started, poll_interval={poll_interval}s dbs={[l for l, _ in targets]}"
    )
    cycle = 0
    while True:
        cycle += 1
        for label, db_url in targets:
            print(f"[status] run_worker_loop: cycle {cycle} db={label}")
            try:
                run_one_cycle(extraction_lock, db_url=db_url)
            except Exception as e:
                log(f"Job worker: error in cycle (db={label}): {e}")
                print(f"[status] run_worker_loop: cycle error db={label}: {e}")
                import traceback

                traceback.print_exc()
        print(f"[status] run_worker_loop: sleeping {poll_interval}s")
        time.sleep(poll_interval)
