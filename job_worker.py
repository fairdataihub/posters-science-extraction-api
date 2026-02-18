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
    print(f"[status] {key}: {val}")
    return val


DATABASE_URL = _env("DATABASE_URL")
BUNNY_PRIVATE_STORAGE = _env("BUNNY_PRIVATE_STORAGE")
BUNNY_PRIVATE_STORAGE_KEY = _env("BUNNY_PRIVATE_STORAGE_KEY")
POLL_INTERVAL_SECONDS = int(_env("POLL_INTERVAL_SECONDS") or "30")
STUCK_PROCESSING_MINUTES = int(_env("STUCK_PROCESSING_MINUTES") or "5")


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
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"[status] download_from_bunny: done, wrote {len(resp.content)} bytes")


# --- DB: claim job, update status, save metadata ----------------------------


def get_conn():
    """Return a new DB connection (caller must close)."""
    print("[status] get_conn: opening database connection")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg2.connect(DATABASE_URL)
    print("[status] get_conn: connected")
    return conn


def claim_next_job(conn) -> Optional[dict]:
    """
    Claim the next pending ExtractionJob (status -> processing).
    Returns the job row as dict or None if no job available.
    """
    print("[status] claim_next_job: querying for pending job")
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
    Strips internal keys and renames extraction fields to DB column names
    """
    print("[status] _extraction_to_metadata_row: mapping extraction to metadata row")
    # Keys we never persist
    skip = {"_validation", "validation_warnings", "error", "raw"}
    row = {k: v for k, v in extraction.items() if k not in skip and not k.startswith("_")}

    # Ensure JSON-serializable and Prisma-compatible types
    out = {}
    for k, v in row.items():
        if v is None and k in ("publicationYear", "doi", "language", "version", "domain"):
            out[k] = None
        elif k in ("sizes", "formats") and isinstance(v, list):
            # PosterMetadata sizes/formats are String[]; keep as list for psycopg2
            out[k] = v
        elif isinstance(v, (dict, list)) and not isinstance(v, str):
            out[k] = json.dumps(v)
        else:
            out[k] = v

    # Spread conference object into DB columns if present
    conference = extraction.get("conference")
    if isinstance(conference, dict):
        out["conferenceName"] = conference.get("name")
        out["conferenceLocation"] = conference.get("location")
        out["conferenceUri"] = conference.get("uri")
        out["conferenceIdentifier"] = conference.get("identifier")
        out["conferenceIdentifierType"] = conference.get("identifierType")
        out["conferenceSchemaUri"] = conference.get("schemaUri")
        out["conferenceStartDate"] = conference.get("startDate")
        out["conferenceEndDate"] = conference.get("endDate")
        out["conferenceAcronym"] = conference.get("acronym")
        out["conferenceSeries"] = conference.get("series")
    if "conference" in out:
        del out["conference"]

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
    "rightsIdentifier",
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
        "language", "relatedIdentifiers", "size", "format", "version", "rightsIdentifier",
        "fundingReferences", "conferenceName", "conferenceLocation", "conferenceUri",
        "conferenceIdentifier", "conferenceIdentifierType", "conferenceYear", "conferenceStartDate",
        "conferenceEndDate", "conferenceAcronym", "conferenceSeries", "posterContent",
        "tableCaptions", "imageCaptions", "domain",
        "created", "updated"
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
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
        "rightsIdentifier" = EXCLUDED."rightsIdentifier",
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


def save_poster_metadata(conn, poster_id: int, extraction: dict) -> None:
    """
    Upsert PosterMetadata for the given poster from validated extraction result.
    Uses a static SQL string and bound parameters only (no string composition).
    """
    print(f"[status] save_poster_metadata: poster_id={poster_id}")
    row = _extraction_to_metadata_row(extraction)
    row["posterId"] = poster_id

    # Build values tuple in fixed column order (all columns; missing keys → None)
    values = tuple(row.get(c) for c in _POSTER_METADATA_COLUMNS)

    with conn.cursor() as cur:
        cur.execute(_POSTER_METADATA_UPSERT_SQL, values)
        conn.commit()
    print(f"[status] save_poster_metadata: done for poster_id={poster_id}")


# --- Worker loop ------------------------------------------------------------


def run_one_cycle(extraction_lock) -> bool:
    """
    Poll DB for one job, claim it, download from Bunny, extract, save metadata.
    Uses extraction_lock to serialize with any other extractors (e.g. Flask).
    Returns True if a job was processed, False if no job found.
    """
    print("[status] run_one_cycle: starting")
    if not DATABASE_URL or not BUNNY_PRIVATE_STORAGE or not BUNNY_PRIVATE_STORAGE_KEY:
        log("Job worker: DATABASE_URL, BUNNY_PRIVATE_STORAGE, BUNNY_PRIVATE_STORAGE_KEY required")
        print("[status] run_one_cycle: missing env, aborting")
        return False

    conn = get_conn()
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
            if not extraction_lock.acquire(blocking=True):
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

    return False


def run_worker_loop(extraction_lock, poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
    """Run the poll loop forever. Use in a background thread."""
    log(f"Job worker: starting poll loop (interval={poll_interval}s)")
    print(f"[status] run_worker_loop: started, poll_interval={poll_interval}s")
    cycle = 0
    while True:
        cycle += 1
        print(f"[status] run_worker_loop: cycle {cycle}")
        try:
            run_one_cycle(extraction_lock)
        except Exception as e:
            log(f"Job worker: error in cycle: {e}")
            print(f"[status] run_worker_loop: cycle error: {e}")
            import traceback

            traceback.print_exc()
        print(f"[status] run_worker_loop: sleeping {poll_interval}s")
        time.sleep(poll_interval)
