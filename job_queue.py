"""Queue claims, including compatibility with databases before batches."""

from typing import Optional
from psycopg2.extras import RealDictCursor


def has_batch_columns(conn) -> bool:
    # Inspect each connection, rather than cache a negative result across a
    # migration. Production can remain on the old schema during staging rollout.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT count(*) = 3 FROM pg_attribute
               WHERE attrelid = to_regclass('"ExtractionJob"')
                 AND attname IN ('queueClass', 'batchId', 'fileUploaded')
                 AND NOT attisdropped"""
        )
        return cur.fetchone()[0]


def claim_next_job(
    conn, queue_class: str = "single", after_batch_id: Optional[str] = None,
    batch_columns: Optional[bool] = None,
) -> Optional[dict]:
    """Claim one poster, rotating bulk batches by id and preserving FIFO within them."""
    if queue_class not in {"single", "bulk"}:
        raise ValueError(f"Unknown queue class: {queue_class}")
    modern = has_batch_columns(conn) if batch_columns is None else batch_columns
    if not modern and queue_class == "bulk":
        return None

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if modern:
            cur.execute(
                """
                SELECT id, "posterId", "fileName", "filePath", status,
                       "queueClass", "batchId"
                FROM "ExtractionJob"
                WHERE completed = false AND "fileUploaded" = true
                  AND status IN ('pending-extraction', 'pending-thumbnail')
                  AND "queueClass" = %s
                ORDER BY
                  CASE WHEN %s = 'bulk' AND "batchId" > %s THEN 0 ELSE 1 END,
                  CASE WHEN %s = 'bulk' THEN "batchId" END,
                  created, id
                LIMIT 1 FOR UPDATE SKIP LOCKED
                """,
                (queue_class, queue_class, after_batch_id or "", queue_class),
            )
        else:
            cur.execute(
                """
                SELECT id, "posterId", "fileName", "filePath", status
                FROM "ExtractionJob"
                WHERE completed = false
                  AND status IN ('pending-extraction', 'pending-thumbnail')
                ORDER BY created, id
                LIMIT 1 FOR UPDATE SKIP LOCKED
                """
            )
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(
            """UPDATE "ExtractionJob" SET status = 'processing', updated = now()
               WHERE id = %s""", (row["id"],),
        )
    conn.commit()
    return dict(row)


def fail_interrupted_uploads(conn) -> None:
    """Abandoned browser uploads become visible failures, never extraction jobs."""
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE "ExtractionJob"
               SET status = 'failed', completed = true, updated = now(),
                   error = 'Upload interrupted. Select this file again to upload it.'
               WHERE "fileUploaded" = false AND completed = false
                 AND status IN ('pending-upload', 'uploading')
                 AND updated < now() - interval '15 minutes'"""
        )
    conn.commit()
