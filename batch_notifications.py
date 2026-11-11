"""Email a batch owner once every poster in their extraction batch has finished.

A batch is finished when it still has at least one job and none of its jobs is
incomplete. Jobs reach a terminal state through extraction, thumbnail work,
the stuck-processing cleanup or the abandoned-upload timeout, so the worker
sweeps for finished batches every cycle instead of hooking a single code path.

`ExtractionBatch."completionEmailSentAt"` records the send, so each batch is
emailed at most once. A failed job that the user retries after the email does
not trigger a second email.
"""

import html
from typing import Optional

import requests
from psycopg2.extras import RealDictCursor

import config
from feature_flags import FeatureContext

FEATURE = "batch_completion_email"
RESEND_URL = "https://api.resend.com/emails"
FROM_ADDRESS = "Posters.science <noreply@posters.science>"
MAX_EMAILS_PER_SWEEP = 20

# The website each database belongs to. Override with <ENV>_SITE_URL.
_DEFAULT_SITE_URLS = {
    "production": "https://posters.science",
    "staging": "https://sandbox.posters.science",
    "development": "http://localhost:3000",
}

_missing_key_logged = False


def _log(message: str) -> None:
    print(f"[status] batch_notifications: {message}")


def site_url(environment: str) -> str:
    configured = config.get_env(f"{environment.upper()}_SITE_URL")
    return (configured or _DEFAULT_SITE_URLS.get(environment, "")).rstrip("/")


def has_completion_email_column(conn) -> bool:
    # Checked per connection, like has_batch_columns, so production can stay on
    # an older schema while staging runs the migration.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT EXISTS (
                   SELECT 1 FROM pg_attribute
                   WHERE attrelid = to_regclass('"ExtractionBatch"')
                     AND attname = 'completionEmailSentAt'
                     AND NOT attisdropped)"""
        )
        return cur.fetchone()[0]


def render_email(
    given_name: str, batch_name: str, total: int, completed: int, failed: int,
    dashboard_url: str,
) -> tuple[str, str]:
    """Return (subject, html) for a finished batch."""
    name = html.escape(given_name.strip()) if given_name and given_name.strip() else "there"
    batch = html.escape(batch_name)
    url = html.escape(dashboard_url, quote=True)
    removed = max(0, total - completed - failed)

    lines = [f"<li><strong>{completed}</strong> extracted and ready to review</li>"]
    if failed:
        lines.append(
            f"<li><strong>{failed}</strong> could not be extracted. "
            "You can retry these from the batch.</li>"
        )
    if removed:
        lines.append(f"<li><strong>{removed}</strong> removed before extraction finished</li>")

    subject = "Your poster batch has finished processing - Posters.science"
    body = f"""
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; color: #333;">
  <h2 style="color: #1a1a1a;">Your Batch Has Finished</h2>
  <p>Hi {name},</p>
  <p>We've finished processing every poster in your batch <strong>{batch}</strong>.</p>
  <ul>
    {"".join(lines)}
  </ul>
  <p>Extracted details are drafts. Please review each poster before publishing.</p>
  <div style="text-align: center; margin: 32px 0;">
    <a href="{url}"
       style="background-color: #4F46E5; color: #ffffff; padding: 12px 24px; border-radius: 6px; text-decoration: none; font-weight: bold; display: inline-block;">
      Review My Posters
    </a>
  </div>
  <p style="font-size: 13px; color: #666;">
    If the button doesn't work, copy and paste this link into your browser:<br/>
    <a href="{url}" style="color: #4F46E5; word-break: break-all;">{url}</a>
  </p>
  <hr style="border: none; border-top: 1px solid #eee; margin: 24px 0;" />
  <p style="font-size: 13px; color: #555;">~ Posters.science</p>
</div>
"""
    return subject, body


def send_email(api_key: str, to: str, subject: str, body: str) -> Optional[bool]:
    """Send through Resend.

    Returns True when accepted, False when Resend rejected the message for a
    reason a retry will not fix (for example an invalid address), and None for
    a transient failure that should be retried on a later sweep.
    """
    domain = to.split("@")[-1] if "@" in to else "unknown"
    try:
        resp = requests.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"from": FROM_ADDRESS, "to": [to], "subject": subject, "html": body},
            timeout=20,
        )
    except requests.RequestException as error:
        _log(f"send failed ({type(error).__name__}), recipientDomain={domain}; will retry")
        return None

    if resp.ok:
        email_id = (resp.json() or {}).get("id") if resp.content else None
        _log(f"email accepted by Resend, emailId={email_id}, recipientDomain={domain}")
        return True
    # 429 and 5xx are worth retrying; other 4xx will fail the same way again.
    if resp.status_code == 429 or resp.status_code >= 500:
        _log(f"Resend returned {resp.status_code}, recipientDomain={domain}; will retry")
        return None
    _log(f"Resend rejected email ({resp.status_code}), recipientDomain={domain}; not retrying")
    return False


_FINISHED_BATCH_SQL = """
    SELECT b.id, b.name, b."totalFiles", u."emailAddress", u."givenName",
           count(j.id) FILTER (WHERE j.status = 'completed') AS completed,
           count(j.id) FILTER (WHERE j.status <> 'completed') AS failed
    FROM "ExtractionBatch" b
    JOIN "User" u ON u.id = b."userId"
    JOIN "ExtractionJob" j ON j."batchId" = b.id
    WHERE b."completionEmailSentAt" IS NULL
      AND b.id = (
          SELECT c.id FROM "ExtractionBatch" c
          WHERE c."completionEmailSentAt" IS NULL
            AND EXISTS (SELECT 1 FROM "ExtractionJob" x WHERE x."batchId" = c.id)
            AND NOT EXISTS (
                SELECT 1 FROM "ExtractionJob" x
                WHERE x."batchId" = c.id AND x.completed = false)
          ORDER BY c.created, c.id
          LIMIT 1 FOR UPDATE SKIP LOCKED
      )
    GROUP BY b.id, u.id
"""


def notify_finished_batches(conn, context: FeatureContext) -> int:
    """Email owners of finished batches in this database. Returns emails sent.

    Each batch row is locked while its email is sent and marked in the same
    transaction, so a second worker cannot send a duplicate. A transient send
    failure rolls back and leaves the batch for the next sweep.
    """
    global _missing_key_logged

    if not context.enabled(FEATURE):
        return 0
    if not has_completion_email_column(conn):
        conn.rollback()
        return 0

    api_key = config.get_env("RESEND_API_KEY")
    if not api_key:
        if not _missing_key_logged:
            _log("RESEND_API_KEY is unset; batch completion emails are not sent")
            _missing_key_logged = True
        conn.rollback()
        return 0

    dashboard_url = f"{site_url(context.environment)}/dashboard"
    sent = 0
    for _ in range(MAX_EMAILS_PER_SWEEP):
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_FINISHED_BATCH_SQL)
            batch = cur.fetchone()
            if not batch:
                conn.rollback()
                break

            subject, body = render_email(
                batch["givenName"], batch["name"], batch["totalFiles"],
                batch["completed"], batch["failed"], dashboard_url,
            )
            result = send_email(api_key, batch["emailAddress"], subject, body)
            if result is None:
                conn.rollback()
                break

            cur.execute(
                """UPDATE "ExtractionBatch"
                   SET "completionEmailSentAt" = now(), updated = now()
                   WHERE id = %s""",
                (batch["id"],),
            )
        conn.commit()
        _log(
            f"batch {batch['id']} in {context.environment} "
            f"{'emailed' if result else 'marked without email'}"
        )
        if result:
            sent += 1
    return sent
