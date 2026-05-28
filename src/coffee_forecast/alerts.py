import logging
import os

import requests

log = logging.getLogger(__name__)

_RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
_ALERT_TO = "lucparrot1@gmail.com"


def send_pipeline_alert(script_name: str, error_text: str) -> None:
    """Email a failure alert via Resend. Silently skips if key not set."""
    if not _RESEND_API_KEY:
        log.warning("RESEND_API_KEY not set — skipping alert email")
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {_RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": "onboarding@resend.dev",
                "to": [_ALERT_TO],
                "subject": f"[ALERT] Pipeline failed: {script_name}",
                "html": f"<pre>{error_text}</pre>",
            },
            timeout=10,
        )
    except Exception:
        log.exception("Failed to send alert email (original error above)")
