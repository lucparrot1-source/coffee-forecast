import logging
import os

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

_ALERT_TO = "lucparrot1@gmail.com"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _post_alert(api_key: str, script_name: str, error_text: str) -> None:
    requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
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


def send_pipeline_alert(script_name: str, error_text: str) -> None:
    """Email a failure alert via Resend. Silently skips if key not set."""
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        log.warning("RESEND_API_KEY not set — skipping alert email")
        return
    try:
        _post_alert(api_key, script_name, error_text)
    except Exception:
        log.exception("Failed to send alert email (original error above)")
