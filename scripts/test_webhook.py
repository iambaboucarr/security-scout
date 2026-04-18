#!/usr/bin/env python3
"""Send a simulated GitHub repository_advisory webhook to the local server.

Signs the payload with the webhook secret from .env so the HMAC check passes.
Uses a real DHIS2 advisory (GHSA-fj38-585h-hxgj — SQL Injection in Tracker API,
CVSS 8.6) as test data.

Usage:
    uv run python scripts/test_webhook.py
    # Custom URL:
    uv run python scripts/test_webhook.py --url http://127.0.0.1:8000/webhooks/github
    # Use a specific GHSA from the 18 known dhis2-core advisories:
    uv run python scripts/test_webhook.py --ghsa GHSA-3fr2-wvqx-cmr5
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import uuid
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

import httpx

# Add src/ to path so we can read config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Settings

DEFAULT_URL = "http://127.0.0.1:8000/webhooks/github"

# Real advisory data from dhis2/dhis2-core (verified against GitHub API 2026-04-16).
ADVISORIES = {
    "GHSA-fj38-585h-hxgj": {
        "ghsa_id": "GHSA-fj38-585h-hxgj",
        "cve_id": "CVE-2021-32704",
        "summary": "SQL Injection in DHIS2 Tracker API",
        "description": (
            "A SQL injection vulnerability was found in the /api/trackedEntityInstances "
            "endpoint of DHIS2. An authenticated user with access to the Tracker API could "
            "craft a specially formed request parameter that results in arbitrary SQL execution. "
            "Affected versions: 2.34.4, 2.35.2-2.35.4, 2.36.0. Patched in 2.34.5, 2.35.5, 2.36.1."
        ),
        "severity": "high",
        "cvss": {
            "vector_string": "CVSS:3.1/AV:N/AC:H/PR:L/UI:N/S:C/C:H/I:H/A:H",
            "score": 8.6,
        },
        "cwes": [{"cwe_id": "CWE-89", "name": "SQL Injection"}],
        "identifiers": [
            {"type": "GHSA", "value": "GHSA-3fr2-wvqx-cmr5"},
            {"type": "CVE", "value": "CVE-2021-32704"},
        ],
        "vulnerabilities": [
            {
                "package": {"ecosystem": "dhis2", "name": "dhis2-core"},
                "vulnerable_version_range": ">= 2.34.4, < 2.36.1",
                "patched_versions": "2.34.5, 2.35.5, 2.36.1",
            }
        ],
        "published_at": "2021-06-24T07:58:08Z",
        "updated_at": "2021-06-24T07:58:08Z",
    },
    "GHSA-3fr2-wvqx-cmr5": {
        "ghsa_id": "GHSA-3fr2-wvqx-cmr5",
        "cve_id": None,
        "summary": "Unsafe Java Deserialization - Remote Code Execution (RCE)",
        "description": "Unsafe deserialization in DHIS2 leading to RCE.",
        "severity": "critical",
        "cvss": {"vector_string": None, "score": None},
        "cwes": [{"cwe_id": "CWE-502", "name": "Deserialization of Untrusted Data"}],
        "identifiers": [{"type": "GHSA", "value": "GHSA-3fr2-wvqx-cmr5"}],
        "vulnerabilities": [{"package": {"ecosystem": "dhis2", "name": "dhis2-core"}}],
        "published_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    },
}

DEFAULT_GHSA = "GHSA-fj38-585h-hxgj"


def _build_payload(ghsa_id: str) -> dict:
    advisory = ADVISORIES.get(ghsa_id)
    if advisory is None:
        advisory = {
            "ghsa_id": ghsa_id,
            "cve_id": None,
            "summary": f"Test advisory {ghsa_id}",
            "description": "Simulated advisory for webhook testing.",
            "severity": "high",
            "cvss": {"vector_string": None, "score": None},
            "cwes": [],
            "identifiers": [{"type": "GHSA", "value": ghsa_id}],
            "vulnerabilities": [{"package": {"ecosystem": "dhis2", "name": "dhis2-core"}}],
            "published_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

    return {
        "action": "published",
        "repository_advisory": advisory,
        "repository": {
            "id": 16800037,
            "name": "dhis2-core",
            "full_name": "dhis2/dhis2-core",
            "owner": {"login": "dhis2", "id": 6902207},
            "html_url": "https://github.com/dhis2/dhis2-core",
            "description": "DHIS2 core application",
            "private": False,
        },
        "organization": {"login": "dhis2", "id": 6902207},
        "sender": {"login": "security-bot", "id": 1},
    }


def sign_payload(body: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode(), body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send simulated GitHub advisory webhook")
    parser.add_argument("--url", default=DEFAULT_URL, help="Webhook endpoint URL")
    parser.add_argument("--ghsa", default=DEFAULT_GHSA, help="GHSA ID to send")
    args = parser.parse_args()

    settings = Settings()
    secret = settings.github_webhook_secret

    payload = _build_payload(args.ghsa)
    body = json.dumps(payload).encode()
    signature = sign_payload(body, secret)
    delivery_id = str(uuid.uuid4())

    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "repository_advisory",
        "X-GitHub-Delivery": delivery_id,
        "X-Hub-Signature-256": signature,
        "Date": format_datetime(datetime.now(UTC), usegmt=True),
    }

    ghsa_id = payload["repository_advisory"]["ghsa_id"]
    summary = payload["repository_advisory"]["summary"]
    severity = payload["repository_advisory"]["severity"]
    print(f"Sending repository_advisory webhook to {args.url}")
    print(f"  GHSA: {ghsa_id} ({severity})")
    print(f"  Summary: {summary}")
    print(f"  Delivery ID: {delivery_id}")
    print(f"  Signature: {signature[:30]}...")
    print(f"  Secret source: .env (GITHUB_WEBHOOK_SECRET)")
    print()

    resp = httpx.post(args.url, content=body, headers=headers)
    print(f"Response: {resp.status_code}")
    if resp.text:
        print(f"Body: {resp.text}")

    if resp.status_code == 202:
        print("\nWebhook accepted! Advisory enqueued for triage.")
        print("Check the ARQ worker terminal for processing output.")
    elif resp.status_code == 401:
        print("\nHMAC verification failed — check GITHUB_WEBHOOK_SECRET in .env")
    else:
        print(f"\nUnexpected status: {resp.status_code}")


if __name__ == "__main__":
    main()
