"""Axis 6 — Endpoint smoke.

Post-M3 native grew several routes beyond /v1/chat/completions. This axis pokes
the cheap, shape-comparable ones — /health, /v1/completions, /v1/embeddings —
against omlx and records whether native matches. Every probe is defensive: a
missing endpoint or a raised request records a GAP/note and never hard-crashes
the run.
"""

from __future__ import annotations

import pytest
import requests

from parity import client
from parity.config import ERROR_MODEL
from parity.report import GAP, PASS, REPORT

AXIS = "6. Endpoint smoke"
MODEL = ERROR_MODEL


@pytest.fixture
def native(native_pool, omlx_server):
    handle = native_pool.get(MODEL)
    absent = [
        name
        for name, srv in (("native", handle), ("omlx", omlx_server))
        if MODEL not in srv.discovered_ids()
    ]
    if absent:
        pytest.skip(f"smoke model {MODEL!r} not served by {absent}")
    return handle


def _get(server, path: str):
    """GET returning (status, json-or-None); (0, None) on transport error."""
    try:
        resp = requests.get(
            f"{server.base_url}{path}", headers=server.headers(), timeout=15
        )
    except requests.RequestException as exc:  # noqa: BLE001
        return 0, None, f"request raised: {exc}"
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body, ""


def _post(server, path: str, payload: str):
    try:
        return client.raw_post(server, path, payload, timeout=180)
    except requests.RequestException as exc:  # noqa: BLE001
        return client.ChatResult(status=0, body=None, raw=f"request raised: {exc}")


def test_health(native, omlx_server):
    """GET /health: both should report 200 + a 'healthy' status object."""
    for label, server in (("native", native), ("omlx", omlx_server)):
        status, body, err = _get(server, "/health")
        healthy = status == 200 and isinstance(body, dict) and body.get("status") == "healthy"
        REPORT.record(
            AXIS,
            f"/health ({label})",
            PASS if healthy else GAP,
            f"HTTP {status}, status={body.get('status') if isinstance(body, dict) else None!r}"
            + (f" ({err})" if err else ""),
        )


def test_completions_shape(native, omlx_server):
    """POST /v1/completions: legacy text-completion shape (choices[].text)."""
    payload = f'{{"model":"{MODEL}","prompt":"The ocean is","max_tokens":8}}'
    for label, server in (("native", native), ("omlx", omlx_server)):
        res = _post(server, "/v1/completions", payload)
        choices = (res.body or {}).get("choices") or []
        text = choices[0].get("text") if choices else None
        obj_ok = (res.body or {}).get("object") == "text_completion"
        ok = res.status == 200 and isinstance(text, str) and len(text) > 0 and obj_ok
        REPORT.record(
            AXIS,
            f"/v1/completions ({label})",
            PASS if ok else GAP,
            f"HTTP {res.status}, object={(res.body or {}).get('object')!r}, "
            f"text[:40]={(text or '')[:40]!r}",
        )


def test_embeddings_shape(native, omlx_server):
    """POST /v1/embeddings with an LLM (no embedding model loaded): both should
    REFUSE gracefully with an error envelope. We compare native's refusal to
    omlx's and note any status divergence — either way it's a shape smoke, not a
    hard gate."""
    payload = f'{{"model":"{MODEL}","input":"hello"}}'
    nat = _post(native, "/v1/embeddings", payload)
    oml = _post(omlx_server, "/v1/embeddings", payload)

    def refuses(res) -> bool:
        err = (res.body or {}).get("error") if isinstance(res.body, dict) else None
        return res.status >= 400 and isinstance(err, dict) and "message" in err

    for label, res in (("native", nat), ("omlx", oml)):
        REPORT.record(
            AXIS,
            f"/v1/embeddings refuses LLM ({label})",
            PASS if refuses(res) else GAP,
            f"HTTP {res.status}; "
            + (
                f"error.type={(res.body or {}).get('error', {}).get('type')!r}"
                if isinstance(res.body, dict) and isinstance(res.body.get("error"), dict)
                else f"body[:80]={res.raw[:80]!r}"
            ),
        )
    same_status = nat.status == oml.status
    REPORT.record(
        AXIS,
        "/v1/embeddings status parity (native vs omlx)",
        PASS if same_status else GAP,
        f"native HTTP {nat.status} vs omlx HTTP {oml.status} "
        "(both refuse an LLM used as an embedding model; native has no embedding "
        "model loaded so it returns 'backend unavailable')",
    )
