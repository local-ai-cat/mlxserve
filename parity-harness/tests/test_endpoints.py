"""Axis 7 — Endpoint smoke.

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

AXIS = "7. Endpoint smoke"
MODEL = ERROR_MODEL
EXPECTED_RESPONSES_REASONING_EVENTS = {
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done",
}
LEGACY_RESPONSES_REASONING_EVENTS = {
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
    "response.reasoning_part.added",
    "response.reasoning_part.done",
}


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


def test_messages_count_tokens_exact_shape(native, omlx_server):
    """POST /v1/messages/count_tokens: both should return the Anthropic
    reference shape and agree on exact chat-template token count."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count this short request."}],
    }
    nat = client.post_json(native, "/v1/messages/count_tokens", payload, timeout=180)
    oml = client.post_json(omlx_server, "/v1/messages/count_tokens", payload, timeout=180)

    def count_shape_ok(res) -> bool:
        body = res.body or {}
        return (
            res.status == 200
            and set(body.keys()) == {"input_tokens"}
            and isinstance(body.get("input_tokens"), int)
            and body["input_tokens"] > 0
        )

    shape_ok = count_shape_ok(nat) and count_shape_ok(oml)
    value_ok = shape_ok and nat.body["input_tokens"] == oml.body["input_tokens"]
    REPORT.record(
        AXIS,
        "/v1/messages/count_tokens exact shape/value",
        PASS if value_ok else GAP,
        f"native HTTP {nat.status} body={nat.body}; omlx HTTP {oml.status} body={oml.body}",
    )
    assert value_ok


def test_responses_reasoning_event_names(native, omlx_server):
    """POST /v1/responses stream: reasoning SSE event names should byte-match
    omlx's current summary event names, not legacy reasoning_text names."""
    prompt = "Think briefly about why 2+2 equals 4, then answer with one sentence."
    stop_after = {"response.reasoning_summary_part.done"}
    nat = client.stream_responses(
        native, MODEL, prompt, max_output_tokens=96, stop_after_events=stop_after
    )
    oml = client.stream_responses(
        omlx_server, MODEL, prompt, max_output_tokens=96, stop_after_events=stop_after
    )

    nat_names = nat.names()
    oml_names = oml.names()
    nat_reasoning = {name for name in nat_names if "reasoning" in name}
    oml_reasoning = {name for name in oml_names if "reasoning" in name}
    legacy = (nat_reasoning | oml_reasoning) & LEGACY_RESPONSES_REASONING_EVENTS
    ok = (
        nat.status == 200
        and oml.status == 200
        and not nat.read_error
        and not oml.read_error
        and not legacy
        and EXPECTED_RESPONSES_REASONING_EVENTS <= nat_reasoning
        and EXPECTED_RESPONSES_REASONING_EVENTS <= oml_reasoning
    )
    REPORT.record(
        AXIS,
        "/v1/responses reasoning SSE event names",
        PASS if ok else GAP,
        f"native reasoning={sorted(nat_reasoning)}; omlx reasoning={sorted(oml_reasoning)}; "
        f"legacy={sorted(legacy)}; status native/omlx={nat.status}/{oml.status}; "
        f"errors={nat.read_error or '-'} / {oml.read_error or '-'}",
    )
    assert ok
