"""Axis 4 — Error semantics.

Unknown-model and malformed-body requests: assert the HTTP status + error-body
shape. Native is expected to DIVERGE from omlx today (that is the baseline gap) —
we record every divergence as a GAP rather than failing hard.
"""

from __future__ import annotations

import pytest

from parity import client
from parity.config import ERROR_MODEL
from parity.report import GAP, PASS, REPORT

AXIS = "4. Error semantics"
ERR_MODEL = ERROR_MODEL
# omlx's structured OpenAI error envelope.
OMLX_ERROR_KEYS = {"message", "type", "param", "code"}


@pytest.fixture
def native(native_pool):
    return native_pool.get(ERR_MODEL)


def _error_obj(body) -> dict | None:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"]
    return None


def _record_case(case: str, native_res, omlx_res, expect_status: int) -> None:
    nat_err = _error_obj(native_res.body)
    oml_err = _error_obj(omlx_res.body)

    # omlx is the reference: it should give the structured error + expected-ish status.
    oml_shape_ok = oml_err is not None and OMLX_ERROR_KEYS <= set(oml_err.keys())
    REPORT.record(
        AXIS,
        f"{case}: omlx status/shape",
        PASS if oml_shape_ok and omlx_res.status == expect_status else GAP,
        f"HTTP {omlx_res.status}, error keys "
        f"{sorted(oml_err.keys()) if oml_err else None}",
    )

    # native: does it match omlx status + full error envelope?
    nat_keys = set(nat_err.keys()) if nat_err else set()
    status_match = native_res.status == omlx_res.status
    shape_match = nat_err is not None and OMLX_ERROR_KEYS <= nat_keys
    verdict = PASS if status_match and shape_match else GAP
    detail = (
        f"native HTTP {native_res.status} (omlx {omlx_res.status}); "
        f"native error keys {sorted(nat_keys) if nat_err else None}"
    )
    if not shape_match and nat_err is not None:
        detail += f"; MISSING {sorted(OMLX_ERROR_KEYS - nat_keys)}"
    elif nat_err is None:
        detail += f"; native body not an error envelope: {native_res.raw[:100]!r}"
    REPORT.record(AXIS, f"{case}: native vs omlx", verdict, detail)


def test_unknown_model(native, omlx_server):
    body = (
        '{"model":"model-that-does-not-exist","messages":'
        '[{"role":"user","content":"hi"}],"max_tokens":5}'
    )
    nat = client.raw_post(native, "/v1/chat/completions", body)
    oml = client.raw_post(omlx_server, "/v1/chat/completions", body)
    _record_case("unknown-model", nat, oml, expect_status=404)
    # Record the headline gap explicitly: native ignores the model field.
    if nat.status == 200:
        REPORT.record(
            AXIS,
            "unknown-model: native validates model field",
            GAP,
            "native returns HTTP 200 and serves its launch-pinned model for an "
            "unknown model id (does NOT validate `model`); omlx returns 404.",
        )


def test_malformed_json(native, omlx_server):
    body = "{not valid json"
    nat = client.raw_post(native, "/v1/chat/completions", body)
    oml = client.raw_post(omlx_server, "/v1/chat/completions", body)
    _record_case("malformed-json", nat, oml, expect_status=422)


def test_missing_messages(native, omlx_server):
    body = f'{{"model":"{ERR_MODEL}","max_tokens":5}}'
    nat = client.raw_post(native, "/v1/chat/completions", body)
    oml = client.raw_post(omlx_server, "/v1/chat/completions", body)
    _record_case("missing-messages", nat, oml, expect_status=422)


def test_omlx_auth_required(omlx_server):
    """Reference behavior: omlx guards /v1/* behind the API key."""
    import requests

    resp = requests.get(f"{omlx_server.base_url}/v1/models", timeout=10)
    REPORT.record(
        AXIS,
        "auth: omlx requires API key",
        PASS if resp.status_code == 401 else GAP,
        f"omlx no-auth GET /v1/models -> HTTP {resp.status_code}; "
        "native has no auth layer (open loopback).",
    )
