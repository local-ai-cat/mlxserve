"""Harness addition 3: non-default launch config axis."""

from __future__ import annotations

from pathlib import Path

import pytest

from parity import client, config
from parity.report import GAP, PASS, REPORT
from parity.servers import start_native, start_omlx

AXIS = "10. Config axis"
MODEL = config.ERROR_MODEL
PROMPT = [{"role": "user", "content": "Write one short sentence about parity."}]
ERROR_KEYS = {"message", "type", "param", "code"}
NONDEFAULT_ARGS = ["--max-concurrent-requests", "2"]
CHUNKED_PREFILL_NOTE = (
    "chunked_prefill not exercised: installed native/omlx CLIs expose no "
    "chunked-prefill launch flag"
)


@pytest.fixture(scope="module")
def nondefault_servers(tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("config-axis-logs")
    native = start_native(Path(log_dir), extra_args=NONDEFAULT_ARGS)
    omlx = start_omlx(Path(log_dir), extra_args=NONDEFAULT_ARGS)
    try:
        yield native, omlx
    finally:
        native.stop()
        omlx.stop()


def _error_obj(body) -> dict | None:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"]
    return None


def test_nondefault_config_streaming_and_error_axes(nondefault_servers):
    native, omlx = nondefault_servers
    absent = [
        name
        for name, srv in (("native", native), ("omlx", omlx))
        if MODEL not in srv.discovered_ids(refresh=True)
    ]
    if absent:
        pytest.skip(f"config-axis model {MODEL!r} not served by {absent}")

    nat_stream = client.stream_chat(native, MODEL, PROMPT, max_tokens=12, read_timeout=180)
    oml_stream = client.stream_chat(omlx, MODEL, PROMPT, max_tokens=12, read_timeout=180)
    stream_ok = (
        nat_stream.status == 200
        and oml_stream.status == 200
        and nat_stream.saw_done
        and oml_stream.saw_done
        and nat_stream.output_tokens() is not None
        and oml_stream.output_tokens() is not None
        and not nat_stream.read_error
        and not oml_stream.read_error
    )
    REPORT.record(
        AXIS,
        "streaming under max_concurrent_requests=2",
        PASS if stream_ok else GAP,
        f"native HTTP {nat_stream.status} done={nat_stream.saw_done} "
        f"tokens={nat_stream.output_tokens()} err={nat_stream.read_error or '-'}; "
        f"omlx HTTP {oml_stream.status} done={oml_stream.saw_done} "
        f"tokens={oml_stream.output_tokens()} err={oml_stream.read_error or '-'}; "
        f"{CHUNKED_PREFILL_NOTE}",
    )

    bad_body = '{"model":"model-that-does-not-exist","messages":[{"role":"user","content":"x"}]}'
    nat_error = client.raw_post(native, "/v1/chat/completions", bad_body, timeout=60)
    oml_error = client.raw_post(omlx, "/v1/chat/completions", bad_body, timeout=60)
    nat_obj = _error_obj(nat_error.body)
    oml_obj = _error_obj(oml_error.body)
    native_keys = set(nat_obj or {})
    omlx_keys = set(oml_obj or {})
    error_ok = (
        nat_error.status == 404
        and oml_error.status == 404
        and nat_obj is not None
        and oml_obj is not None
        and ERROR_KEYS <= native_keys
        and ERROR_KEYS <= omlx_keys
    )
    REPORT.record(
        AXIS,
        "unknown-model error under max_concurrent_requests=2",
        PASS if error_ok else GAP,
        f"native HTTP {nat_error.status} keys={sorted(native_keys)}; "
        f"omlx HTTP {oml_error.status} keys={sorted(omlx_keys)}; "
        f"{CHUNKED_PREFILL_NOTE}",
    )

    assert stream_ok and error_ok
