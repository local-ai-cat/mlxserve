"""Harness addition 5: sequential small-model perf fleet cell."""

from __future__ import annotations

import requests
import pytest

from parity import client, config
from parity.report import GAP, PASS, REPORT, BenchRow

AXIS = "11. Perf fleet"
PROMPT = [{"role": "user", "content": "Write one concise sentence about local AI."}]
MAX_TOKENS = 24


@pytest.fixture
def fleet_native(native_pool):
    return native_pool.get()


def _decode_tg(result) -> float | None:
    usage = (result.usage_chunk or {}).get("usage") if result.usage_chunk else None
    if usage and usage.get("generation_tokens_per_second"):
        return usage["generation_tokens_per_second"]
    out = result.output_tokens()
    if result.gen_s and out:
        return (out - 1) / result.gen_s if out > 1 else out / result.gen_s
    return None


def _prompt_pp(result) -> float | None:
    usage = (result.usage_chunk or {}).get("usage") if result.usage_chunk else None
    if usage and usage.get("prompt_tokens_per_second"):
        return usage["prompt_tokens_per_second"]
    return None


def _observed_tokens(result) -> int | None:
    return result.output_tokens() or result.content_deltas or result.reasoning_deltas or None


def _measure(server, model_id: str):
    result = client.stream_chat(
        server,
        model_id,
        PROMPT,
        max_tokens=MAX_TOKENS,
        include_usage=True,
        extra={"thinking_budget": 0, "chat_template_kwargs": {"enable_thinking": False}},
        read_timeout=300,
    )
    ttft_ms = result.ttft_s * 1000.0 if result.ttft_s is not None else None
    REPORT.record_bench(
        BenchRow(
            server=f"{server.name} · {model_id}",
            prompt_pp=_prompt_pp(result),
            gen_tg=_decode_tg(result),
            ttft_ms=ttft_ms,
        )
    )
    return result


def _unload(server, model_id: str) -> None:
    try:
        requests.post(
            f"{server.base_url}/v1/models/{model_id}/unload",
            headers=server.headers(),
            timeout=30,
        )
    except requests.RequestException:
        pass


@pytest.mark.parametrize("fleet_model_id", config.PERF_FLEET_MODELS)
def test_small_model_perf_fleet_sequential(fleet_native, omlx_server, fleet_model_id):
    absent = [
        name
        for name, srv in (("native", fleet_native), ("omlx", omlx_server))
        if fleet_model_id not in srv.discovered_ids(refresh=True)
    ]
    if absent:
        pytest.skip(f"perf-fleet model {fleet_model_id!r} not served by {absent}")

    native_res = None
    omlx_res = None
    try:
        native_res = _measure(fleet_native, fleet_model_id)
        omlx_res = _measure(omlx_server, fleet_model_id)
    finally:
        _unload(fleet_native, fleet_model_id)
        _unload(omlx_server, fleet_model_id)

    ok = (
        native_res.status == 200
        and omlx_res.status == 200
        and native_res.saw_done
        and omlx_res.saw_done
        and not native_res.read_error
        and not omlx_res.read_error
        and (_observed_tokens(native_res) or 0) > 0
        and (_observed_tokens(omlx_res) or 0) > 0
    )
    REPORT.record(
        AXIS,
        f"sequential perf fleet · {fleet_model_id}",
        PASS if ok else GAP,
        f"native HTTP {native_res.status} tokens={_observed_tokens(native_res)} "
        f"tg={_decode_tg(native_res)} err={native_res.read_error or '-'}; "
        f"omlx HTTP {omlx_res.status} tokens={_observed_tokens(omlx_res)} "
        f"tg={_decode_tg(omlx_res)} err={omlx_res.read_error or '-'}",
    )
    assert ok
