"""Axis 1 — Schema conformance.

Same JSON keys/types in: the non-stream chat response, the SSE chunk, and the
usage object. We check native conforms structurally, then diff its key set
against omlx and record divergences as GAPs.
"""

from __future__ import annotations

from parity import client, report
from parity.report import GAP, PASS, REPORT

AXIS = "1. Schema conformance"
PROMPT = [{"role": "user", "content": "Reply with a short greeting."}]

# OpenAI-canonical shapes.
TOPLEVEL_KEYS = {"id", "object", "choices", "created", "model", "usage"}
CHOICE_KEYS = {"index", "message", "finish_reason"}
MESSAGE_KEYS = {"role", "content"}
USAGE_BASE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}
# omlx-style timing extension (present in native always; omlx only on stream).
USAGE_TIMING_KEYS = {
    "time_to_first_token",
    "prompt_eval_duration",
    "generation_duration",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
}


def _nonstream(server, model_id):
    return client.chat(server, model_id, PROMPT, max_tokens=16)


def test_nonstream_toplevel_shape(native_server, omlx_server, model_id):
    nat = _nonstream(native_server, model_id)
    oml = _nonstream(omlx_server, model_id)
    assert nat.status == 200 and nat.body is not None
    assert oml.status == 200 and oml.body is not None

    nat_keys = set(nat.body.keys())
    oml_keys = set(oml.body.keys())

    missing = TOPLEVEL_KEYS - nat_keys
    verdict = PASS if not missing else GAP
    REPORT.record(
        AXIS,
        f"non-stream top-level keys · {model_id}",
        verdict,
        f"native has {sorted(nat_keys)}"
        + (f"; MISSING {sorted(missing)}" if missing else " (conforms)"),
    )

    # object type + value check.
    obj_ok = nat.body.get("object") == "chat.completion"
    REPORT.record(
        AXIS,
        f"non-stream object=='chat.completion' · {model_id}",
        PASS if obj_ok else GAP,
        f"native object={nat.body.get('object')!r}",
    )

    # choice + message shape.
    choice = (nat.body.get("choices") or [{}])[0]
    ch_missing = CHOICE_KEYS - set(choice.keys())
    REPORT.record(
        AXIS,
        f"choice keys · {model_id}",
        PASS if not ch_missing else GAP,
        f"native choice keys {sorted(choice.keys())}"
        + (f"; MISSING {sorted(ch_missing)}" if ch_missing else ""),
    )
    msg = choice.get("message") or {}
    msg_missing = MESSAGE_KEYS - set(msg.keys())
    REPORT.record(
        AXIS,
        f"message keys · {model_id}",
        PASS if not msg_missing else GAP,
        f"native message keys {sorted(msg.keys())}",
    )

    # key-set divergence vs omlx.
    only_nat = nat_keys - oml_keys
    only_oml = oml_keys - nat_keys
    same = not only_nat and not only_oml
    REPORT.record(
        AXIS,
        f"top-level key-set native vs omlx · {model_id}",
        PASS if same else GAP,
        "identical key set"
        if same
        else f"native-only {sorted(only_nat)}; omlx-only {sorted(only_oml)}",
    )

    # These are structural expectations we hold native to (hard assertions).
    assert not missing, f"native non-stream missing {missing}"
    assert obj_ok
    assert not ch_missing


def test_usage_object_shape(native_server, omlx_server, model_id):
    nat = _nonstream(native_server, model_id)
    oml = _nonstream(omlx_server, model_id)
    nat_usage = (nat.body or {}).get("usage") or {}
    oml_usage = (oml.body or {}).get("usage") or {}

    for label, usage in (("native", nat_usage), ("omlx", oml_usage)):
        base_missing = USAGE_BASE_KEYS - set(usage.keys())
        REPORT.record(
            AXIS,
            f"usage base keys ({label}) · {model_id}",
            PASS if not base_missing else GAP,
            f"keys {sorted(usage.keys())}"
            + (f"; MISSING base {sorted(base_missing)}" if base_missing else ""),
        )
        timing_present = USAGE_TIMING_KEYS & set(usage.keys())
        has_all_timing = timing_present == USAGE_TIMING_KEYS
        REPORT.record(
            AXIS,
            f"usage timing extension, non-stream ({label}) · {model_id}",
            PASS if has_all_timing else GAP,
            f"timing keys present: {sorted(timing_present)}"
            if timing_present
            else "no timing keys (omlx omits timing on non-stream responses)",
        )

    # Hard: native must carry the base usage triple.
    assert not (USAGE_BASE_KEYS - set(nat_usage.keys()))


def test_sse_chunk_shape(native_server, omlx_server, model_id):
    nat = client.stream_chat(native_server, model_id, PROMPT, max_tokens=12)
    oml = client.stream_chat(omlx_server, model_id, PROMPT, max_tokens=12)
    assert nat.status == 200 and nat.first_chunk is not None
    assert oml.status == 200 and oml.first_chunk is not None

    required = {"object", "choices", "model", "id", "created"}
    # Pick a native chunk that carries a delta (skip role-only edge cases).
    nat_chunk = next(
        (c for c in nat.chunks if (c.get("choices") or [])), nat.first_chunk
    )
    nat_missing = required - set(nat_chunk.keys())
    obj_ok = nat_chunk.get("object") == "chat.completion.chunk"
    REPORT.record(
        AXIS,
        f"SSE chunk keys · {model_id}",
        PASS if not nat_missing and obj_ok else GAP,
        f"native chunk keys {sorted(nat_chunk.keys())}, object={nat_chunk.get('object')!r}"
        + (f"; MISSING {sorted(nat_missing)}" if nat_missing else ""),
    )
    assert obj_ok
    assert not nat_missing
