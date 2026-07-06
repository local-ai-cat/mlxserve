"""Harness additions 1-2: grammar differential cells and overhead ratios."""

from __future__ import annotations

import json
import re

import pytest

from parity import client
from parity.config import GRAMMAR_MODEL
from parity.report import GAP, PASS, REPORT, GrammarBenchRow

AXIS = "9. Grammar constraints"
PROMPT = "Return exactly the value required by the active output constraint."
JSON_SCHEMA = {
    "name": "parity_json_literal",
    "strict": True,
    "schema": {"type": "string", "const": "PARITY_JSON"},
}
REGEX_PATTERN = "PARITY_REGEX"
GBNF_GRAMMAR = 'root ::= "PARITY_GBNF"'


@pytest.fixture
def grammar_native(native_pool):
    return native_pool.get(GRAMMAR_MODEL)


def _require_model(native, omlx_server) -> None:
    absent = [
        name
        for name, srv in (("native", native), ("omlx", omlx_server))
        if GRAMMAR_MODEL not in srv.discovered_ids(refresh=True)
    ]
    if absent:
        pytest.skip(f"grammar model {GRAMMAR_MODEL!r} not served by {absent}")


def _payload(extra: dict, *, max_tokens: int = 64) -> dict:
    payload = {
        "model": GRAMMAR_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 11,
        "thinking_budget": 0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    payload.update(extra)
    return payload


def _content(result) -> str:
    if not isinstance(result.body, dict):
        return ""
    choices = result.body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _record_diff(cell: str, native_res, omlx_res, validator) -> None:
    native_text = _content(native_res).strip()
    omlx_text = _content(omlx_res).strip()
    native_valid = native_res.status == 200 and validator(native_text)
    omlx_valid = omlx_res.status == 200 and validator(omlx_text)
    ok = native_valid and omlx_valid and native_text == omlx_text
    note = (
        f"native HTTP {native_res.status} output={native_text!r}; "
        f"omlx HTTP {omlx_res.status} output={omlx_text!r}"
    )
    if not ok:
        note += (
            f"; native_raw={native_res.raw[:180]!r}; "
            f"omlx_raw={omlx_res.raw[:180]!r}"
        )
    REPORT.record(
        AXIS,
        cell,
        PASS if ok else GAP,
        note,
    )
    assert ok


def test_json_schema_diff_cell(grammar_native, omlx_server):
    _require_model(grammar_native, omlx_server)
    payload = _payload({"structured_outputs": {"json": JSON_SCHEMA["schema"]}})
    native_res = client.post_json(
        grammar_native, "/v1/chat/completions", payload, timeout=240
    )
    omlx_res = client.post_json(omlx_server, "/v1/chat/completions", payload, timeout=240)

    def validator(text: str) -> bool:
        try:
            return json.loads(text) == "PARITY_JSON"
        except ValueError:
            return False

    _record_diff("json_schema literal output", native_res, omlx_res, validator)


def test_regex_diff_cell(grammar_native, omlx_server):
    _require_model(grammar_native, omlx_server)
    payload = _payload({"structured_outputs": {"regex": REGEX_PATTERN}})
    native_res = client.post_json(
        grammar_native, "/v1/chat/completions", payload, timeout=240
    )
    omlx_res = client.post_json(omlx_server, "/v1/chat/completions", payload, timeout=240)
    _record_diff(
        "regex literal output",
        native_res,
        omlx_res,
        lambda text: re.fullmatch(REGEX_PATTERN, text) is not None,
    )


def test_gbnf_diff_cell(grammar_native, omlx_server):
    _require_model(grammar_native, omlx_server)
    payload = _payload({"guided_grammar": GBNF_GRAMMAR})
    native_res = client.post_json(
        grammar_native, "/v1/chat/completions", payload, timeout=240
    )
    omlx_res = client.post_json(omlx_server, "/v1/chat/completions", payload, timeout=240)
    _record_diff(
        "GBNF literal output",
        native_res,
        omlx_res,
        lambda text: text == "PARITY_GBNF",
    )


SMALL_SCHEMA = {
    "name": "small_object",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "color": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["color", "count"],
        "additionalProperties": False,
    },
}
LARGE_SCHEMA = {
    "name": "large_object",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string"},
            "role": {"type": "string"},
            "status": {"type": "string"},
            "score": {"type": "integer"},
            "rank": {"type": "integer"},
            "active": {"type": "boolean"},
            "verified": {"type": "boolean"},
        },
        "required": [
            "name",
            "city",
            "role",
            "status",
            "score",
            "rank",
            "active",
            "verified",
        ],
        "additionalProperties": False,
    },
}


def _decode_tg(result) -> float | None:
    usage = (result.usage_chunk or {}).get("usage") if result.usage_chunk else None
    if usage and usage.get("generation_tokens_per_second"):
        return usage["generation_tokens_per_second"]
    out = result.output_tokens()
    if result.gen_s and out:
        return (out - 1) / result.gen_s if out > 1 else out / result.gen_s
    return None


def _measure_schema(server, schema: dict) -> tuple[float | None, float | None, bool]:
    messages = [
        {
            "role": "user",
            "content": "Return a compact JSON object about a parity benchmark.",
        }
    ]
    plain = client.stream_chat(
        server,
        GRAMMAR_MODEL,
        messages,
        max_tokens=64,
        include_usage=True,
        extra={"thinking_budget": 0, "chat_template_kwargs": {"enable_thinking": False}},
        read_timeout=240,
    )
    grammar = client.stream_chat(
        server,
        GRAMMAR_MODEL,
        messages,
        max_tokens=64,
        include_usage=True,
        extra={
            "thinking_budget": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": schema},
        },
        read_timeout=240,
    )
    ok = (
        plain.status == 200
        and grammar.status == 200
        and plain.saw_done
        and grammar.saw_done
        and not plain.read_error
        and not grammar.read_error
    )
    return _decode_tg(plain), _decode_tg(grammar), ok


@pytest.mark.parametrize(
    ("schema_name", "schema"), [("small", SMALL_SCHEMA), ("large", LARGE_SCHEMA)]
)
def test_grammar_overhead_bench(grammar_native, omlx_server, schema_name, schema):
    _require_model(grammar_native, omlx_server)
    all_ok = True
    notes: list[str] = []
    for server in (grammar_native, omlx_server):
        plain_tg, grammar_tg, ok = _measure_schema(server, schema)
        ratio = (
            grammar_tg / plain_tg
            if plain_tg is not None and grammar_tg is not None and plain_tg > 0
            else None
        )
        REPORT.record_grammar_bench(
            GrammarBenchRow(
                server=server.name,
                schema=schema_name,
                plain_tg=plain_tg,
                grammar_tg=grammar_tg,
                ratio=ratio,
            )
        )
        all_ok = all_ok and ok
        ratio_note = f"{ratio:.2f}x" if ratio is not None else "n/a"
        notes.append(
            f"{server.name}: plain={plain_tg} grammar={grammar_tg} ratio={ratio_note}"
        )

    REPORT.record(
        AXIS,
        f"grammar overhead bench ({schema_name} schema)",
        PASS if all_ok else GAP,
        "; ".join(notes),
    )
    assert all_ok
