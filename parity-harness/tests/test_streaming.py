"""Axis 5 — Streaming framing.

SSE event order, the [DONE] terminator, and stream_options.include_usage
gating. Divergences from omlx are recorded as GAPs.
"""

from __future__ import annotations

from parity import client
from parity.report import GAP, PASS, REPORT

AXIS = "5. Streaming framing"
PROMPT = [{"role": "user", "content": "Count: one two three."}]


def test_done_terminator(native_server, omlx_server, model_id):
    nat = client.stream_chat(native_server, model_id, PROMPT, max_tokens=12)
    oml = client.stream_chat(omlx_server, model_id, PROMPT, max_tokens=12)

    for label, res in (("native", nat), ("omlx", oml)):
        REPORT.record(
            AXIS,
            f"[DONE] terminator emitted ({label}) · {model_id}",
            PASS if res.saw_done else GAP,
            f"saw [DONE]={res.saw_done}, [DONE] was last SSE line={res.done_is_last}",
        )
    assert nat.saw_done, "native did not emit [DONE]"
    assert oml.saw_done, "omlx did not emit [DONE]"


def test_event_order(native_server, omlx_server, model_id):
    """First real chunk should open the message; a terminal chunk carries
    finish_reason before [DONE]."""
    nat = client.stream_chat(native_server, model_id, PROMPT, max_tokens=12)
    oml = client.stream_chat(omlx_server, model_id, PROMPT, max_tokens=12)

    def finish_present(res):
        return any(
            (c.get("choices") or [{}])[0].get("finish_reason")
            for c in res.chunks
            if c.get("choices")
        )

    REPORT.record(
        AXIS,
        f"terminal finish_reason before [DONE] · {model_id}",
        PASS if finish_present(nat) and finish_present(oml) else GAP,
        f"native finish_reason seen={finish_present(nat)}, omlx={finish_present(oml)}",
    )

    # omlx opens with a synthetic model=="keepalive" chunk; native does not.
    def first_model(res):
        return (res.first_chunk or {}).get("model")

    nat_first = first_model(nat)
    oml_first = first_model(oml)
    REPORT.record(
        AXIS,
        f"first-chunk framing · {model_id}",
        GAP if nat_first != oml_first else PASS,
        f"native first-chunk model={nat_first!r}; omlx first-chunk model={oml_first!r} "
        "(omlx sends a 'keepalive' priming chunk)",
    )
    assert finish_present(nat)


def test_include_usage_gating(native_server, omlx_server, model_id):
    """usage terminal chunk present iff stream_options.include_usage=True."""
    for label, server in (("native", native_server), ("omlx", omlx_server)):
        on = client.stream_chat(server, model_id, PROMPT, max_tokens=12, include_usage=True)
        off = client.stream_chat(
            server, model_id, PROMPT, max_tokens=12, include_usage=False
        )
        on_ok = on.usage_chunk is not None
        off_ok = off.usage_chunk is None
        gated = on_ok and off_ok
        REPORT.record(
            AXIS,
            f"include_usage gating ({label}) · {model_id}",
            PASS if gated else GAP,
            f"usage chunk present with include_usage=True: {on_ok}; "
            f"absent when False: {off_ok}",
        )
    # No hard assert on native gating (baseline may differ); omlx is the reference.


def test_socket_close_after_done(native_server, omlx_server, model_id):
    """Whether the server closes the SSE socket after [DONE]. Native sends
    Connection:close but keeps it open (a real framing gap); omlx closes cleanly."""
    nat_closed = client.probe_socket_closes(native_server, model_id, PROMPT)
    oml_closed = client.probe_socket_closes(omlx_server, model_id, PROMPT)
    REPORT.record(
        AXIS,
        f"closes SSE socket after [DONE] (native) · {model_id}",
        PASS if nat_closed else GAP,
        "native closed socket after [DONE]"
        if nat_closed
        else "native advertises Connection:close but leaves the SSE socket OPEN "
        "after [DONE] (a slow client that waits for close would hang)",
    )
    REPORT.record(
        AXIS,
        f"closes SSE socket after [DONE] (omlx) · {model_id}",
        PASS if oml_closed else GAP,
        "omlx closed socket after [DONE]" if oml_closed else "omlx left socket open",
    )
