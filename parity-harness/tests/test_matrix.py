"""Axis 3 — Model-architecture matrix (the most important axis).

For each model spanning families, POST a short completion to BOTH servers. A
cell PASSES iff both return 200 + non-empty coherent output. A cell where native
faults but omlx succeeds is a real native gap mapped to the milestone that will
fix it (this is the axis that would have caught the earlier 27B Mamba-hybrid
crash). Native launch/generation faults are caught and recorded, never allowed
to hard-error the run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from parity import client
from parity.config import spec_for
from parity.report import FAIL, GAP, PASS, REPORT, MatrixCell

AXIS = "3. Model-architecture matrix"
PROMPT = [{"role": "user", "content": "Write one short sentence about the ocean."}]


def _coherent(text: str) -> bool:
    stripped = (text or "").strip()
    return len(stripped) >= 3 and any(ch.isalpha() for ch in stripped)


def _probe(server, model_id, messages=None):
    """Return (ok, status, text, note)."""
    try:
        res = client.chat(server, model_id, messages or PROMPT, max_tokens=32, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return False, 0, "", f"request raised: {exc}"
    if res.status != 200 or res.body is None:
        detail = ""
        if isinstance(res.body, dict) and isinstance(res.body.get("error"), dict):
            detail = res.body["error"].get("message", "")
        return False, res.status, "", f"HTTP {res.status}: {detail or res.raw[:140]}"
    choices = res.body.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content") or ""
    # Reasoning models (Harmony channels / think tags) may spend the whole
    # short budget in the reasoning channel, leaving content empty on BOTH
    # servers. Coherent reasoning output still proves the cell's contract
    # (model loads + generates), so fall back to it.
    if not _coherent(text):
        text = message.get("reasoning_content") or ""
    return _coherent(text), res.status, text, ""


def test_architecture_cell(native_pool, omlx_server, model_id):
    spec = spec_for(model_id)

    # --- omlx side (reference) ---
    oml_ok, _oml_status, oml_text, oml_note = _probe(omlx_server, model_id)

    # --- native side (launch or generation may fault) ---
    native = None
    launch_err = None
    try:
        native = native_pool.get(model_id)
    except Exception as exc:  # noqa: BLE001
        launch_err = exc

    if launch_err is not None:
        nat_ok, nat_status, nat_text, nat_note = False, 0, "", f"launch fault: {launch_err}"
    else:
        nat_ok, nat_status, nat_text, nat_note = _probe(native, model_id)

    # --- classify + record structured cell ---
    if nat_ok and oml_ok:
        verdict = PASS
        note = (
            f"both 200+coherent. native[:50]={nat_text.strip()[:50]!r} "
            f"omlx[:50]={oml_text.strip()[:50]!r}"
        )
    elif oml_ok and not nat_ok:
        verdict = GAP
        note = f"NATIVE FAULT: {nat_note} (status {nat_status}); omlx OK"
    elif nat_ok and not oml_ok:
        verdict = GAP
        note = f"omlx failed ({oml_note}); native OK"
    else:
        verdict = FAIL
        note = f"both failed. native={nat_note} omlx={oml_note}"

    REPORT.record(AXIS, f"{spec.family_label} · {model_id}", verdict, note)
    REPORT.record_matrix(
        MatrixCell(
            model_id=model_id,
            family=spec.family,
            family_label=spec.family_label,
            tier=spec.tier,
            native_ok=nat_ok,
            omlx_ok=oml_ok,
            native_note=nat_note if not nat_ok else "",
            omlx_note=oml_note if not oml_ok else "",
            milestone=spec.milestone,
        )
    )

    # The matrix records gaps rather than failing hard — a known native fault is
    # the expected baseline for sliding-window / hybrid / VLM families. We only
    # hard-fail when native is EXPECTED to work (dense/moe) but didn't, or when
    # even omlx (the reference) failed.
    assert oml_ok, f"omlx (reference) failed on {model_id}: {oml_note}"
    if not spec.expect_native_gap:
        assert nat_ok, f"native unexpectedly faulted on {model_id}: {nat_note}"

    # --- concurrency probe (native only — parallel batching is the product) ---
    # 4 in-flight requests: every one must come back coherent, and the request
    # repeating the solo prompt must reproduce the solo greedy output exactly —
    # continuous batching must not corrupt determinism.
    if nat_ok and native is not None:
        _assert_concurrent_consistent(native, model_id, nat_text)


CONCURRENT_PROMPTS = [
    PROMPT,  # index 0 repeats the solo prompt — must match the solo output
    [{"role": "user", "content": "Name three primary colors."}],
    # Keep every prompt wordy: terse models answer "What is 2+2?" with a bare
    # "4", which the coherence heuristic (>=3 chars + a letter) rejects.
    [{"role": "user", "content": "Count from one to five in words."}],
    [{"role": "user", "content": "Say hello in French."}],
]


def _assert_concurrent_consistent(native, model_id, solo_text):
    with ThreadPoolExecutor(max_workers=len(CONCURRENT_PROMPTS)) as pool:
        results = list(
            pool.map(lambda msgs: _probe(native, model_id, msgs), CONCURRENT_PROMPTS)
        )
    for i, (ok, status, _text, note) in enumerate(results):
        assert ok, (
            f"native concurrent request {i} failed on {model_id}: "
            f"status {status} {note}"
        )
    assert results[0][2] == solo_text, (
        f"native batched greedy output diverged from solo on {model_id}: "
        f"solo[:60]={solo_text[:60]!r} batched[:60]={results[0][2][:60]!r}"
    )
