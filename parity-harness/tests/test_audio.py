"""Axis 6 — Audio.

Speech-to-text coverage for native MLXServe, plus a differential cell against
omlx when this machine has a local omlx STT model. The omlx side intentionally
skips loudly rather than downloading models during the harness run.
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from parity import client, config
from parity.report import FAIL, GAP, PASS, REPORT

AXIS = "6. Audio"
FIXTURE = Path(__file__).parent / "fixtures" / "test_speech.wav"
EXPECTED_WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
MIN_EXPECTED_WORDS = 7
MIN_OMLX_AGREEMENT = 0.80


@pytest.fixture(scope="session")
def native_audio(native_pool):
    if not Path(config.NATIVE_BIN).exists():
        pytest.skip(f"missing native binary: {config.NATIVE_BIN}")
    models_dir = Path(config.WHISPERKIT_MODELS).expanduser()
    model_dir = models_dir / config.NATIVE_AUDIO_MODEL
    if not model_dir.exists():
        pytest.skip(f"missing WhisperKit model dir: {model_dir}")
    handle = native_pool.get(config.NATIVE_AUDIO_MODEL)
    if config.NATIVE_AUDIO_MODEL not in handle.discovered_ids(refresh=True):
        pytest.fail(
            f"native did not list speech model {config.NATIVE_AUDIO_MODEL!r}; "
            f"check --whisperkit-models-dir {models_dir}"
        )
    return handle


def _words(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", lowered)


def _transcript_ok(text: str) -> bool:
    words = _words(text)
    counts = Counter(words)
    matched = 0
    for word in EXPECTED_WORDS:
        if counts[word] > 0:
            matched += 1
            counts[word] -= 1
    return matched >= MIN_EXPECTED_WORDS


def _word_agreement(left: str, right: str) -> float:
    left_counts = Counter(_words(left))
    right_counts = Counter(_words(right))
    denominator = max(sum(left_counts.values()), sum(right_counts.values()), 1)
    overlap = sum((left_counts & right_counts).values())
    return overlap / denominator


def _transcribe(server, model_id: str, timeout: float = 180.0):
    return client.audio_transcription(server, model_id, str(FIXTURE), timeout=timeout)


def _error_message(body) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if "detail" in body:
            return str(body["detail"])
    return ""


def _record_and_assert(cell: str, ok: bool, note: str) -> None:
    REPORT.record(AXIS, cell, PASS if ok else FAIL, note)
    assert ok, note


def test_native_transcription_solo(native_audio):
    res = _transcribe(native_audio, config.NATIVE_AUDIO_MODEL)
    text = (res.body or {}).get("text") if isinstance(res.body, dict) else None
    ok = res.status == 200 and isinstance(text, str) and text.strip() and _transcript_ok(text)
    _record_and_assert(
        "native solo transcription",
        ok,
        f"HTTP {res.status}; text={text!r}; raw={res.raw[:160]!r}",
    )


def test_native_transcription_4_way_concurrency(native_audio):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: _transcribe(native_audio, config.NATIVE_AUDIO_MODEL), range(4))
        )

    texts = [
        (res.body or {}).get("text") if isinstance(res.body, dict) else None
        for res in results
    ]
    statuses_ok = all(res.status == 200 for res in results)
    texts_ok = all(isinstance(text, str) and _transcript_ok(text) for text in texts)
    identical = len(set(texts)) == 1
    ok = statuses_ok and texts_ok and identical
    _record_and_assert(
        "native 4-way transcription concurrency",
        ok,
        f"statuses={[res.status for res in results]}; texts={texts!r}",
    )


def test_native_audio_error_semantics(native_audio):
    unknown = _transcribe(native_audio, "model-that-does-not-exist", timeout=60)
    unknown_message = _error_message(unknown.body)
    unknown_ok = (
        unknown.status == 404
        and "available" in unknown_message.lower()
        and config.NATIVE_AUDIO_MODEL in unknown_message
    )
    REPORT.record(
        AXIS,
        "native unknown audio model",
        PASS if unknown_ok else FAIL,
        f"HTTP {unknown.status}; message={unknown_message!r}",
    )
    assert unknown_ok, f"unexpected unknown-model response: {unknown.status} {unknown.raw[:200]!r}"

    garbage = client.audio_transcription_bytes(
        native_audio,
        config.NATIVE_AUDIO_MODEL,
        b"not a wav file",
        timeout=60,
    )
    garbage_error = _error_message(garbage.body)
    garbage_ok = (
        400 <= garbage.status < 600
        and isinstance(garbage.body, dict)
        and (isinstance(garbage.body.get("error"), dict) or "detail" in garbage.body)
    )
    health = requests.get(
        f"{native_audio.base_url}/v1/models", headers=native_audio.headers(), timeout=10
    )
    healthy_after = health.status_code == 200
    ok = garbage_ok and healthy_after
    _record_and_assert(
        "native garbage audio error + health",
        ok,
        f"garbage HTTP {garbage.status}; message={garbage_error!r}; "
        f"post-error /v1/models HTTP {health.status_code}",
    )


def _local_omlx_audio_candidates() -> list[str]:
    candidates: list[str] = []
    if os.environ.get("PARITY_OMLX_AUDIO_MODEL"):
        candidates.append(config.OMLX_AUDIO_MODEL)

    root = Path(config.MODEL_STORE).expanduser()
    if root.is_dir():
        for model_dir in sorted(root.glob("*/*")) + sorted(root.glob("*")):
            if not model_dir.is_dir() or not (model_dir / "config.json").exists():
                continue
            name = model_dir.name
            rel = str(model_dir.relative_to(root))
            haystack = f"{rel} {name}".lower()
            if any(token in haystack for token in ("whisper", "asr", "stt", "vibevoice")):
                candidates.extend([name, rel])

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _discover_omlx_audio_candidates(omlx_server) -> list[str]:
    ids = omlx_server.discovered_ids(refresh=True)
    discovered = [
        model_id
        for model_id in sorted(ids)
        if any(token in model_id.lower() for token in ("whisper", "asr", "stt", "vibevoice"))
    ]
    candidates = _local_omlx_audio_candidates() + discovered
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _omlx_transcription_or_skip(omlx_server):
    candidates = _discover_omlx_audio_candidates(omlx_server)
    if not candidates:
        note = (
            f"SKIP: no local omlx STT candidate in {config.MODEL_STORE}; "
            "not downloading a model during harness run"
        )
        REPORT.record(AXIS, "native vs omlx transcription", GAP, note)
        pytest.skip(note)

    notes: list[str] = []
    for model_id in candidates:
        res = _transcribe(omlx_server, model_id, timeout=240)
        if res.status == 200:
            return model_id, res
        notes.append(f"{model_id}: HTTP {res.status} {_error_message(res.body) or res.raw[:80]}")

    note = "SKIP: no local omlx STT candidate could serve transcription; " + "; ".join(notes)
    REPORT.record(AXIS, "native vs omlx transcription", GAP, note)
    pytest.skip(note)


def test_differential_transcription_vs_omlx(native_audio, omlx_server):
    native_res = _transcribe(native_audio, config.NATIVE_AUDIO_MODEL)
    native_text = (native_res.body or {}).get("text") if isinstance(native_res.body, dict) else ""
    assert native_res.status == 200 and isinstance(native_text, str), (
        f"native transcription failed before differential: "
        f"HTTP {native_res.status} {native_res.raw[:160]!r}"
    )

    oml_model, oml_res = _omlx_transcription_or_skip(omlx_server)
    oml_text = (oml_res.body or {}).get("text") if isinstance(oml_res.body, dict) else ""
    agreement = _word_agreement(native_text, oml_text if isinstance(oml_text, str) else "")
    ok = (
        native_res.status == 200
        and oml_res.status == 200
        and isinstance(oml_text, str)
        and agreement >= MIN_OMLX_AGREEMENT
    )
    _record_and_assert(
        "native vs omlx transcription",
        ok,
        f"native_model={config.NATIVE_AUDIO_MODEL}; omlx_model={oml_model}; "
        f"agreement={agreement:.0%}; native={native_text!r}; omlx={oml_text!r}",
    )
