#!/usr/bin/env python3
"""Stress-test native mlxserve under sustained mixed concurrent load.

Boots the native server on the real model store, then hammers one model with
ramping waves of workers. Each worker loops: random prompt, random max_tokens,
~20% streaming requests, ~5% unknown-model requests (must 404 cleanly). At the
end the run hard-fails if ANY request produced a 5xx / transport error / bad
stream framing, or if the server no longer answers /v1/models. Peak and final
RSS are reported for the operator (memory verdicts stay human — jetsam math is
platform-specific).

Usage:
  python3 stress_driver.py --model Qwen3-0.6B-4bit --seconds 90 --max-workers 16
  PARITY_NATIVE_BIN=... python3 stress_driver.py --model Qwen3.6-27B-4bit --seconds 60 --max-workers 8
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import requests

from parity import servers

PROMPTS = [
    "Write one short sentence about the ocean.",
    "Name three primary colors.",
    "What is 2+2? Answer with a number.",
    "Say hello in French.",
    "List two planets.",
    "Give a one-line definition of gravity.",
    "What rhymes with cat?",
    "Translate 'good morning' to Spanish.",
]

STREAM_RATIO = 0.20
UNKNOWN_MODEL_RATIO = 0.05
RAMP = (2, 4, 8)  # worker counts for the ramp phases; final phase = --max-workers


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.outcomes: Counter[str] = Counter()
        self.failures: list[str] = []
        self.latencies: list[float] = []

    def record(self, outcome: str, latency: float, detail: str = "") -> None:
        with self.lock:
            self.outcomes[outcome] += 1
            self.latencies.append(latency)
            if detail and len(self.failures) < 20:
                self.failures.append(detail)


def one_request(base_url: str, model: str, stats: Stats) -> None:
    unknown = random.random() < UNKNOWN_MODEL_RATIO
    stream = not unknown and random.random() < STREAM_RATIO
    payload = {
        "model": "no-such-model-stress" if unknown else model,
        "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
        "max_tokens": random.choice((16, 32, 64, 96)),
        "temperature": 0.0,
        "stream": stream,
    }
    start = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions", json=payload, timeout=300, stream=stream
        )
        if unknown:
            if resp.status_code == 404:
                stats.record("expected_404", time.monotonic() - start)
            else:
                stats.record(
                    "wrong_status_for_unknown",
                    time.monotonic() - start,
                    f"unknown model got HTTP {resp.status_code}",
                )
            return
        if resp.status_code != 200:
            stats.record(
                "http_error",
                time.monotonic() - start,
                f"HTTP {resp.status_code}: {resp.text[:120]}",
            )
            return
        if stream:
            saw_done = False
            for line in resp.iter_lines():
                if line == b"data: [DONE]":
                    saw_done = True
            if saw_done:
                stats.record("stream_ok", time.monotonic() - start)
            else:
                stats.record(
                    "stream_bad_framing",
                    time.monotonic() - start,
                    "stream ended without [DONE]",
                )
        else:
            body = resp.json()
            text = (body["choices"][0]["message"].get("content") or "") + (
                body["choices"][0]["message"].get("reasoning_content") or ""
            )
            if text.strip():
                stats.record("ok", time.monotonic() - start)
            else:
                stats.record(
                    "empty_output", time.monotonic() - start, "200 with empty output"
                )
    except Exception as exc:  # noqa: BLE001
        stats.record(
            "transport_error", time.monotonic() - start, f"{type(exc).__name__}: {exc}"
        )


def worker(base_url: str, model: str, stats: Stats, stop: threading.Event) -> None:
    while not stop.is_set():
        one_request(base_url, model, stats)


def rss_gb(pid: int) -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) / 1024 / 1024 if out else 0.0


def run_phase(base_url, model, stats, workers, seconds, native_pid, peak):
    stop = threading.Event()
    threads = [
        threading.Thread(target=worker, args=(base_url, model, stats, stop), daemon=True)
        for _ in range(workers)
    ]
    for t in threads:
        t.start()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        peak[0] = max(peak[0], rss_gb(native_pid))
        time.sleep(2)
    stop.set()
    for t in threads:
        t.join(timeout=310)
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seconds", type=int, default=90, help="total run length")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    log_dir = Path(tempfile.mkdtemp(prefix="stress-native-"))
    handle = servers.start_native(log_dir)
    print(f"native up at {handle.base_url}; logs in {log_dir}", flush=True)
    base_url, pid = handle.base_url, handle.proc.pid
    stats = Stats()
    peak = [rss_gb(pid)]
    try:
        # Warm the model with one solo request before load.
        one_request(base_url, args.model, stats)
        phases = [w for w in RAMP if w < args.max_workers] + [args.max_workers]
        per_phase = max(10, args.seconds // len(phases))
        for workers in phases:
            print(f"phase: {workers} workers × {per_phase}s …", flush=True)
            run_phase(base_url, args.model, stats, workers, per_phase, pid, peak)

        health = requests.get(f"{base_url}/v1/models", timeout=30)
        final_rss = rss_gb(pid)
    finally:
        handle.stop()

    lat = sorted(stats.latencies)
    p50 = lat[len(lat) // 2] if lat else 0
    p95 = lat[int(len(lat) * 0.95)] if lat else 0
    print(json.dumps(stats.outcomes, indent=2))
    print(f"requests={sum(stats.outcomes.values())} p50={p50:.2f}s p95={p95:.2f}s")
    print(f"rss peak={peak[0]:.1f}GB final={final_rss:.1f}GB (operator judges)")
    for f in stats.failures:
        print(f"  failure: {f}")

    bad = (
        stats.outcomes["http_error"]
        + stats.outcomes["transport_error"]
        + stats.outcomes["stream_bad_framing"]
        + stats.outcomes["empty_output"]
        + stats.outcomes["wrong_status_for_unknown"]
    )
    assert health.status_code == 200, "server unhealthy after stress"
    assert bad == 0, f"{bad} bad outcomes under stress"
    print("STRESS PASS")


if __name__ == "__main__":
    main()
