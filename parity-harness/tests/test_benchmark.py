"""Benchmark capture — native vs omlx throughput/latency on one dense model.

Light-touch (a couple of timed streaming runs) so it fits every pass; feeds the
HTML report's Benchmark section. Not a hard parity gate — throughput differs by
design; the numbers are informational.
"""

from __future__ import annotations

import uuid

import pytest

from parity import client
from parity.config import BENCH_MODEL, spec_for
from parity.report import GAP, PASS, REPORT, BenchRow

PROMPT = [{"role": "user", "content": "Write a two sentence note about the tides."}]
MAX_TOKENS = 64
TIMED_RUNS = 2
PREFIX_AXIS = "8. Prefix cache"
PREFIX_WORD_COUNT = 2_600
PREFIX_TTFT_RATIO_LIMIT = 0.40


@pytest.fixture
def bench_native(native_pool):
    return native_pool.get(BENCH_MODEL)


def _measure(server) -> BenchRow:
    # One warmup (weights hot / cache primed), then timed runs.
    client.stream_chat(server, BENCH_MODEL, PROMPT, max_tokens=16)
    ttfts: list[float] = []
    tgs: list[float] = []
    pps: list[float] = []
    for _ in range(TIMED_RUNS):
        res = client.stream_chat(server, BENCH_MODEL, PROMPT, max_tokens=MAX_TOKENS)
        out = res.output_tokens()
        if res.ttft_s is not None:
            ttfts.append(res.ttft_s * 1000.0)
        if res.gen_s and out:
            tgs.append((out - 1) / res.gen_s if out > 1 else out / res.gen_s)
        usage = (res.usage_chunk or {}).get("usage") if res.usage_chunk else None
        if usage and usage.get("prompt_tokens_per_second"):
            pps.append(usage["prompt_tokens_per_second"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return BenchRow(
        server=server.name,
        prompt_pp=avg(pps),
        gen_tg=avg(tgs),
        ttft_ms=avg(ttfts),
    )


def test_benchmark(bench_native, omlx_server):
    # Ensure the bench model exists in the matrix registry.
    _ = spec_for(BENCH_MODEL)
    REPORT.record_bench(_measure(bench_native))
    REPORT.record_bench(_measure(omlx_server))


def test_prefix_cache_same_session_ttft_delta(bench_native):
    """M10a harness cell: second same-session extended prompt should skip
    re-prefilling the long shared prefix."""
    _ = spec_for(BENCH_MODEL)
    if BENCH_MODEL not in bench_native.discovered_ids(refresh=True):
        pytest.skip(f"{BENCH_MODEL} not served by native")

    # Warm the model so the first measured request captures prompt prefill cost,
    # not lazy model load.
    warm = client.stream_chat(
        bench_native,
        BENCH_MODEL,
        [{"role": "user", "content": "Reply with OK."}],
        max_tokens=2,
        include_usage=False,
        read_timeout=120.0,
    )
    if warm.status != 200:
        pytest.skip(f"native warmup returned HTTP {warm.status}")

    corpus = " ".join(f"cacheable-fragment-{idx:04d}" for idx in range(PREFIX_WORD_COUNT))
    base = (
        "Use this corpus as reference material. Reply with exactly OK after reading it.\n"
        f"{corpus}"
    )
    extended = base + "\nAdditional instruction: answer with exactly OK."
    session = f"parity-prefix-{uuid.uuid4()}"

    first = client.stream_chat(
        bench_native,
        BENCH_MODEL,
        [{"role": "user", "content": base}],
        max_tokens=2,
        include_usage=False,
        extra={"cache_session": session},
        read_timeout=240.0,
    )
    second = client.stream_chat(
        bench_native,
        BENCH_MODEL,
        [{"role": "user", "content": extended}],
        max_tokens=2,
        include_usage=False,
        extra={"cache_session": session},
        read_timeout=240.0,
    )

    first_ms = first.ttft_s * 1000.0 if first.ttft_s is not None else None
    second_ms = second.ttft_s * 1000.0 if second.ttft_s is not None else None
    ratio = (
        second.ttft_s / first.ttft_s
        if first.ttft_s is not None and second.ttft_s is not None and first.ttft_s > 0
        else None
    )
    ok = (
        first.status == 200
        and second.status == 200
        and first.saw_done
        and second.saw_done
        and ratio is not None
        and ratio < PREFIX_TTFT_RATIO_LIMIT
    )
    if ratio is not None and first_ms is not None and second_ms is not None:
        note = (
            f"words={PREFIX_WORD_COUNT}; first={first_ms:.1f}ms "
            f"second={second_ms:.1f}ms ratio={ratio:.3f} "
            f"limit<{PREFIX_TTFT_RATIO_LIMIT}"
        )
    else:
        note = (
            f"first HTTP {first.status} ttft={first_ms}; "
            f"second HTTP {second.status} ttft={second_ms}"
        )
    REPORT.record(
        PREFIX_AXIS,
        f"same-session extended prompt TTFT · {BENCH_MODEL}",
        PASS if ok else GAP,
        note,
    )
    assert ok, (
        "prefix-cache TTFT ratio did not meet gate: "
        f"first={first_ms}ms second={second_ms}ms ratio={ratio}"
    )
