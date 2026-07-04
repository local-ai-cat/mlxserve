"""Benchmark capture — native vs omlx throughput/latency on one dense model.

Light-touch (a couple of timed streaming runs) so it fits every pass; feeds the
HTML report's Benchmark section. Not a hard parity gate — throughput differs by
design; the numbers are informational.
"""

from __future__ import annotations

import pytest

from parity import client
from parity.config import BENCH_MODEL, spec_for
from parity.report import REPORT, BenchRow

PROMPT = [{"role": "user", "content": "Write a two sentence note about the tides."}]
MAX_TOKENS = 64
TIMED_RUNS = 2


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
