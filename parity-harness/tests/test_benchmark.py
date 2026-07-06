"""Benchmark capture — native vs omlx throughput/latency on one dense model.

Light-touch (a couple of timed streaming runs) so it fits every pass; feeds the
HTML report's Benchmark section. Not a hard parity gate — throughput differs by
design; the numbers are informational.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from parity import client
from parity.config import BENCH_MODEL, spec_for
from parity.report import GAP, PASS, REPORT, BenchRow
from parity.servers import start_native

PROMPT = [{"role": "user", "content": "Write a two sentence note about the tides."}]
MAX_TOKENS = 64
TIMED_RUNS = 2
PREFIX_AXIS = "8. Prefix cache"
PREFIX_WORD_COUNT = 2_600
PREFIX_TTFT_RATIO_LIMIT = 0.40
PREFIX_CORRECTNESS_WORD_COUNT = 900
MEMORY_GUARD_FREE_PERCENT = 40


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


def _free_memory_percent() -> int | None:
    try:
        out = subprocess.check_output(["memory_pressure", "-Q"], text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        if "System-wide memory free percentage:" not in line:
            continue
        try:
            return int(line.rsplit(" ", 1)[-1].rstrip("%"))
        except ValueError:
            return None
    return None


def _prefix_correctness_prompts() -> tuple[list[dict], list[dict]]:
    corpus = " ".join(
        f"deterministic-prefix-fragment-{idx:04d}"
        for idx in range(PREFIX_CORRECTNESS_WORD_COUNT)
    )
    first = (
        "Use this corpus as background material and answer briefly.\n"
        f"{corpus}\n"
        "Warmup request: reply with the city name Paris."
    )
    second = (
        first
        + "\nExtended request: in one short sentence, name France's capital city."
    )
    return (
        [{"role": "user", "content": first}],
        [{"role": "user", "content": second}],
    )


def _signature(result: client.StreamResult) -> tuple[str, str, int | None]:
    return (result.reasoning_text, result.text, result.output_tokens())


def _run_prefix_sequence(log_dir: Path, *, cached: bool) -> client.StreamResult:
    first_prompt, second_prompt = _prefix_correctness_prompts()
    server = start_native(log_dir)
    try:
        if BENCH_MODEL not in server.discovered_ids(refresh=True):
            pytest.skip(f"{BENCH_MODEL} not served by native")
        session = f"parity-prefix-correctness-{uuid.uuid4()}"
        if cached:
            warm = client.stream_chat(
                server,
                BENCH_MODEL,
                first_prompt,
                max_tokens=8,
                extra={"cache_session": session},
                read_timeout=240.0,
            )
            if warm.status != 200 or not warm.saw_done:
                pytest.skip(
                    f"native prefix warmup failed: HTTP {warm.status} err={warm.read_error or '-'}"
                )
            return client.stream_chat(
                server,
                BENCH_MODEL,
                second_prompt,
                max_tokens=8,
                extra={"cache_session": session},
                read_timeout=240.0,
            )
        return client.stream_chat(
            server,
            BENCH_MODEL,
            second_prompt,
            max_tokens=8,
            extra={"cache_session": f"parity-prefix-cold-{uuid.uuid4()}"},
            read_timeout=240.0,
        )
    finally:
        server.stop()


def test_benchmark(bench_native, omlx_server):
    # Ensure the bench model exists in the matrix registry.
    _ = spec_for(BENCH_MODEL)
    REPORT.record_bench(_measure(bench_native))
    REPORT.record_bench(_measure(omlx_server))


def test_prefix_cache_same_session_matches_fresh_output():
    """Fixround correctness cell: a reused session must generate the same
    observable stream as a fresh process with no warm prefix cache."""
    _ = spec_for(BENCH_MODEL)
    free_percent = _free_memory_percent()
    if free_percent is not None and free_percent <= MEMORY_GUARD_FREE_PERCENT:
        pytest.skip(
            f"memory guard: memory_pressure free {free_percent}% <= {MEMORY_GUARD_FREE_PERCENT}%"
        )

    with tempfile.TemporaryDirectory(prefix="prefix-correctness-") as tmp:
        log_dir = Path(tmp)
        cached = _run_prefix_sequence(log_dir, cached=True)
        cold = _run_prefix_sequence(log_dir, cached=False)

    cached_signature = _signature(cached)
    cold_signature = _signature(cold)
    ok = (
        cached.status == 200
        and cold.status == 200
        and cached.saw_done
        and cold.saw_done
        and cached.read_error == ""
        and cold.read_error == ""
        and cached_signature == cold_signature
    )
    note = (
        f"free={free_percent if free_percent is not None else 'unknown'}%; "
        f"cached=(HTTP {cached.status}, tokens={cached.output_tokens()}, "
        f"text={cached.text!r}, reasoning={cached.reasoning_text!r}); "
        f"fresh=(HTTP {cold.status}, tokens={cold.output_tokens()}, "
        f"text={cold.text!r}, reasoning={cold.reasoning_text!r})"
    )
    REPORT.record(
        PREFIX_AXIS,
        f"same-session output matches fresh cache-disabled run · {BENCH_MODEL}",
        PASS if ok else GAP,
        note,
    )
    assert ok, "cached same-session output diverged from fresh cache-disabled output: " + note


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
