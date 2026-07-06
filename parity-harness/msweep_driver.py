#!/usr/bin/env python3
"""M-sweep driver — run the parity harness ONE MODEL AT A TIME.

Why this exists: a single pytest session keeps both servers alive across the
whole model farm, so loaded models accumulate (~130GB marched through two
processes on 2026-07-05 and OOM'd the machine). This driver runs one pytest
session per model — fresh servers, clean teardown, memory preflight, wedge
timeout — and appends a verdict row per model so progress is inspectable live.

Usage:
  python3 msweep_driver.py                  # all full-tier models, smallest first
  python3 msweep_driver.py --max-gb 8       # only models up to 8 GB on disk
  python3 msweep_driver.py --models a,b     # explicit subset
  python3 msweep_driver.py --dry-run        # plan only

Env: same PARITY_* overrides as the harness (PARITY_NATIVE_BIN etc.).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
RESULTS = HARNESS_DIR / "msweep-results.md"

# Per-model pytest wall clock cap. A wedged load must not hang the sweep.
MODEL_TIMEOUT_SECONDS = 25 * 60
# Working-set safety: require free_bytes >= RAM_FACTOR * model_size + RAM_FLOOR.
# Both servers load the model (2x), plus activation/KV overhead.
RAM_FACTOR = 2.4
RAM_FLOOR = 12 * 1024**3


def store_root() -> Path:
    return Path(
        os.environ.get(
            "PARITY_MODEL_FARM",
            os.environ.get(
                "PARITY_MODEL_STORE",
                str(Path.home() / "Library/Caches/models"),
            ),
        )
    )


def farm_models() -> list[tuple[str, int]]:
    """(model_id, size_bytes) for every full-tier harness model, smallest first."""
    sys.path.insert(0, str(HARNESS_DIR))
    from parity import config  # noqa: PLC0415

    sized: list[tuple[str, int]] = []
    for spec in config.matrix_models("full"):
        path = find_model_dir(spec.model_id)
        size = dir_size(path) if path else 0
        sized.append((spec.model_id, size))
    return sorted(sized, key=lambda pair: pair[1])


def find_model_dir(model_id: str) -> Path | None:
    root = store_root()
    for org_dir in root.iterdir() if root.exists() else []:
        candidate = org_dir / model_id
        if candidate.is_dir():
            return candidate
    return None


def dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def free_memory_bytes() -> int:
    # memory_pressure -Q is Apple's own availability estimate; it counts
    # droppable file-backed cache that vm_stat's free/inactive/purgeable sum
    # misses (that sum under-reports by ~2x on a warm machine). Fall back to
    # the vm_stat sum if the tool is unavailable.
    try:
        out = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, check=True
        ).stdout
        total = int(re.search(r"The system has (\d+)", out).group(1))
        percent = int(re.search(r"free percentage: (\d+)%", out).group(1))
        return total * percent // 100
    except (subprocess.CalledProcessError, AttributeError, FileNotFoundError):
        pass
    page_size = 16384
    free_pages = 0
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        match = re.match(r"page size of (\d+)", line)
        if match:
            page_size = int(match.group(1))
        for key in ("Pages free", "Pages inactive", "Pages purgeable"):
            if line.startswith(key):
                free_pages += int(re.search(r"(\d+)\.", line).group(1))
    return free_pages * page_size


def kill_strays() -> None:
    for pattern in ("mlxserve-http", "omlx-sidecar-venv"):
        subprocess.run(["pkill", "-f", pattern], capture_output=True, check=False)


def append_row(model: str, size: int, verdict: str, detail: str, seconds: float) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    detail = detail.replace("|", "/").replace("\n", " ")[:220]
    with RESULTS.open("a") as handle:
        handle.write(
            f"| {model} | {size / 1024**3:.1f} GB | {verdict} | {detail} | {seconds:.0f}s | {stamp} |\n"
        )


def ensure_results_header() -> None:
    if RESULTS.exists():
        return
    RESULTS.write_text(
        "# M-sweep results (per-model harness runs)\n\n"
        "Verdicts: PASS = both serve, cells green · GAP = one side faulted (see detail) ·"
        " FAIL = harness cells failed · SKIP = preflight/timeout.\n\n"
        "| model | size | verdict | detail | wall | at |\n"
        "|---|---|---|---|---|---|\n"
    )


def run_model(model_id: str, size: int) -> tuple[str, str, float]:
    free = free_memory_bytes()
    needed = int(RAM_FACTOR * size + RAM_FLOOR)
    if free < needed:
        return (
            "SKIP",
            f"preflight: free {free / 1024**3:.0f}GB < needed {needed / 1024**3:.0f}GB",
            0.0,
        )

    env = os.environ.copy()
    env["PARITY_HARNESS"] = "1"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--tier",
                "full",
                "-k",
                f"architecture_cell and {model_id}",
                "-q",
                "--no-header",
            ],
            cwd=HARNESS_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=MODEL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        kill_strays()
        return ("SKIP", f"wedged: no result in {MODEL_TIMEOUT_SECONDS // 60}min, killed", time.monotonic() - started)
    finally:
        kill_strays()

    elapsed = time.monotonic() - started
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else "no output"
    gap_lines = [line for line in tail if "GAP" in line or "NATIVE FAULT" in line]
    if proc.returncode == 0:
        verdict = "GAP" if gap_lines else "PASS"
        detail = gap_lines[0] if gap_lines else summary
    else:
        verdict = "FAIL"
        detail = summary
    return (verdict, detail, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = farm_models()
    if args.models:
        wanted = {name.strip() for name in args.models.split(",")}
        models = [pair for pair in models if pair[0] in wanted]
    if args.max_gb is not None:
        models = [pair for pair in models if pair[1] <= args.max_gb * 1024**3]

    print(f"M-sweep plan ({len(models)} models, smallest first):")
    for model_id, size in models:
        print(f"  {size / 1024**3:6.1f} GB  {model_id}")
    if args.dry_run:
        return

    ensure_results_header()
    for index, (model_id, size) in enumerate(models, 1):
        print(f"[{index}/{len(models)}] {model_id} …", flush=True)
        verdict, detail, elapsed = run_model(model_id, size)
        append_row(model_id, size, verdict, detail, elapsed)
        print(f"    -> {verdict} ({elapsed:.0f}s) {detail[:120]}", flush=True)
        time.sleep(5)  # let Metal/file caches settle between models

    print(f"Done. Results: {RESULTS}")


if __name__ == "__main__":
    main()
