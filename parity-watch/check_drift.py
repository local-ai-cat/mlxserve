#!/usr/bin/env python3
"""Upstream drift-watch checker (MLXServe <-> omlx parity, Milestone M9).

Re-extracts the live omlx/mlx surface and diffs it against the committed
`omlx-surface.lock.json`. The parity-relevant signal is ADDITIONS:

  * a new (method, path) route      -> a parity gap (native MLXServe may not serve it)
  * a new ChatCompletionRequest field -> a request knob not yet plumbed
  * a new model_type basename       -> a new harness matrix cell to add

Any addition exits NON-ZERO (drift detected). Removals and metadata changes
(omlx version bump, path moves) are reported as informational but do NOT fail
by default -- upstream removing surface is not unmet parity work. Use --strict
to also fail on removals.

Usage:
    python3 check_drift.py                 # diff live surface vs committed lockfile
    python3 check_drift.py --update        # refresh the lockfile to the live surface
    python3 check_drift.py --lock PATH     # diff against a specific lockfile
    python3 check_drift.py --against PATH  # diff committed lock vs a given surface JSON
                                           #   (for simulating drift without re-extracting)
    python3 check_drift.py --json          # machine-readable diff on stdout
    python3 check_drift.py --strict        # also exit non-zero on removals

Exit codes: 0 = no additions (parity maintained); 1 = additions found (parity work);
2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

import extract_surface as ex

LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ex.LOCK_FILENAME)

# ANSI (only when stdout is a tty)
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
RED = lambda s: _c("31", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
BOLD = lambda s: _c("1", s)


def _route_key(r: Dict[str, str]) -> str:
    return f"{r['method']} {r['path']}"


def _route_set(surface: Dict) -> Set[str]:
    return {_route_key(r) for r in surface.get("routes", [])}


def _arch_set(surface: Dict) -> Set[str]:
    out: Set[str] = set()
    for pkg, names in surface.get("model_architectures", {}).items():
        for n in names:
            out.add(f"{pkg}:{n}")
    return out


def diff_sets(baseline: Set[str], current: Set[str]) -> Tuple[List[str], List[str]]:
    added = sorted(current - baseline)
    removed = sorted(baseline - current)
    return added, removed


def compute_diff(baseline: Dict, current: Dict) -> Dict:
    """Return a structured diff of the parity-relevant surface dimensions."""
    diff: Dict[str, Dict[str, List[str]]] = {}

    r_add, r_rem = diff_sets(_route_set(baseline), _route_set(current))
    diff["routes"] = {"added": r_add, "removed": r_rem}

    f_add, f_rem = diff_sets(
        set(baseline.get("chat_request_fields", [])),
        set(current.get("chat_request_fields", [])),
    )
    diff["chat_request_fields"] = {"added": f_add, "removed": f_rem}

    a_add, a_rem = diff_sets(_arch_set(baseline), _arch_set(current))
    diff["model_architectures"] = {"added": a_add, "removed": a_rem}

    e_add, e_rem = diff_sets(
        {str(c) for c in baseline.get("error_codes", [])},
        {str(c) for c in current.get("error_codes", [])},
    )
    diff["error_codes"] = {"added": e_add, "removed": e_rem}

    diff["_meta"] = {
        "baseline_version": baseline.get("omlx_version"),
        "current_version": current.get("omlx_version"),
    }
    return diff


def has_additions(diff: Dict) -> bool:
    return any(
        diff[k]["added"]
        for k in ("routes", "chat_request_fields", "model_architectures", "error_codes")
    )


def has_removals(diff: Dict) -> bool:
    return any(
        diff[k]["removed"]
        for k in ("routes", "chat_request_fields", "model_architectures", "error_codes")
    )


def print_report(diff: Dict) -> None:
    meta = diff["_meta"]
    if meta["baseline_version"] != meta["current_version"]:
        print(
            YELLOW(
                f"omlx version: {meta['baseline_version']} (locked) -> "
                f"{meta['current_version']} (live)"
            )
        )

    labels = {
        "routes": "Routes",
        "chat_request_fields": "ChatCompletionRequest fields",
        "model_architectures": "Model architectures (model_type)",
        "error_codes": "Error status codes",
    }
    for key, label in labels.items():
        added = diff[key]["added"]
        removed = diff[key]["removed"]
        if not added and not removed:
            continue
        print(BOLD(label))
        for item in added:
            print(RED(f"  + {item}") + "   (NEW upstream -> parity work)")
        for item in removed:
            print(YELLOW(f"  - {item}") + "   (removed upstream)")

    if has_additions(diff):
        print()
        print(RED(BOLD("DRIFT: upstream omlx added surface not reflected in the lockfile.")))
        print("Each `+` above is parity work: wire it into native MLXServe (new route =")
        print("parity gap; new field = request knob; new model_type = harness matrix cell),")
        print("then run `check_drift.py --update` to re-baseline.")
    elif has_removals(diff):
        print()
        print(YELLOW("No additions. Upstream removed surface only (informational)."))
    else:
        print(GREEN("No drift: live omlx surface matches the committed lockfile."))


def load_json(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff live omlx surface vs committed lockfile")
    parser.add_argument("--lock", default=LOCK_PATH, help="committed lockfile path")
    parser.add_argument(
        "--against", default=None,
        help="diff the lockfile against this surface JSON instead of re-extracting live "
             "(useful for simulating drift)",
    )
    parser.add_argument("--update", action="store_true", help="refresh the lockfile to the live surface and exit")
    parser.add_argument("--strict", action="store_true", help="also exit non-zero on removals")
    parser.add_argument("--json", action="store_true", help="emit machine-readable diff JSON")
    args = parser.parse_args()

    if args.update:
        omlx_src = ex.resolve_omlx_src()
        mlx_site = ex.resolve_mlx_site()
        surface = ex.build_surface(omlx_src, mlx_site)
        with open(args.lock, "w") as fh:
            fh.write(json.dumps(surface, indent=2) + "\n")
        c = surface["counts"]
        print(f"Updated {args.lock}")
        print(
            f"  routes={c['routes']}  chat_request_fields={c['chat_request_fields']}  "
            f"error_codes={c['error_codes']}  "
            + "  ".join(f"{k}={v}" for k, v in c["model_architectures"].items())
        )
        return 0

    if not os.path.exists(args.lock):
        print(f"ERROR: lockfile not found: {args.lock}", file=sys.stderr)
        print("Run `extract_surface.py` first to create the baseline.", file=sys.stderr)
        return 2
    baseline = load_json(args.lock)

    if args.against:
        current = load_json(args.against)
    else:
        omlx_src = ex.resolve_omlx_src()
        mlx_site = ex.resolve_mlx_site()
        current = ex.build_surface(omlx_src, mlx_site)

    diff = compute_diff(baseline, current)

    if args.json:
        print(json.dumps(diff, indent=2))
    else:
        print_report(diff)

    if has_additions(diff):
        return 1
    if args.strict and has_removals(diff):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
