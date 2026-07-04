#!/usr/bin/env python3
"""Upstream drift-watch surface extractor (MLXServe <-> omlx parity, Milestone M9).

Introspects the *installed* omlx + mlx-lm (+ mlx-vlm / mlx-embeddings) and emits a
canonical JSON lockfile capturing omlx's programmatic surface:

  * routes                -- every (method, path) omlx registers (FastAPI decorators)
  * chat_request_fields   -- field names of ChatCompletionRequest
  * model_architectures   -- loadable model_type basenames (mlx-lm / mlx-vlm / mlx-embeddings)
  * error_codes           -- HTTP status codes omlx maps (best-effort)

The lockfile is the committed baseline; `check_drift.py` diffs the live surface against
it and flags ADDITIONS (new route / field / model_type) = parity work not yet covered.

Design notes
------------
* Pure stdlib (ast / json / re / glob). We deliberately do NOT `import omlx` -- it pulls
  torch stubs, mlx, huge modules with import side effects. AST parsing of the source is
  faster, side-effect-free, and works even when the package can't be imported.
* "Loadable architecture" for mlx-lm = a `models/<name>.py` that defines a top-level
  `class Model`. Helper modules (base.py, cache.py, activations.py, ...) do not, so they
  are excluded. This is the semantically correct "what model_types can omlx load" set.
* Router prefixes are read from each file's `APIRouter(prefix=...)` and applied to
  `@router.<method>` decorators; `@app.<method>` decorators use no prefix (root app).

Run:
    python3 extract_surface.py                 # writes omlx-surface.lock.json next to this file
    python3 extract_surface.py --stdout        # print JSON to stdout, don't write
    python3 extract_surface.py -o /path/x.json # custom output path

Path overrides (env, else auto-detected from the sidecar venv):
    OMLX_SRC   -- omlx package dir (contains server.py, api/, mcp/, admin/)
    MLX_SITE   -- site-packages dir containing mlx_lm / mlx_vlm / mlx_embeddings
"""

from __future__ import annotations

import argparse
import ast
import datetime
import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

SCHEMA_VERSION = 1
LOCK_FILENAME = "omlx-surface.lock.json"

# Fallback locations (the sidecar venv/src laid down by the parity effort). Auto-detection
# via `pip show omlx` is attempted first; these are the last-resort defaults.
DEFAULT_VENV = "/private/tmp/omlx-sidecar-venv-20260703"
DEFAULT_OMLX_SRC = "/tmp/omlx-sidecar-src-20260703/omlx"

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "websocket"}

# Files whose FastAPI route decorators we scan, relative to the omlx package root.
ROUTE_SOURCE_FILES = [
    "server.py",
    "api/mcp_routes.py",
    "api/audio_routes.py",
    "api/anthropic_routes.py",  # may not exist; skipped gracefully
    "admin/routes.py",
]

# Files scanned (best-effort) for HTTP status codes omlx maps.
ERROR_CODE_SOURCE_FILES = [
    "server.py",
    "exceptions.py",
    "api/mcp_routes.py",
    "api/audio_routes.py",
    "admin/routes.py",
]

STATUS_CODE_RE = re.compile(r"status_code\s*=\s*(\d{3})")
HTTP_CONST_RE = re.compile(r"HTTP_(\d{3})")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    return None


def resolve_omlx_src() -> str:
    """Locate the installed omlx package source dir."""
    env = os.environ.get("OMLX_SRC")
    if env and os.path.isdir(env):
        return env
    # Try `pip show omlx` in the sidecar venv for the editable/install location.
    pip = os.path.join(DEFAULT_VENV, "bin", "pip")
    if os.path.exists(pip):
        show = _run([pip, "show", "omlx"])
        if show:
            editable = None
            location = None
            for line in show.splitlines():
                if line.startswith("Editable project location:"):
                    editable = line.split(":", 1)[1].strip()
                elif line.startswith("Location:"):
                    location = line.split(":", 1)[1].strip()
            for base in (editable, location):
                if base:
                    cand = os.path.join(base, "omlx")
                    if os.path.isdir(cand):
                        return cand
    if os.path.isdir(DEFAULT_OMLX_SRC):
        return DEFAULT_OMLX_SRC
    raise SystemExit(
        "ERROR: could not locate omlx source. Set OMLX_SRC to the omlx package dir."
    )


def resolve_mlx_site() -> str:
    """Locate the site-packages dir holding mlx_lm / mlx_vlm / mlx_embeddings."""
    env = os.environ.get("MLX_SITE")
    if env and os.path.isdir(env):
        return env
    py = os.path.join(DEFAULT_VENV, "bin", "python")
    if os.path.exists(py):
        out = _run([py, "-c", "import mlx_lm, os; print(os.path.dirname(os.path.dirname(mlx_lm.__file__)))"])
        if out:
            cand = out.strip()
            if os.path.isdir(cand):
                return cand
    # Last resort: glob the default venv.
    for cand in glob.glob(os.path.join(DEFAULT_VENV, "lib", "python*", "site-packages")):
        if os.path.isdir(os.path.join(cand, "mlx_lm")):
            return cand
    raise SystemExit(
        "ERROR: could not locate mlx_lm site-packages. Set MLX_SITE to the dir containing mlx_lm/."
    )


def read_omlx_version(omlx_src: str) -> Optional[str]:
    vfile = os.path.join(omlx_src, "_version.py")
    if os.path.exists(vfile):
        try:
            with open(vfile) as fh:
                text = fh.read()
            m = re.search(r"""__version__\s*=\s*['"]([^'"]+)['"]""", text)
            if m:
                return m.group(1)
            m = re.search(r"""version\s*=\s*['"]([^'"]+)['"]""", text)
            if m:
                return m.group(1)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Route extraction
# ---------------------------------------------------------------------------

def _router_prefix(tree: ast.AST) -> str:
    """Find `APIRouter(prefix="...")` in a module and return the prefix (or '')."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == "APIRouter":
            for kw in node.keywords:
                if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return ""


def _call_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_routes_from_file(path: str, source_label: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    prefix = _router_prefix(tree)
    routes: List[Dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not isinstance(func, ast.Attribute):
                continue
            base = func.value
            if not isinstance(base, ast.Name) or base.id not in ("app", "router"):
                continue
            method = func.attr.lower()
            if method not in HTTP_METHODS:
                continue
            if not dec.args or not isinstance(dec.args[0], ast.Constant):
                continue
            raw_path = str(dec.args[0].value)
            full = raw_path if base.id == "app" else (prefix + raw_path)
            routes.append(
                {"method": method.upper(), "path": full, "source": source_label}
            )
    return routes


def extract_routes(omlx_src: str) -> List[Dict[str, str]]:
    routes: List[Dict[str, str]] = []
    for rel in ROUTE_SOURCE_FILES:
        path = os.path.join(omlx_src, rel)
        if not os.path.exists(path):
            continue
        routes.extend(extract_routes_from_file(path, rel))
    # Deduplicate on (method, path); keep first source. Sort for determinism.
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for r in sorted(routes, key=lambda x: (x["path"], x["method"])):
        key = (r["method"], r["path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# ChatCompletionRequest field extraction
# ---------------------------------------------------------------------------

def extract_chat_fields(omlx_src: str) -> List[str]:
    path = os.path.join(omlx_src, "api", "openai_models.py")
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    fields: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ChatCompletionRequest":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    if not name.startswith("_"):
                        fields.append(name)
            break
    return sorted(set(fields))


# ---------------------------------------------------------------------------
# Model architecture extraction
# ---------------------------------------------------------------------------

def _file_defines_model_class(path: str) -> bool:
    try:
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return True
    return False


def extract_mlx_lm_archs(mlx_site: str) -> List[str]:
    models_dir = os.path.join(mlx_site, "mlx_lm", "models")
    archs: List[str] = []
    for path in glob.glob(os.path.join(models_dir, "*.py")):
        base = os.path.splitext(os.path.basename(path))[0]
        if base == "__init__":
            continue
        if _file_defines_model_class(path):
            archs.append(base)
    return sorted(set(archs))


def extract_dir_archs(mlx_site: str, pkg: str) -> List[str]:
    """VLM-style: each subdirectory under <pkg>/models is an architecture package."""
    models_dir = os.path.join(mlx_site, pkg, "models")
    if not os.path.isdir(models_dir):
        return []
    archs: List[str] = []
    for entry in os.listdir(models_dir):
        full = os.path.join(models_dir, entry)
        if not os.path.isdir(full) or entry == "__pycache__":
            continue
        archs.append(entry)
    return sorted(set(archs))


def extract_module_archs(mlx_site: str, pkg: str, skip: Set[str]) -> List[str]:
    """Embeddings-style: each models/<name>.py (minus helpers) is an architecture."""
    models_dir = os.path.join(mlx_site, pkg, "models")
    if not os.path.isdir(models_dir):
        return []
    archs: List[str] = []
    for path in glob.glob(os.path.join(models_dir, "*.py")):
        base = os.path.splitext(os.path.basename(path))[0]
        if base in skip or base.startswith("__"):
            continue
        archs.append(base)
    return sorted(set(archs))


def extract_model_architectures(mlx_site: str) -> Dict[str, List[str]]:
    return {
        "mlx_lm": extract_mlx_lm_archs(mlx_site),
        "mlx_vlm": extract_dir_archs(mlx_site, "mlx_vlm"),
        "mlx_embeddings": extract_module_archs(
            mlx_site, "mlx_embeddings", skip={"base", "cache"}
        ),
    }


# ---------------------------------------------------------------------------
# Error code extraction (best-effort)
# ---------------------------------------------------------------------------

def extract_error_codes(omlx_src: str) -> List[int]:
    codes: Set[int] = set()
    for rel in ERROR_CODE_SOURCE_FILES:
        path = os.path.join(omlx_src, rel)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            text = fh.read()
        for m in STATUS_CODE_RE.finditer(text):
            codes.add(int(m.group(1)))
        for m in HTTP_CONST_RE.finditer(text):
            codes.add(int(m.group(1)))
    # Keep plausible HTTP status codes only (1xx-5xx).
    return sorted(c for c in codes if 100 <= c <= 599)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build_surface(omlx_src: str, mlx_site: str) -> Dict:
    routes = extract_routes(omlx_src)
    chat_fields = extract_chat_fields(omlx_src)
    archs = extract_model_architectures(mlx_site)
    error_codes = extract_error_codes(omlx_src)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "omlx_version": read_omlx_version(omlx_src),
        "sources": {"omlx_src": omlx_src, "mlx_site": mlx_site},
        "counts": {
            "routes": len(routes),
            "chat_request_fields": len(chat_fields),
            "model_architectures": {k: len(v) for k, v in archs.items()},
            "error_codes": len(error_codes),
        },
        "routes": routes,
        "chat_request_fields": chat_fields,
        "model_architectures": archs,
        "error_codes": error_codes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract omlx programmatic surface -> lockfile JSON")
    parser.add_argument(
        "-o", "--output", default=None,
        help=f"output path (default: ./{LOCK_FILENAME} next to this script)",
    )
    parser.add_argument("--stdout", action="store_true", help="print JSON to stdout, do not write a file")
    args = parser.parse_args()

    omlx_src = resolve_omlx_src()
    mlx_site = resolve_mlx_site()
    surface = build_surface(omlx_src, mlx_site)
    text = json.dumps(surface, indent=2, sort_keys=False) + "\n"

    if args.stdout:
        sys.stdout.write(text)
        return 0

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCK_FILENAME)
    with open(out_path, "w") as fh:
        fh.write(text)
    c = surface["counts"]
    print(f"Wrote {out_path}")
    print(
        f"  routes={c['routes']}  chat_request_fields={c['chat_request_fields']}  "
        f"error_codes={c['error_codes']}"
    )
    print(
        "  model_architectures: "
        + "  ".join(f"{k}={v}" for k, v in c["model_architectures"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
