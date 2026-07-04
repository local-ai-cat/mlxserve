# Upstream Drift-Watch — omlx surface lockfile (M9)

Keeps parity a **maintained property**, not a one-time achievement.

The [differential parity harness](../parity-harness/) proves native MLXServe
matches `omlx` *today*. But `omlx` is a moving target: every upstream release can
add a route, a request knob, or a new model architecture. When it does, native
MLXServe silently falls behind and the harness matrix doesn't even know to test
the new thing. The drift-watch closes that loop.

It captures `omlx`'s programmatic surface in a committed JSON **lockfile**
(`omlx-surface.lock.json`) and, on demand, diffs the live installed `omlx`/`mlx-lm`
against it. Any **addition** — new endpoint, new `ChatCompletionRequest` field, new
loadable `model_type` — is flagged as parity work that isn't covered yet.

## What the lockfile captures

| Dimension | Source | Meaning |
|-----------|--------|---------|
| **Routes** — every `(method, path)` | FastAPI `@app.*` / `@router.*` decorators in `server.py`, `api/mcp_routes.py`, `api/audio_routes.py`, `admin/routes.py` (router `prefix=` applied) | The HTTP surface native must serve. |
| **Chat request fields** | Field annotations of `ChatCompletionRequest` in `api/openai_models.py` | Every generation knob a client can send. |
| **Model architectures** | `mlx_lm/models/*.py` that define a top-level `class Model`; `mlx_vlm/models/*` dirs; `mlx_embeddings/models/*.py` | The set of loadable `model_type`s = harness matrix rows. |
| **Error codes** | `status_code=NNN` / `HTTP_NNN` in `server.py`, `exceptions.py`, routers (best-effort) | HTTP statuses omlx maps. |

Extraction is **pure stdlib AST** — it never imports `omlx` (which drags in torch
stubs, mlx, and heavy import side effects). "Loadable architecture" for `mlx-lm`
means a `models/<name>.py` defining `class Model`; helper modules (`base.py`,
`cache.py`, `activations.py`, …) are correctly excluded.

## Baseline snapshot (committed)

| Dimension | Count |
|-----------|-------|
| Routes (15 core `@app` + 3 MCP + 3 audio + 85 admin) | **106** |
| ChatCompletionRequest fields | **26** |
| `mlx-lm` model_types | **108** |
| `mlx-vlm` architectures | **77** |
| `mlx-embeddings` architectures | **10** |
| Error status codes | **13** |

## Usage

Everything runs from this directory. No third-party deps.

```bash
# Re-generate the lockfile from the installed omlx/mlx-lm (writes omlx-surface.lock.json)
python3 extract_surface.py
python3 extract_surface.py --stdout          # print, don't write

# Check for drift: re-extract the live surface and diff vs the committed lockfile
python3 check_drift.py                        # exit 0 = clean, 1 = additions found
python3 check_drift.py --json                 # machine-readable diff

# After you've absorbed the drift into native MLXServe + the harness, re-baseline:
python3 check_drift.py --update
```

### Path resolution

The scripts auto-detect the omlx source and `mlx-lm` install from the parity-effort
sidecar venv (`/private/tmp/omlx-sidecar-venv-20260703`, via `pip show omlx`).
Override if your layout differs:

```bash
OMLX_SRC=/path/to/omlx MLX_SITE=/path/to/site-packages python3 extract_surface.py
```

### Exit codes (for CI / cron)

`check_drift.py` exits:

- **0** — no additions; parity surface is fully accounted for.
- **1** — additions found (new route / field / model_type) → parity work to do.
- **2** — usage / missing-lockfile error.

Removals (upstream dropped surface) are reported as informational and do **not**
fail by default — losing upstream surface isn't unmet parity work. Pass `--strict`
to fail on removals too.

## How it feeds the parity effort

Run it after every `omlx` bump. Each `+` line maps to a concrete action:

- **New `model_type`** → add a cell to the harness model-architecture matrix
  (axis 3 in [`../parity-harness`](../parity-harness/README.md)) and confirm native
  loads it.
- **New route** → a parity gap: native MLXServe must serve the endpoint (or
  consciously decline it).
- **New `ChatCompletionRequest` field** → a request knob to plumb through the
  native request path.
- **New error code** → an error-semantics cell (axis 4) to align.

Then wire the new coverage into native, and `check_drift.py --update` to re-baseline
the lockfile — the diff is empty again until upstream moves next.

### Wiring to CI / cron (later)

Not yet automated. When wired, the intended shape is a scheduled job that
`pip install -U omlx` into the sidecar venv, runs `python3 check_drift.py`, and
opens an issue / fails the job on exit 1. The committed lockfile is the baseline it
diffs against, so drift is reviewed in a PR that bumps the lockfile alongside the
native code that covers the new surface.

## Simulating drift (to test the watcher)

`check_drift.py --against <surface.json>` diffs the committed lockfile against an
arbitrary surface JSON instead of re-extracting live — handy for tests and demos:

```bash
python3 extract_surface.py --stdout > /tmp/s.json
# ...hand-edit /tmp/s.json to add a fake route/field/model_type...
python3 check_drift.py --against /tmp/s.json   # exits 1, lists the additions
```

## Files

- `extract_surface.py` — surface extractor → lockfile.
- `check_drift.py` — re-extract + diff + `--update`.
- `omlx-surface.lock.json` — committed baseline snapshot (the thing drift is measured against).
