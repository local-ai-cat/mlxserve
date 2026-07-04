# Differential Parity Harness — native MLXServe ⇄ omlx

The verification instrument that proves native MLXServe matches the Python
`omlx` server **cell by cell**. It launches both OpenAI-compatible servers, fires
identical requests at each, and diffs the responses across **five axes**. The
generated matrix quantifies exactly how far native is from omlx today.

This is the source of truth we judge future Swift work against: land a native
milestone, re-run the harness, watch red cells turn green.

## The five axes

| # | Axis | What it diffs |
|---|------|---------------|
| 1 | **Schema conformance** | Same JSON keys/types in the non-stream response, the SSE chunk, and the `usage` object. |
| 2 | **Semantic agreement** | Greedy (temp 0), same prompt → output-token counts agree (from server `usage`, not client chunk counts) and, where reasoning is off, the visible text agrees. |
| 3 | **Model-architecture matrix** | The centerpiece. A short completion to BOTH servers per model spanning families (dense / sliding-window / MoE / Mamba-hybrid / VLM). A cell passes iff both return 200 + coherent output. A native fault where omlx succeeds is a real gap, mapped to the milestone that fixes it. |
| 4 | **Error semantics** | Unknown-model and malformed-body: HTTP status + error-body shape. |
| 5 | **Streaming framing** | SSE event order, `[DONE]` terminator, socket close, `stream_options.include_usage` gating. |

### Key comparison gotchas the harness handles

- **Reasoning split** — omlx streams chain-of-thought as `delta.reasoning_content`;
  native streams everything as `delta.content`. Semantic agreement counts *all*
  deltas and trusts `usage.output_tokens`, never client chunk counts.
- **Never-closing SSE socket** — native emits `[DONE]` but does **not** close the
  connection and may not flush a trailing newline after it. The stream reader
  scans raw chunks for the `[DONE]` sentinel and stops, instead of blocking on a
  line terminator that never comes. The non-close is recorded as a framing gap.
- **Prompt-token off-by-a-few** — the two chat templates tokenize the prompt
  slightly differently, so output-token agreement is asserted within a tolerance.

## Requirements

- A real **Apple-Silicon GPU** (MLX does not run on the iOS Simulator, and both
  servers need Metal). Without `PARITY_HARNESS=1` the whole suite **skips cleanly**.
- Python 3.9+ with `pytest` and `requests` (`pip install -r requirements.txt`).
- The frozen native binary + colocated `mlx.metallib`, the `omlx` binary, and the
  model farm (see paths below).

## Running

```bash
# Freezes the native binary, launches both servers, runs all axes, writes reports.
./run_parity.sh                    # smoke tier (default)
./run_parity.sh --tier full        # add the heavy on-demand models
./run_parity.sh -k matrix          # a single axis
```

`run_parity.sh` copies `mlxserve-http` + `mlx.metallib` to a frozen dir
(`/private/tmp/parity-native-baseline`) first — GOTCHA: native only finds Metal
if `mlx.metallib` sits **beside** the binary, and freezing keeps a later Swift
rebuild from disturbing a run. It then sets `PARITY_HARNESS=1` and invokes pytest.

Direct pytest invocation also works:

```bash
PARITY_HARNESS=1 python3 -m pytest -v --tier smoke
```

## Model tiers (`--tier`)

`smoke` runs every pass (small, fast); `full` adds the heavy on-demand models.
Conformance axes (schema / semantic / streaming) always run on the dense smoke
models native is known to serve; the tier only changes the **matrix** axis.

| Tier | Model | Family | Native today |
|------|-------|--------|--------------|
| smoke | `Qwen3-0.6B-4bit` | dense full-attn | 🟢 |
| smoke | `Llama-3.2-1B-Instruct-4bit` | dense full-attn | 🟢 |
| smoke | `DeepSeek-R1-Distill-Qwen-1.5B-4bit` | dense, reasoning | 🟢 |
| smoke | `gemma-4-e2b-it-4bit` | **sliding-window** | 🔴 M7+ |
| full | `gpt-oss-20b-MXFP4-Q8` | sliding-window + MoE + harmony | 🔴 M7+ |
| full | `Qwen3-Coder-30B-A3B-Instruct-4bit` | MoE | (verify) |
| full | `Qwen3.6-27B-MLX-4bit` | **Mamba-hybrid** | 🔴 M7+ |
| full | `Qwen2-VL-2B-Instruct-4bit` | VLM | 🔴 M7+ |

A model not present in the farm is skipped with a loud reason, not a hard fail.

## Outputs

Written to this directory on every run:

- **`report.html`** — the human-review artifact. Self-contained (inline CSS/JS,
  light/dark aware), with the parity score, per-axis pass rates, the color-coded
  architecture matrix, benchmark bars, and a gap list grouped by milestone.
- **`report.md`** — the plain conformance matrix (feeds the docs page later).
- **`report-junit.xml`** — JUnit XML for CI.
- **`logs/`** — captured native/omlx server stdout for debugging.

Verdicts: **PASS** = native matches omlx · **GAP** = native diverges (recorded,
not a hard fail — this is the baseline distance) · **FAIL** = harness-level
failure (server unreachable / crash). Matrix cells for families native is not
expected to serve yet (sliding-window / hybrid / VLM) record a GAP rather than
failing the run; dense/MoE cells hard-fail if native regresses.

## Configuration

Paths are env-overridable (see `parity/config.py`):

| Env var | Default |
|---------|---------|
| `PARITY_NATIVE_BIN` | `/private/tmp/parity-native-baseline/mlxserve-http` |
| `PARITY_OMLX_BIN` | `/private/tmp/omlx-sidecar-venv-20260703/bin/omlx` |
| `PARITY_MODEL_FARM` | `/private/tmp/omlx-all-models` |
| `PARITY_OMLX_API_KEY` | `devkey` |
| `PARITY_NATIVE_REPO` | `/Users/.../mlxserve-native` (for the build SHA in the report header) |

## Layout

```
parity-harness/
├── conftest.py            # gating, --tier, server fixtures, report emit hook
├── pytest.ini             # junitxml + markers
├── run_parity.sh          # freeze native binary + run
├── parity/
│   ├── config.py          # binaries, model farm, tiered model registry + families
│   ├── servers.py         # launch/health/teardown for native + omlx
│   ├── client.py          # HTTP + SSE client (stops on [DONE], captures timing)
│   └── report.py          # md + self-contained html + matrix/bench data
└── tests/
    ├── test_schema.py     # axis 1
    ├── test_semantic.py   # axis 2
    ├── test_matrix.py     # axis 3
    ├── test_errors.py     # axis 4
    ├── test_streaming.py  # axis 5
    └── test_benchmark.py  # benchmark capture (informational)
```
