# MLXServe Native ⇄ omlx Parity Conformance Matrix

Generated: 2026-07-06 21:17  
native `mlxserve-http` (git c33f07f) vs `omlx` (0.4.5.dev1) · tier=**full** · 17 models

Legend: **PASS** = native matches omlx · **GAP** = native diverges (recorded, not a hard fail — the baseline distance) · **FAIL** = harness-level failure (server unreachable / crash).

## Summary

| Axis | PASS | GAP | FAIL |
| --- | ---: | ---: | ---: |
| 1. Schema conformance | 0 | 0 | 0 |
| 2. Semantic agreement | 0 | 0 | 0 |
| 3. Model-architecture matrix | 1 | 0 | 0 |
| 4. Error semantics | 0 | 0 | 0 |
| 5. Streaming framing | 0 | 0 | 0 |
| 6. Audio | 0 | 0 | 0 |
| 7. Endpoint smoke | 0 | 0 | 0 |
| 8. Prefix cache | 0 | 0 | 0 |
| 9. Grammar constraints | 0 | 0 | 0 |
| 10. Config axis | 0 | 0 | 0 |
| 11. Perf fleet | 0 | 0 | 0 |


## 1. Schema conformance

_no cells recorded_


## 2. Semantic agreement

_no cells recorded_


## 3. Model-architecture matrix

| Cell | Verdict | Note |
| --- | --- | --- |
| Mamba-hybrid + MoE (Qwen3.6-35B-A3B) · Qwen3.6-35B-A3B-4bit | PASS | both 200+coherent. native[:50]='Thinking Process:\n\n1.  **Deconstruct the request:*' omlx[:50]='Thinking Process:\n\n1.  **Deconstruct the request:*' |


## 4. Error semantics

_no cells recorded_


## 5. Streaming framing

_no cells recorded_


## 6. Audio

_no cells recorded_


## 7. Endpoint smoke

_no cells recorded_


## 8. Prefix cache

_no cells recorded_


## 9. Grammar constraints

_no cells recorded_


## 10. Config axis

_no cells recorded_


## 11. Perf fleet

_no cells recorded_


## Grammar overhead

_grammar overhead not captured in this pass_

