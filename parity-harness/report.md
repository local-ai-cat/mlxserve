# MLXServe Native ⇄ omlx Parity Conformance Matrix

Generated: 2026-07-04 18:37  
native `mlxserve-http` (git 672ec7b) vs `omlx` (0.4.5.dev1) · tier=**smoke** · 4 models

Legend: **PASS** = native matches omlx · **GAP** = native diverges (recorded, not a hard fail — the baseline distance) · **FAIL** = harness-level failure (server unreachable / crash).

## Summary

| Axis | PASS | GAP | FAIL |
| --- | ---: | ---: | ---: |
| 1. Schema conformance | 27 | 3 | 0 |
| 2. Semantic agreement | 12 | 0 | 0 |
| 3. Model-architecture matrix | 3 | 1 | 0 |
| 4. Error semantics | 4 | 4 | 0 |
| 5. Streaming framing | 15 | 9 | 0 |


## 1. Schema conformance

| Cell | Verdict | Note |
| --- | --- | --- |
| non-stream top-level keys · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native object='chat.completion' |
| choice keys · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | identical key set |
| usage base keys (native) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |
| non-stream top-level keys · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native object='chat.completion' |
| choice keys · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | identical key set |
| usage base keys (native) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · mlx-community--Llama-3.2-1B-Instruct-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |
| non-stream top-level keys · mlx-community--Qwen3-0.6B-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · mlx-community--Qwen3-0.6B-4bit | PASS | native object='chat.completion' |
| choice keys · mlx-community--Qwen3-0.6B-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · mlx-community--Qwen3-0.6B-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · mlx-community--Qwen3-0.6B-4bit | PASS | identical key set |
| usage base keys (native) · mlx-community--Qwen3-0.6B-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · mlx-community--Qwen3-0.6B-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · mlx-community--Qwen3-0.6B-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · mlx-community--Qwen3-0.6B-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · mlx-community--Qwen3-0.6B-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |


## 2. Semantic agreement

| Cell | Verdict | Note |
| --- | --- | --- |
| usage.output_tokens agree (±2) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native=40, omlx=40, \|Δ\|=0 |
| stream total deltas vs usage · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native deltas(content=40,reason=0) usage.out=40 \| omlx deltas(content=1,reason=2) usage.out=40 |
| reasoning split behavior · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native reasoning_content deltas=0 (native streams CoT as content); omlx reasoning_content deltas=2 |
| visible text agreement · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | skipped hard compare (reasoning model; native inlines CoT into content, omlx splits it). native[:40]='Okay, so I need to figure out the capita' omlx[:40]='Okay, so I need to figure out the capita' |
| usage.output_tokens agree (±2) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native=8, omlx=7, \|Δ\|=1 |
| stream total deltas vs usage · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native deltas(content=7,reason=0) usage.out=8 \| omlx deltas(content=1,reason=0) usage.out=7 |
| reasoning split behavior · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native reasoning_content deltas=0 (native streams CoT as content); omlx reasoning_content deltas=0 |
| visible text agreement · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | leading-word overlap=6; native='The capital of France is Paris.' omlx='The capital of France is Paris.' |
| usage.output_tokens agree (±2) · mlx-community--Qwen3-0.6B-4bit | PASS | native=40, omlx=40, \|Δ\|=0 |
| stream total deltas vs usage · mlx-community--Qwen3-0.6B-4bit | PASS | native deltas(content=40,reason=0) usage.out=40 \| omlx deltas(content=1,reason=1) usage.out=40 |
| reasoning split behavior · mlx-community--Qwen3-0.6B-4bit | PASS | native reasoning_content deltas=0 (native streams CoT as content); omlx reasoning_content deltas=1 |
| visible text agreement · mlx-community--Qwen3-0.6B-4bit | PASS | skipped hard compare (reasoning model; native inlines CoT into content, omlx splits it). native[:40]='<think>\nOkay, the user is asking for the' omlx[:40]='Okay, the user is asking for the capital' |


## 3. Model-architecture matrix

| Cell | Verdict | Note |
| --- | --- | --- |
| dense full-attn, reasoning (DeepSeek) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | both 200+coherent. native[:50]='Okay, so I need to write a short sentence about th' omlx[:50]='Okay, so I need to write a short sentence about th' |
| dense full-attn (Llama) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | both 200+coherent. native[:50]="The ocean covers over 70% of the Earth's surface a" omlx[:50]="The ocean covers over 70% of the Earth's surface a" |
| dense full-attn (Qwen3) · mlx-community--Qwen3-0.6B-4bit | PASS | both 200+coherent. native[:50]='<think>\nOkay, the user wants me to write a short s' omlx[:50]='Okay, the user wants me to write a short sentence ' |
| sliding-window (Gemma3) · mlx-community--gemma-4-e2b-it-4bit | GAP | NATIVE FAULT: HTTP 500: BatchKVCache cannot merge rotating cache type RotatingKVCache. (status 500); omlx OK |


## 4. Error semantics

| Cell | Verdict | Note |
| --- | --- | --- |
| unknown-model: omlx status/shape | PASS | HTTP 404, error keys ['code', 'message', 'param', 'type'] |
| unknown-model: native vs omlx | GAP | native HTTP 200 (omlx 404); native error keys None; native body not an error envelope: '{"model":"model-that-does-not-exist","usage":{"prompt_tokens_per_second":1091.2838311628709,"input_t' |
| unknown-model: native validates model field | GAP | native returns HTTP 200 and serves its launch-pinned model for an unknown model id (does NOT validate `model`); omlx returns 404. |
| malformed-json: omlx status/shape | PASS | HTTP 422, error keys ['code', 'message', 'param', 'type'] |
| malformed-json: native vs omlx | GAP | native HTTP 500 (omlx 422); native error keys ['message']; MISSING ['code', 'param', 'type'] |
| missing-messages: omlx status/shape | PASS | HTTP 422, error keys ['code', 'message', 'param', 'type'] |
| missing-messages: native vs omlx | GAP | native HTTP 500 (omlx 422); native error keys ['message']; MISSING ['code', 'param', 'type'] |
| auth: omlx requires API key | PASS | omlx no-auth GET /v1/models -> HTTP 401; native has no auth layer (open loopback). |


## 5. Streaming framing

| Cell | Verdict | Note |
| --- | --- | --- |
| [DONE] terminator emitted (native) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | native first-chunk model='mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | usage chunk present with include_usage=True: True; absent when False: False |
| include_usage gating (omlx) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | omlx closed socket after [DONE] |
| [DONE] terminator emitted (native) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · mlx-community--Llama-3.2-1B-Instruct-4bit | GAP | native first-chunk model='mlx-community--Llama-3.2-1B-Instruct-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · mlx-community--Llama-3.2-1B-Instruct-4bit | GAP | usage chunk present with include_usage=True: True; absent when False: False |
| include_usage gating (omlx) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · mlx-community--Llama-3.2-1B-Instruct-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · mlx-community--Llama-3.2-1B-Instruct-4bit | PASS | omlx closed socket after [DONE] |
| [DONE] terminator emitted (native) · mlx-community--Qwen3-0.6B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · mlx-community--Qwen3-0.6B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · mlx-community--Qwen3-0.6B-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · mlx-community--Qwen3-0.6B-4bit | GAP | native first-chunk model='mlx-community--Qwen3-0.6B-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · mlx-community--Qwen3-0.6B-4bit | GAP | usage chunk present with include_usage=True: True; absent when False: False |
| include_usage gating (omlx) · mlx-community--Qwen3-0.6B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · mlx-community--Qwen3-0.6B-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · mlx-community--Qwen3-0.6B-4bit | PASS | omlx closed socket after [DONE] |

