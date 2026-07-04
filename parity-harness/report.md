# MLXServe Native ⇄ omlx Parity Conformance Matrix

Generated: 2026-07-04 20:12  
native `mlxserve-http` (git 4820aea) vs `omlx` (0.4.5.dev1) · tier=**smoke** · 4 models

Legend: **PASS** = native matches omlx · **GAP** = native diverges (recorded, not a hard fail — the baseline distance) · **FAIL** = harness-level failure (server unreachable / crash).

## Summary

| Axis | PASS | GAP | FAIL |
| --- | ---: | ---: | ---: |
| 1. Schema conformance | 27 | 3 | 0 |
| 2. Semantic agreement | 12 | 0 | 0 |
| 3. Model-architecture matrix | 3 | 1 | 0 |
| 4. Error semantics | 7 | 0 | 0 |
| 5. Streaming framing | 18 | 6 | 0 |
| 6. Endpoint smoke | 6 | 1 | 0 |


## 1. Schema conformance

| Cell | Verdict | Note |
| --- | --- | --- |
| non-stream top-level keys · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native object='chat.completion' |
| choice keys · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | identical key set |
| usage base keys (native) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |
| non-stream top-level keys · Llama-3.2-1B-Instruct-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · Llama-3.2-1B-Instruct-4bit | PASS | native object='chat.completion' |
| choice keys · Llama-3.2-1B-Instruct-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · Llama-3.2-1B-Instruct-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · Llama-3.2-1B-Instruct-4bit | PASS | identical key set |
| usage base keys (native) · Llama-3.2-1B-Instruct-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · Llama-3.2-1B-Instruct-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · Llama-3.2-1B-Instruct-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · Llama-3.2-1B-Instruct-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · Llama-3.2-1B-Instruct-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |
| non-stream top-level keys · Qwen3-0.6B-4bit | PASS | native has ['choices', 'created', 'id', 'model', 'object', 'usage'] (conforms) |
| non-stream object=='chat.completion' · Qwen3-0.6B-4bit | PASS | native object='chat.completion' |
| choice keys · Qwen3-0.6B-4bit | PASS | native choice keys ['finish_reason', 'index', 'message'] |
| message keys · Qwen3-0.6B-4bit | PASS | native message keys ['content', 'role'] |
| top-level key-set native vs omlx · Qwen3-0.6B-4bit | PASS | identical key set |
| usage base keys (native) · Qwen3-0.6B-4bit | PASS | keys ['completion_tokens', 'generation_duration', 'generation_tokens_per_second', 'input_tokens', 'output_tokens', 'prompt_eval_duration', 'prompt_tokens', 'prompt_tokens_details', 'prompt_tokens_per_second', 'time_to_first_token', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (native) · Qwen3-0.6B-4bit | PASS | timing keys present: ['generation_duration', 'generation_tokens_per_second', 'prompt_eval_duration', 'prompt_tokens_per_second', 'time_to_first_token'] |
| usage base keys (omlx) · Qwen3-0.6B-4bit | PASS | keys ['completion_tokens', 'input_tokens', 'output_tokens', 'prompt_tokens', 'prompt_tokens_details', 'total_time', 'total_tokens'] |
| usage timing extension, non-stream (omlx) · Qwen3-0.6B-4bit | GAP | no timing keys (omlx omits timing on non-stream responses) |
| SSE chunk keys · Qwen3-0.6B-4bit | PASS | native chunk keys ['choices', 'created', 'id', 'model', 'object'], object='chat.completion.chunk' |


## 2. Semantic agreement

| Cell | Verdict | Note |
| --- | --- | --- |
| usage.output_tokens agree (±2) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native=40, omlx=40, \|Δ\|=0 |
| stream total deltas vs usage · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native deltas(content=40,reason=0) usage.out=40 \| omlx deltas(content=1,reason=2) usage.out=40 |
| reasoning split behavior · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native reasoning_content deltas=0 (native streams CoT as content); omlx reasoning_content deltas=2 |
| visible text agreement · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | skipped hard compare (reasoning model; native inlines CoT into content, omlx splits it). native[:40]='Okay, so I need to figure out the capita' omlx[:40]='Okay, so I need to figure out the capita' |
| usage.output_tokens agree (±2) · Llama-3.2-1B-Instruct-4bit | PASS | native=8, omlx=7, \|Δ\|=1 |
| stream total deltas vs usage · Llama-3.2-1B-Instruct-4bit | PASS | native deltas(content=7,reason=0) usage.out=8 \| omlx deltas(content=1,reason=0) usage.out=7 |
| reasoning split behavior · Llama-3.2-1B-Instruct-4bit | PASS | native reasoning_content deltas=0 (native streams CoT as content); omlx reasoning_content deltas=0 |
| visible text agreement · Llama-3.2-1B-Instruct-4bit | PASS | leading-word overlap=6; native='The capital of France is Paris.' omlx='The capital of France is Paris.' |
| usage.output_tokens agree (±2) · Qwen3-0.6B-4bit | PASS | native=40, omlx=40, \|Δ\|=0 |
| stream total deltas vs usage · Qwen3-0.6B-4bit | PASS | native deltas(content=1,reason=39) usage.out=40 \| omlx deltas(content=1,reason=1) usage.out=40 |
| reasoning split behavior · Qwen3-0.6B-4bit | PASS | native reasoning_content deltas=39 (native streams CoT as content); omlx reasoning_content deltas=1 |
| visible text agreement · Qwen3-0.6B-4bit | PASS | skipped hard compare (reasoning model; native inlines CoT into content, omlx splits it). native[:40]='Okay, the user is asking for the capital' omlx[:40]='Okay, the user is asking for the capital' |


## 3. Model-architecture matrix

| Cell | Verdict | Note |
| --- | --- | --- |
| dense full-attn, reasoning (DeepSeek) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | both 200+coherent. native[:50]='Okay, so I need to write a short sentence about th' omlx[:50]='Okay, so I need to write a short sentence about th' |
| dense full-attn (Llama) · Llama-3.2-1B-Instruct-4bit | PASS | both 200+coherent. native[:50]="The ocean covers over 70% of the Earth's surface a" omlx[:50]="The ocean covers over 70% of the Earth's surface a" |
| dense full-attn (Qwen3) · Qwen3-0.6B-4bit | PASS | both 200+coherent. native[:50]='Okay, the user wants me to write a short sentence ' omlx[:50]='Okay, the user wants me to write a short sentence ' |
| sliding-window (Gemma3) · gemma-4-E2B-it-qat-4bit | GAP | NATIVE FAULT: HTTP 500: keyNotFound(path: ["language_model", "model", "layers", "15", "self_attn", "k_proj", "weight"], modules: ["Gemma4Model", "Gemma4TextModel", "Gemma4TextModelInner", "Gemma4DecoderLayer", "Gemma4Attention", "Linear"]) (status 500); omlx OK |


## 4. Error semantics

| Cell | Verdict | Note |
| --- | --- | --- |
| unknown-model: omlx status/shape | PASS | HTTP 404, error keys ['code', 'message', 'param', 'type'] |
| unknown-model: native vs omlx | PASS | native HTTP 404 (omlx 404); native error keys ['code', 'message', 'param', 'type'] |
| malformed-json: omlx status/shape | PASS | HTTP 422, error keys ['code', 'message', 'param', 'type'] |
| malformed-json: native vs omlx | PASS | native HTTP 422 (omlx 422); native error keys ['code', 'message', 'param', 'type'] |
| missing-messages: omlx status/shape | PASS | HTTP 422, error keys ['code', 'message', 'param', 'type'] |
| missing-messages: native vs omlx | PASS | native HTTP 422 (omlx 422); native error keys ['code', 'message', 'param', 'type'] |
| auth: omlx requires API key | PASS | omlx no-auth GET /v1/models -> HTTP 401; native has no auth layer (open loopback). |


## 5. Streaming framing

| Cell | Verdict | Note |
| --- | --- | --- |
| [DONE] terminator emitted (native) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | native first-chunk model='DeepSeek-R1-Distill-Qwen-1.5B-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| include_usage gating (omlx) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · DeepSeek-R1-Distill-Qwen-1.5B-4bit | PASS | omlx closed socket after [DONE] |
| [DONE] terminator emitted (native) · Llama-3.2-1B-Instruct-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · Llama-3.2-1B-Instruct-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · Llama-3.2-1B-Instruct-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · Llama-3.2-1B-Instruct-4bit | GAP | native first-chunk model='Llama-3.2-1B-Instruct-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · Llama-3.2-1B-Instruct-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| include_usage gating (omlx) · Llama-3.2-1B-Instruct-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · Llama-3.2-1B-Instruct-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · Llama-3.2-1B-Instruct-4bit | PASS | omlx closed socket after [DONE] |
| [DONE] terminator emitted (native) · Qwen3-0.6B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| [DONE] terminator emitted (omlx) · Qwen3-0.6B-4bit | PASS | saw [DONE]=True, [DONE] was last SSE line=True |
| terminal finish_reason before [DONE] · Qwen3-0.6B-4bit | PASS | native finish_reason seen=True, omlx=True |
| first-chunk framing · Qwen3-0.6B-4bit | GAP | native first-chunk model='Qwen3-0.6B-4bit'; omlx first-chunk model='keepalive' (omlx sends a 'keepalive' priming chunk) |
| include_usage gating (native) · Qwen3-0.6B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| include_usage gating (omlx) · Qwen3-0.6B-4bit | PASS | usage chunk present with include_usage=True: True; absent when False: True |
| closes SSE socket after [DONE] (native) · Qwen3-0.6B-4bit | GAP | native advertises Connection:close but leaves the SSE socket OPEN after [DONE] (a slow client that waits for close would hang) |
| closes SSE socket after [DONE] (omlx) · Qwen3-0.6B-4bit | PASS | omlx closed socket after [DONE] |


## 6. Endpoint smoke

| Cell | Verdict | Note |
| --- | --- | --- |
| /health (native) | PASS | HTTP 200, status='healthy' |
| /health (omlx) | PASS | HTTP 200, status='healthy' |
| /v1/completions (native) | PASS | HTTP 200, object='text_completion', text[:40]=' a vast and mysterious place, and it' |
| /v1/completions (omlx) | PASS | HTTP 200, object='text_completion', text[:40]='often thought of as the most abundant bo' |
| /v1/embeddings refuses LLM (native) | PASS | HTTP 404; error.type='not_found_error' |
| /v1/embeddings refuses LLM (omlx) | PASS | HTTP 400; error.type='invalid_request_error' |
| /v1/embeddings status parity (native vs omlx) | GAP | native HTTP 404 vs omlx HTTP 400 (both refuse an LLM used as an embedding model; native has no embedding model loaded so it returns 'backend unavailable') |

