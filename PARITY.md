# MLXServe Native vs oMLX Parity

Date: 2026-07-03

## Methodology

Model: `Qwen3-0.6B-4bit`

Native command:

```bash
swift build -c release
MLXSERVE_TEST_MODEL=/Users/timapple/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit \
  /tmp/mlxserve-m6-bench/mlxserve-bench \
  --runs 5 --warmup 2 --decode-tokens 16 \
  --output /tmp/mlxserve-native-bench.md
```

oMLX commands:

```bash
/tmp/omlx-sidecar-venv-20260703/bin/omlx serve \
  --model-dir /Users/timapple/Library/Caches/models/mlx-community \
  --port 18080 --base-path /tmp/omlx-m6-nocache-base \
  --no-cache --log-level warning

/tmp/omlx-sidecar-venv-20260703/bin/omlx serve \
  --model-dir /Users/timapple/Library/Caches/models/mlx-community \
  --port 18080 --base-path /tmp/omlx-m6-cache-base \
  --paged-ssd-cache-dir /tmp/omlx-m6-cache \
  --hot-cache-max-size 2GB --log-level warning
```

Prompt set: same text on both sides. The shared prefix was the sentence
`The capital of France is Paris. Swift concurrency protects shared state. GPU kernels execute matrix operations quickly. `
repeated 20 times, with fixed literal suffixes. Native tokenized this text in-process; oMLX tokenized through `/v1/completions`.

Native metrics are in-process generation-loop medians. oMLX metrics are live-server HTTP measurements. oMLX reported realistic PP/TTFT/cached-token fields, but streamed completions arrived as a full completion chunk, so its `generation_tokens_per_second` field was derived from near-zero `generation_duration` and was rejected as invalid. The oMLX TG value below is therefore HTTP-observed completion throughput for the same request shape, not a pure generation-loop decode metric.

## Results

| Metric | Native MLXServe | oMLX | Native vs oMLX |
| --- | ---: | ---: | ---: |
| Prefill PP | 23473.99 tok/s | 4122.79 tok/s | 5.69x faster |
| Decode TG | 271.21 tok/s | 146.45 tok/s | 1.85x faster |
| TTFT | 9.78 ms | 50.00 ms | 80.44% lower |
| Cache speedup | 0.97x | 1.07x | roughly even |
| Cached tokens observed | 256-token hot prefix hits | 256-token hot prefix hits | equivalent evidence |

| Concurrency | Native Throughput | oMLX HTTP Throughput | Native vs oMLX |
| ---: | ---: | ---: | ---: |
| 1 | 252.86 tok/s | 146.45 tok/s | 1.73x faster |
| 2 | 331.25 tok/s | 162.22 tok/s | 2.04x faster |
| 4 | 280.63 tok/s | 158.20 tok/s | 1.77x faster |
| 8 | 295.98 tok/s | 152.57 tok/s | 1.94x faster |

## Caveats

- Native is measured in-process; oMLX is measured through a live HTTP server.
- oMLX's public streaming endpoint aggregated the 16 generated tokens into one chunk for this model/prompt, so a clean generation-loop TG was not available through `/v1/completions` or `/v1/chat/completions`.
- Native and oMLX token counts differed slightly for the same prompt text: native PP prompt was 409 tokens, oMLX reported 401 prompt tokens. The text was identical; tokenizer/template handling differs by stack.
- Cache speedup was modest on both stacks because the shared prefix hit covered one 256-token block while the full text prompt was about 400 tokens, so each warm path still had suffix/remaining-token work.

## Verdict

Functional parity is complete through M5. For this M6 live-serving benchmark, native MLXServe meets or exceeds observed oMLX serving-path performance: PP is 5.69x faster, TTFT is 80.44% lower, and concurrency throughput is 1.73x to 2.04x faster across the tested range.

Strict engine-to-engine TG parity against oMLX remains caveated because oMLX did not expose per-token decode timing through the live HTTP endpoints in this run. Against HTTP-observed completion throughput, native TG is 1.85x faster.
