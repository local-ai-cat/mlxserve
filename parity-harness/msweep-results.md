# M-sweep results (per-model harness runs)

Verdicts: PASS = both serve, cells green · GAP = one side faulted (see detail) · FAIL = harness cells failed · SKIP = preflight/timeout.

| model | size | verdict | detail | wall | at |
|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 0.3 GB | PASS | .                                                                        [100%] | 5s | 2026-07-06 00:25 |
| Llama-3.2-1B-Instruct-4bit | 0.7 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 00:25 |
| Qwen3-1.7B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 3s | 2026-07-06 00:25 |
| DeepSeek-R1-Distill-Qwen-1.5B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 00:25 |
| Qwen2-VL-2B-Instruct-4bit | 1.2 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 00:25 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | PASS | .                                                                        [100%] | 5s | 2026-07-06 00:25 |
| Qwen3.5-4B-MLX-4bit | 2.9 GB | PASS | .                                                                        [100%] | 7s | 2026-07-06 00:26 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | PASS | .                                                                        [100%] | 12s | 2026-07-06 00:26 |
| DeepSeek-R1-Distill-Qwen-7B-4bit | 4.0 GB | PASS | .                                                                        [100%] | 13s | 2026-07-06 00:26 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | PASS | .                                                                        [100%] | 8s | 2026-07-06 00:26 |
| gemma-4-E4B-it-qat-4bit | 6.4 GB | PASS | .                                                                        [100%] | 9s | 2026-07-06 00:43 |
| Ornith-1.0-9B-6bit | 7.7 GB | PASS | .                                                                        [100%] | 8s | 2026-07-06 00:43 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | SKIP | preflight: free 37GB < needed 39GB | 0s | 2026-07-06 00:43 |
| Qwen3.6-27B-4bit | 15.0 GB | SKIP | preflight: free 36GB < needed 48GB | 0s | 2026-07-06 00:43 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | SKIP | preflight: free 36GB < needed 48GB | 0s | 2026-07-06 00:43 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | SKIP | preflight: free 36GB < needed 50GB | 0s | 2026-07-06 00:43 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 36GB < needed 90GB | 0s | 2026-07-06 00:43 |
| Qwen3-0.6B-4bit | 0.3 GB | PASS | .                                                                        [100%] | 5s | 2026-07-06 01:12 |
| Llama-3.2-1B-Instruct-4bit | 0.7 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:12 |
| Qwen3-1.7B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:13 |
| DeepSeek-R1-Distill-Qwen-1.5B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:13 |
| Qwen2-VL-2B-Instruct-4bit | 1.2 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:13 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:13 |
| Qwen3.5-4B-MLX-4bit | 2.9 GB | PASS | .                                                                        [100%] | 6s | 2026-07-06 01:13 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | PASS | .                                                                        [100%] | 13s | 2026-07-06 01:13 |
| DeepSeek-R1-Distill-Qwen-7B-4bit | 4.0 GB | PASS | .                                                                        [100%] | 13s | 2026-07-06 01:14 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | PASS | .                                                                        [100%] | 8s | 2026-07-06 01:14 |
| Qwen3-0.6B-4bit | 0.3 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen3-0.6B-4bit] - Assert... | 5s | 2026-07-06 01:49 |
| Llama-3.2-1B-Instruct-4bit | 0.7 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Llama-3.2-1B-Instruct-4bit] | 3s | 2026-07-06 01:50 |
| Qwen3-1.7B-4bit | 0.9 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen3-1.7B-4bit] - Assert... | 3s | 2026-07-06 01:50 |
| DeepSeek-R1-Distill-Qwen-1.5B-4bit | 0.9 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[DeepSeek-R1-Distill-Qwen-1.5B-4bit] | 3s | 2026-07-06 01:50 |
| Qwen2-VL-2B-Instruct-4bit | 1.2 GB | PASS | .                                                                        [100%] | 3s | 2026-07-06 01:50 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Llama-3.2-3B-Instruct-4bit] | 3s | 2026-07-06 01:50 |
| Qwen3.5-4B-MLX-4bit | 2.9 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen3.5-4B-MLX-4bit] - As... | 5s | 2026-07-06 01:50 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen2.5-Coder-7B-Instruct-4bit] | 12s | 2026-07-06 01:51 |
| DeepSeek-R1-Distill-Qwen-7B-4bit | 4.0 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[DeepSeek-R1-Distill-Qwen-7B-4bit] | 12s | 2026-07-06 01:51 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | PASS | .                                                                        [100%] | 6s | 2026-07-06 01:51 |
| gemma-4-E4B-it-qat-4bit | 6.4 GB | PASS | .                                                                        [100%] | 8s | 2026-07-06 01:51 |
| Ornith-1.0-9B-6bit | 7.7 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Ornith-1.0-9B-6bit] - Ass... | 6s | 2026-07-06 01:51 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | SKIP | preflight: free 32GB < needed 39GB | 0s | 2026-07-06 01:52 |
| Qwen3.6-27B-4bit | 15.0 GB | SKIP | preflight: free 31GB < needed 48GB | 0s | 2026-07-06 01:52 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | SKIP | preflight: free 31GB < needed 48GB | 0s | 2026-07-06 01:52 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | SKIP | preflight: free 31GB < needed 50GB | 0s | 2026-07-06 01:52 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 30GB < needed 90GB | 0s | 2026-07-06 01:52 |
| Qwen3-0.6B-4bit | 0.3 GB | PASS | .                                                                        [100%] | 5s | 2026-07-06 01:54 |
| Llama-3.2-1B-Instruct-4bit | 0.7 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:54 |
| Qwen3-1.7B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:54 |
| DeepSeek-R1-Distill-Qwen-1.5B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:54 |
| Qwen2-VL-2B-Instruct-4bit | 1.2 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 01:54 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | PASS | .                                                                        [100%] | 7s | 2026-07-06 01:55 |
| Qwen3.5-4B-MLX-4bit | 2.9 GB | PASS | .                                                                        [100%] | 11s | 2026-07-06 01:55 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | PASS | .                                                                        [100%] | 13s | 2026-07-06 01:55 |
| DeepSeek-R1-Distill-Qwen-7B-4bit | 4.0 GB | PASS | .                                                                        [100%] | 15s | 2026-07-06 01:56 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | PASS | .                                                                        [100%] | 10s | 2026-07-06 01:56 |
| gemma-4-E4B-it-qat-4bit | 6.4 GB | PASS | .                                                                        [100%] | 10s | 2026-07-06 01:56 |
| Ornith-1.0-9B-6bit | 7.7 GB | PASS | .                                                                        [100%] | 10s | 2026-07-06 01:56 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | SKIP | preflight: free 36GB < needed 39GB | 0s | 2026-07-06 01:56 |
| Qwen3.6-27B-4bit | 15.0 GB | SKIP | preflight: free 35GB < needed 48GB | 0s | 2026-07-06 01:56 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | SKIP | preflight: free 33GB < needed 48GB | 0s | 2026-07-06 01:57 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | SKIP | preflight: free 33GB < needed 50GB | 0s | 2026-07-06 01:57 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 34GB < needed 90GB | 0s | 2026-07-06 01:57 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | SKIP | preflight: free 26GB < needed 39GB | 0s | 2026-07-06 02:14 |
| Qwen3.6-27B-4bit | 15.0 GB | SKIP | preflight: free 25GB < needed 48GB | 0s | 2026-07-06 02:14 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | SKIP | preflight: free 26GB < needed 48GB | 0s | 2026-07-06 02:14 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | SKIP | preflight: free 37GB < needed 50GB | 0s | 2026-07-06 02:15 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 36GB < needed 90GB | 0s | 2026-07-06 02:15 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | SKIP | preflight: free 26GB < needed 39GB | 0s | 2026-07-06 08:10 |
| Qwen3.6-27B-4bit | 15.0 GB | SKIP | preflight: free 26GB < needed 48GB | 0s | 2026-07-06 08:10 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | SKIP | preflight: free 26GB < needed 48GB | 0s | 2026-07-06 08:10 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | SKIP | preflight: free 26GB < needed 50GB | 0s | 2026-07-06 08:10 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 27GB < needed 90GB | 0s | 2026-07-06 08:10 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[gpt-oss-20b-MXFP4-Q8] - A... | 34s | 2026-07-06 08:12 |
| Qwen3.6-27B-4bit | 15.0 GB | PASS | .                                                                        [100%] | 17s | 2026-07-06 08:12 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | PASS | .                                                                        [100%] | 10s | 2026-07-06 08:12 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | PASS | .                                                                        [100%] | 21s | 2026-07-06 08:13 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 69GB < needed 90GB | 0s | 2026-07-06 08:13 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | PASS | .                                                                        [100%] | 20s | 2026-07-06 08:16 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | PASS | direct pytest cell (preflight bypassed with operator present; Chrome+Docker closed, ~73GB free) | 1 run | 2026-07-06 08:4x |
| Qwen3-0.6B-4bit | 0.3 GB | PASS | .                                                                        [100%] | 11s | 2026-07-06 08:26 |
| Qwen3-0.6B-4bit | 0.3 GB | PASS | .                                                                        [100%] | 6s | 2026-07-06 08:27 |
| Llama-3.2-1B-Instruct-4bit | 0.7 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Llama-3.2-1B-Instruct-4bit] | 9s | 2026-07-06 08:27 |
| Qwen3-1.7B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 08:27 |
| DeepSeek-R1-Distill-Qwen-1.5B-4bit | 0.9 GB | PASS | .                                                                        [100%] | 4s | 2026-07-06 08:27 |
| Qwen2-VL-2B-Instruct-4bit | 1.2 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen2-VL-2B-Instruct-4bit] | 6s | 2026-07-06 08:27 |
| Llama-3.2-3B-Instruct-4bit | 1.7 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Llama-3.2-3B-Instruct-4bit] | 6s | 2026-07-06 08:28 |
| Qwen3.5-4B-MLX-4bit | 2.9 GB | PASS | .                                                                        [100%] | 14s | 2026-07-06 08:28 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen2.5-Coder-7B-Instruct-4bit] | 14s | 2026-07-06 08:28 |
| DeepSeek-R1-Distill-Qwen-7B-4bit | 4.0 GB | PASS | .                                                                        [100%] | 14s | 2026-07-06 08:29 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[gemma-4-E2B-it-qat-4bit] | 12s | 2026-07-06 08:29 |
| gemma-4-E4B-it-qat-4bit | 6.4 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[gemma-4-E4B-it-qat-4bit] | 12s | 2026-07-06 08:29 |
| Ornith-1.0-9B-6bit | 7.7 GB | PASS | .                                                                        [100%] | 18s | 2026-07-06 08:30 |
| gpt-oss-20b-MXFP4-Q8 | 11.3 GB | PASS | .                                                                        [100%] | 18s | 2026-07-06 08:30 |
| Qwen3.6-27B-4bit | 15.0 GB | PASS | .                                                                        [100%] | 20s | 2026-07-06 08:30 |
| Qwen3.6-27B-MLX-4bit | 15.0 GB | PASS | .                                                                        [100%] | 23s | 2026-07-06 08:31 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | FAIL | FAILED tests/test_matrix.py::test_architecture_cell[Qwen3-Coder-30B-A3B-Instruct-4bit] | 18s | 2026-07-06 08:31 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 74GB < needed 90GB | 0s | 2026-07-06 08:31 |
| Qwen2.5-Coder-7B-Instruct-4bit | 4.0 GB | PASS | .                                                                        [100%] | 19s | 2026-07-06 08:37 |
| gemma-4-E2B-it-qat-4bit | 4.1 GB | PASS | .                                                                        [100%] | 21s | 2026-07-06 08:37 |
| gemma-4-E4B-it-qat-4bit | 6.4 GB | PASS | .                                                                        [100%] | 18s | 2026-07-06 08:38 |
| Qwen3-Coder-30B-A3B-Instruct-4bit | 16.0 GB | PASS | .                                                                        [100%] | 27s | 2026-07-06 08:38 |
| Qwen3.6-35B-A3B-4bit | 32.3 GB | SKIP | preflight: free 73GB < needed 90GB | 0s | 2026-07-06 08:38 |
