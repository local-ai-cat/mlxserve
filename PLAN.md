# MLXServe — Implementation Plan (for autonomous execution)

**Repo:** standalone Swift package `mlxserve` (Apache 2.0 + NOTICE crediting `jundot/omlx` + `vllm-mlx`).
**Mission:** port omlx's two parked serving capabilities to native Swift on mlx-swift, verified end-to-end by `swift test` in isolation (no host app):
1. **Continuous / batched decode** — decode N concurrent sequences in one forward pass (omlx serving-plan Phase 1/2).
2. **Tiered prefix KV cache** — chain-hash disk-backed prefix reuse, hot RAM + cold SSD tiers, restart-survivable (Phase 3).

**Reference source (READ, don't guess):** `/Users/timapple/Documents/Guest/omlx/omlx/` (Python). Build target dep: `/Users/timapple/Documents/Guest/mlx-swift-lm` (a.k.a. mlx-swift-examples; `MLXLMCommon`/`MLXLLM`), on `ml-explore/mlx-swift` 0.31.4+.

---

## 0. Ground truth from the deep-read (do not re-derive — it's expensive)

Two findings reshape the naive plan:

- **A1 — omlx "paging" is NOT GPU-resident paging; there is NO custom Metal attention kernel.** A cache block holds *only metadata* (SHA-256 chain-hash + ref-count). KV bytes live as **one safetensors file per block** (MLXServe block_size=256, §0.5) across a hot-RAM tier + cold-SSD tier. On a request, omlx chain-hashes the prompt, finds the longest cached prefix, `concatenate`s those blocks into a **contiguous** K/V, and feeds the **standard** `scaledDotProductAttention`. → Track A reuses mlx-swift's existing attention; the port is a cache/prefix/IO system, not a kernel.
- **B1 — the ragged mask / batched forward is NOT in omlx; it's in mlx-lm's `GenerationBatch`/`BatchGenerator`.** omlx only *drives* it (insert/next_generated/remove/extract_cache/filter/extend). mlx-swift has single-sequence generation only. → the batched forward + ragged causal mask + per-layer batch cache is **net-new Swift** and the hardest single piece.

**Consequence for structure:** the two tracks are parallelizable and each ships "simple correct" first:
- **Track B** ships with cache collaborators = `nil` (BatchGenerator owns KV) → prove batched decode correctness first.
- **Track A** develops against single-sequence generation (reconstruct a cache, continue, compare to fresh prefill) → prove prefix reuse correctness first.
- They meet at the **collaborator protocol** (`PrefixKVStore`, §Seam) wired into the scheduler at admit/finish.

The `PrefixKVStore` seam is defined over **serialized per-layer `(state: [MLXArray], metaState: [String], className: String)` tuples**, NOT live cache objects — because omlx does a full `mx.eval` on the producing thread before any cross-thread handoff (MLX streams are thread-local).

---

## 0.5 Reference implementations & prior art (READ these, port our own way)

We port our own Swift, following mlx-lm's *mechanism* + omlx's *serving approach*. Division:
- **mlx-lm (Python, local `~/Documents/Guest/mlx-lm` @ 0.31.3)** = the batched-decode + prompt-cache *mechanism*. THE reference for Track B's hard piece:
  - **`create_causal_mask` (`base.py:24-42`)** — the ragged mask is a ~15-line pure function (boolean mask from scalar `offset` + per-row `leftPadding` vector; `-inf` fill at SDPA). The `N==1`→no-mask fast path lives in `create_attention_mask` (`base.py:45-55`) and is valid ONLY when `leftPadding.max()==0`; `BatchKVCache.make_mask` (`cache.py:1011`) ALWAYS builds a mask (see M1 ⚠). **Port faithfully, but gate the fast path on left-padding.**
  - **`BatchKVCache` (`cache.py:912-1130`)** — batch cache built by *merging* N single-row `KVCache`s (right-justify into `max_length`, derive left-pad from length gaps, `merge` `:1088`); `extract(idx)` `:1080` peels a finished row back to a contiguous normal cache.
  - **`GenerationBatch._step` (`generate.py:1320-1378`)** — `(B,1)` forward, `logsumexp` normalize, per-row sampler, `async_eval(next)…eval(current)` double-buffer.
  - **`save/load_prompt_cache` + `state`/`meta_state` (`cache.py:15-85,127-175`)** — Track A serialization primitive: one `.safetensors` with tensor payload + string metadata (class names + meta_state).
  - Provenance PRs: **#443** (original batch API), **#1072** (the clean refactor we're reading), **#1359** (return_logprobs).
- **omlx (Python, local `~/Documents/Guest/omlx`, Apache-2.0)** = the *serving architecture*: scheduler, chain-hash blocks, hot RAM + cold SSD tiers, CoW prefix sharing, the driving contract. (Block size: omlx's low-level cache manager defaults to 64 (`paged_cache.py:503`) but its **serving scheduler wires 256** (`scheduler.py:1309`), as does mlx-engine — **MLXServe default = 256**, configurable.) Our primary architectural model (see Appendices A/B for file:line maps).
- **lmstudio-ai/mlx-engine PR #326 (MERGED, MIT, Python)** = closest sibling — disk-chunked KV (256-token chain-hash blocks) + continuous batching + dedicated cache-I/O thread + chunked prefix restoration. A **design cross-check** for Track A's SSD tier (NOT a Swift dependency — it's Python). Modules to skim: `chunks.py`, `records.py`, `blob_store.py`+`disk_budget.py`, `coordinator.py`+`cache_io_thread.py`.
- **Signal — GPU-paged route was rejected upstream:** mlx-lm **#610** and **#629** (full continuous-batching-with-paged-KV) were both **CLOSED, not merged**. Corroborates our disk-backed / reconstruct-contiguous approach (finding A1) over vLLM-style GPU paging.
- **Swift ecosystem: EMPTY.** No batched/continuous generation in mlx-swift or mlx-swift-examples (confirmed). We are building the first Swift batching layer → author from scratch following mlx-lm, nothing to fork.
- **Convergent upstream cache work (context):** mlx-lm **#1218** (opt-in disk-backed L2 prompt cache), **#1283** (per-request prompt cache files) — the ecosystem is converging on exactly this.
- **Known-limitation references (design around):** **#980** (prefix cache broken for hybrid/sliding-window upstream — reinforces our full-attention-only v1), **#903** (Qwen3.5 cache-miss bug), **#1251** (`bool('False')==True` corrupts rotating cache serialization — hazard H7).

**LICENSE (corrected):** MLXServe is **Apache-2.0** (a port of Apache-2.0 omlx = derivative work → same license; this is the clean case, NOT a conflict). Referencing/porting patterns from **MIT** sources (mlx-lm, mlx-engine, mlx-swift) *into* an Apache-2.0 project is fully permitted (MIT is permissive). `NOTICE` credits jundot/omlx + vllm-mlx (Apache attribution) and mlx-lm/mlx-engine (courtesy). There is **no** license blocker.

## 1. Non-goals (scope fence — an autonomous run MUST NOT wander into these)

- ❌ **Sliding-window / rotating caches** (Gemma3-class, gpt-oss). Requires `RotatingKVCacheHandler` + supersede-on-extend tip-lineage + boundary snapshots — the single biggest numerical-subtlety cluster. **Full-attention `KVCache` models ONLY for v1.**
- ❌ **MTP / speculative decode** (omlx `batch_generator.py` patch). Throughput opt, net-negative on M1/M2, couples to model-specific MTP heads.
- ❌ VLM / OCR / embeddings / rerankers / TTS / STT.
- ❌ **GGUF / llama.cpp.** Separate ggml engine. Only negative constraint: keep the scheduler engine-agnostic so a llama.cpp engine could ride it later. Do NOT touch llama.cpp.
- ❌ Cross-impl on-disk cache reuse with the Python omlx cache dir. Define a **fresh canonical token encoding** for the chain hash; don't reproduce Python `repr`.
- ❌ Wiring into the Local AI Chat app. That's a later step (SPM pin by SHA behind `LocalAITextGenerating` + `FEATURE_MLXSERVE`). This repo stays standalone + `swift test`-only.
- ❌ The Python sidecar. Serving topology is decided (native primary + omlx/mlx-engine sidecar as the parked fallback for big coding models / heavy multi-session — see scoping doc). MLXServe is the **native** engine only; the sidecar is a separate `LocalAITextGenerating` backend built later, not here.

---

## 2. Repo layout

```
mlxserve/
  Package.swift                 # deps: mlx-swift-lm (MLXLMCommon, MLXLLM) → transitively mlx-swift
  LICENSE                       # Apache 2.0
  NOTICE                        # credits jundot/omlx + vllm-mlx (derivative work)
  README.md                     # what it is; "not affiliated with omlx"
  PLAN.md                       # this file (the executable spec)
  Sources/MLXServe/
    TrackB/  (batched decode + scheduler)   # §Track B file map
    TrackA/  (prefix + SSD cache)           # §Track A file map
    Seam/    PrefixKVStore.swift            # the collaborator protocol
    Engine/  MLXServeEngine.swift           # public façade (generate / stream)
  Tests/MLXServeTests/
    Fixtures/                   # golden JSON from Python omlx/mlx-lm (committed)
    Support/                    # model-load helpers, tolerance asserts, fixture loader
    *Tests.swift
```

**Pinned test model:** SmolLM-135M-4bit (bundled at `Local AI Chat/BundledModels/mlc-chat-SmolLM-135M-4bit/` — copy into the repo's test resources or reference by path env var `MLXSERVE_TEST_MODEL`). Small, fast, full-attention. Golden fixtures generated from it.

---

## 3. Go/No-Go gates to run in M0 (could force reimplementation — do FIRST)

Before committing to the port, verify two mlx-swift capabilities. If either fails, the plan adapts (documented fallback), it does not block:

- **G1 — bf16 bit-reinterpret view.** omlx stores bf16 by viewing as uint16 (`paged_ssd_cache.py:338-341,365-366`). Confirm mlx-swift can bit-reinterpret an `MLXArray` bf16↔uint16 WITHOUT a value cast (need raw bytes). Fallback: write via `asData()` raw bytes + manual dtype tag.
- **G2 — safetensors loader returns `__metadata__`.** omlx reads per-block `__metadata__` (format version, hashes, layer types) via `mx.load(..., return_metadata=True)` (`paged_ssd_cache.py:2603,1503`). Confirm mlx-swift `loadArrays`/`loadArraysAndMetadata` returns the string metadata dict. Fallback: implement a pure-Swift safetensors reader (the *writer* is already pure-Python/no-mx in omlx and trivially portable — `paged_ssd_cache.py:370-421`).

---

## 4. Milestones (dependency-ordered; each ends GREEN + committed)

> Execution rule for the autonomous run: **build + `swift test` after every milestone; commit per milestone with a conventional message; never leave the tree red.** M1/M2 (Track B) and M3 (Track A) are INDEPENDENT — progress whichever is unblocked. If a Go/No-Go gate or a hazard blocks a milestone, document it in `PLAN.md` under "Blocked", and move to the next independent milestone rather than stalling.

### M0 — Scaffold + gates + single-seq baseline
- `Package.swift`, LICENSE, NOTICE, README. Depends on mlx-swift-lm; `swift build` clean.
- **Test-model bootstrapping (autonomous-run critical):** resolve the pinned model to a guaranteed-LOCAL path (copy SmolLM-135M-4bit into repo test resources, or an `MLXSERVE_TEST_MODEL` env path). Tests must **skip cleanly if the model is absent — NEVER hang on a network download** (mlx-swift-lm's default factory pulls a remote id, `LLMModelFactory.swift:94`; do not rely on it). Golden fixtures are **pre-generated + committed**; `swift test` never invokes Python.
- **Verify-first:** confirm mlx-swift-lm truly has NO reusable batch/left-padding cache before authoring M1 from scratch (survey says single-seq only — but check; if a batch cache exists, extend it instead).
- Test support: load the local model via `loadWeights`/`ModelContainer` (survey §2.4); single-sequence greedy generate; assert against a committed golden fixture.
- Run **G1/G2** (§3); record results in PLAN.md.
- **Gate:** `swift test` green (or cleanly SKIPPED with a loud message if model absent); baseline single-seq generation matches the committed fixture.

### M1 — Track B: STATIC batched decode (THE hard piece)
Port mlx-lm's batched decode faithfully (reference: `~/Documents/Guest/mlx-lm`). **Static batching only** — fill a batch, run to completion, drop it; continuous batching (dynamic insert/remove) is M1.5. Files: `TrackB/BatchGenerator.swift`, `TrackB/BatchCache.swift`, `TrackB/CausalMask.swift`.
- **Prefill is per-row SERIAL, then merge — do NOT batch-prefill ragged prompts (Codex BLOCKER-avoidance).** Prefill each prompt at its true length via the existing single-sequence path (own `KVCache`, no padding), then `BatchKVCache.merge` the N single-row caches (right-justify, derive left-pad from length gaps). Only **decode** is batched. This is *why* `prepare`/`finalize`/`dynamic_roll` are genuinely unneeded — NOT because we left-pad prompts up front (that would poison the KV cache / shift RoPE positions and fail the gate for the wrong reason, `generate.py:1142-1170`).
- **`CausalMask.swift`** — port `create_causal_mask` (`mlx-lm base.py:24-42`): boolean mask from scalar `offset: Int` + per-row `leftPadding: MLXArray` (`leftPadding <= rinds` masks each row's pad columns; `expand_dims(...,(1,2,3))` broadcast onto batch axis). Boolean; `-inf` fill at the SDPA call site. ⚠ **The `N==1`→no-mask fast path is ONLY valid when `leftPadding.max()==0`** (single sequence / uniform batch). A ragged batched *decode* step (N==1 per row) STILL needs the mask or rows attend to their own left-pad columns and silently diverge (fast path lives in `create_attention_mask` base.py:45-55, NOT the batch cache — `BatchKVCache.make_mask` cache.py:1011 always builds it). Gate the fast path on `leftPadding.max()==0`.
- **`BatchCache.swift`** — port `BatchKVCache` (`cache.py:912-1130`): own `keys`/`values`/`leftPadding`/`offset` + scalar `_idx`; `merge` (`:1088`); `extract(idx)` (contiguous single-row peel dropping left-pad, `:1080`).
- **`_step` loop** (`generate.py:1320-1378`): `(B,1)` forward, `logsumexp` normalize, whole-batch argmax default, `async_eval(next)…eval(current)` double-buffer.
- **Gate — batch-invariance (margin-gated, NOT absolute token equality):** batched logits == serial logits within **dtype-aware tolerance**; assert token equality ONLY where the serial top-1/top-2 margin comfortably exceeds the observed max logit error (curate fixtures with wide margins — MLX GPU reductions are non-associative, so absolute token-for-token WILL false-fail on near-ties and burn autonomous-run hours). Test batch sizes {2,4,8} + ragged prompt lengths.

### M1.5 — Track B: continuous batching (dynamic rows)
Only after M1 is green. Add `filter` (fancy-index + re-minimize padding, `cache.py:1016`), `extend`, and `insert`/`remove` row management + `Response {uid, token, finishReason, logprobs?}` + per-row sampler (greedy iff temp==0). This is what lets requests join/leave a running batch mid-flight (required by the M2 scheduler).
- **Gate:** a row can be added and a finished row removed mid-batch; surviving rows still pass batch-invariance vs their solo runs.

### M2 — Track B: scheduler + engine
**Depends on M1.5** (its cancel/finish gate needs mid-batch `remove`/`filter`). Port omlx `Scheduler` (spec-B §3/§4) with collaborators = `nil`. Files: `TrackB/Scheduler.swift` (+ `+Prefill`, `+Cancellation`), `TrackB/Request.swift`, `TrackB/OutputCollector.swift`, `TrackB/Sampling.swift`, `Engine/MLXServeEngine.swift`.
- `waiting`/`running` queues, FCFS admission, `maxConcurrentRequests` cap.
- **External prefill** (`_do_external_prefill`, `scheduler.py:2850`): run model on `tokens[0:N-1]` outside the batch; **withhold last token**, hand to `insert`.
- `step()` pump (spec-B §3.4): drain aborts → admit → one `next_generated()` decode step → detokenize/finish-detect → cleanup.
- **Cancellation** (spec-B §4): enqueue-then-apply-on-step-thread; **`synchronize(stream)` barrier BEFORE removing a row** (hazard H3).
- **Backpressure:** queue depth ≥ `max(maxConcurrent*4, 32)` → reject (503-style error); expose `Retry-After`-style hint.
- Single serial actor owns `step()`; all MLX work on one stream.
- **Gate:** via the engine API, submit {1,4,8} concurrent generations → all complete correctly (each == its M1 solo result); a mid-flight cancel frees its row and doesn't corrupt siblings; queue-full rejects cleanly.

### M3 — Track A: prefix cache, HOT tier only (parallel with M1/M2)
Port the metadata/prefix core (spec-A §1–3,§5), RAM tier only, **KVCache family only**. Files: `TrackA/KVCacheBlock.swift`, `FreeBlockQueue.swift`, `BlockHashIndex.swift`, `BlockHashing.swift`, `PagedCacheManager.swift`, `CacheTypeHandlers.swift` (KVCache + Default only), `BlockAwarePrefixCache.swift`.
- `CacheBlock` (`final class`, intrusive free-list or index-based), `BlockTable`, `FreeBlockQueue` (O(1)).
- **Chain hash** (`compute_block_hash`, `paged_cache.py:78-119`) over a **fresh canonical token encoding** (NOT Python repr). block_size = 256 (MLXServe default; see §0.5).
- **Hot block-PAYLOAD store (REQUIRED — Codex MAJOR).** A `CacheBlock` holds only metadata (hash + ref-count); the KV bytes live in a *separate* store (`paged_cache.py:135-163`; omlx keeps them in the SSD manager). M3 must include an in-RAM `blockHash → (keys, values)` payload store so `reconstruct` has tensors to concatenate. **M4 puts the SSD cold tier UNDER this same store.** Without it there is nothing to reconstruct and the M3 gate cannot pass.
- `PagedCacheManager`: alloc/ref-count/`free`/`touch`/elastic-grow; `fork_block_table` (O(1) ref-bump) + `get_blocks_for_generation` CoW trigger (metadata-only copy).
- `BlockAwarePrefixCache`: `fetchCache` (longest-prefix) → `reconstructCache` (concat blocks along axis 2 → contiguous cache) → `storeCache`. Axis-info-driven generic slicer (`CacheStateAxisInfo`, `type_handlers.py:56-74`).
- **Serialization primitive = mlx-lm's `state`/`meta_state` contract** (`cache.py:127-175`), not a bespoke format: a `KVCache` block serializes as `(keys, values)` tensors + a `metaState` string (store `offset` explicitly). Follow `save/load_prompt_cache` (`cache.py:43-85`) for the safetensors-with-string-metadata shape; diverge by using explicit keyed tensor names (`layer.{i}.keys`) instead of `tree_flatten` dotted auto-keys, and persist the cache class name for load dispatch. (This is the M4 on-disk format too.)
- **Gate — cache-tier-invariance (tol + margin-gated):** prefill prompt P **(length a MULTIPLE of block_size=256 — omlx never caches the trailing partial block, `prefix_cache.py:500-518`; a non-aligned P silently caches fewer tokens and the test asserts the wrong thing)**, store; new request P+suffix reconstructs P's cache and continues → first-continuation logits **within dtype-aware tol** of a fresh full prefill of P+suffix; token equality only where margin exceeds max error. Divergent-branch test: P+X and P+Y share P's blocks, diverge after.

### M4 — Track A: cold SSD tier
Port `PagedSSDCacheManager` (spec-A §4). Files: `TrackA/SafetensorsBlockIO.swift`, `PagedSSDCacheManager.swift` (split `SSDCacheIndex`/`HotCacheBudget`/`SSDWriter` if large).
- Pure-Swift per-block safetensors writer (`_write_safetensors_no_mx`, `:370-421`) — **NO Metal/MLXArray touch off the eval thread** (hazard H2); `mx.eval` on producing thread, extract raw bytes, write on background actor.
- bf16-as-uint16 round-trip (per G1). Atomic temp-file + rename.
- Hot-cache LRU write-back → SSD on overflow; size enforcement; **scan-on-start** rebuild of index from disk (restart survival); per-model compatibility check (shared dir, multi-model).
- **Gate — restart survival:** bf16/tensor bytes **exact** round-trip through safetensors (byte-level, deterministic); then warm cache for block-aligned P, construct a *fresh* manager on the same dir, reconstruct P → continuation logits within tol (as M3's gate). The byte round-trip is exact; the generation comparison is tol/margin-gated.

### M5 — Integration: wire Track A into Track B
Define `Seam/PrefixKVStore.swift` (spec-B §5.1/5.2 consumer protocol). Wire into scheduler admit path: `fetchCache → preloadBlocks → reconstructCache` (produces `request.promptCache`, prefill only the remaining suffix); boundary snapshots every `block_size` tokens during prefill; `storeCache` async at finish; `release`/`clearEntry` on abort.
- **Cache-hit rows merge into the batch via the SAME path as cache-miss rows (Codex MAJOR).** A cache-hit produces a populated single-row cache (from `reconstructCache` + suffix prefill); a cache-miss produces a populated single-row cache (from full serial prefill). Both are single-row `KVCache`s → both go through `BatchKVCache.merge` (`cache.py:1088`) identically. Do NOT create two cache shapes or let cache-hit rows skip merge. (This is why prefill stays per-row/serial before merge — M1.)
- N-vs-(N-1) kickoff-token adjustment on exact prefix hit (`scheduler.py:5843-5891`).
- **Gate:** concurrent requests sharing a common prefix reuse cache AND still pass batch-invariance + cache-tier-invariance together.

### M6 — Benchmark harness (comparison to omlx)
`swift bench` target (NOT a pass/fail test — statistical, medians over N runs + warmup). Metrics: prefill tok/s (PP), decode tok/s (TG), TTFT, **cold-prefill vs warm-restore speedup**, **throughput-vs-concurrency {1,2,4,8}**, peak RAM. Same model/prompts as omlx's `/admin` benchmark for apples-to-apples. Emit a markdown report.

### M7+ — Deferred (documented, not built)
Rotating/sliding-window family (+supersede-on-extend + boundary snapshots), MTP, cat integration. Each is a separate future effort with its own risk profile.

---

## 5. Hazards & invariant RULES (violating these = SIGABRT or silent corruption)

- **H1 — MLX stream/thread locality.** All model/cache MLX work on ONE stream, owned by the scheduler's serial actor. Never materialize a cache array on the wrong thread.
- **H2 — eval-before-handoff.** Before any array crosses to a worker (SSD writer), `mx.eval` it on the producing thread, then hand raw bytes only. The SSD writer touches ZERO MLXArray/Metal.
- **H3 — sync-before-remove.** `synchronize(stream)` MUST precede removing a row from the batch cache (`scheduler.py:6829`), else Metal command-buffer underflow. Same for the deferred async-remove drain.
- **H4 — fresh canonical hash encoding.** Don't chase Python-repr byte compatibility; own the cache dir + encoding. Isolate caches by model name in the hash (`paged_cache.py:101-103`).
- **H5 — ref-count accounting.** fork/CoW/reconstruct-truncation all mutate ref-counts + the block table in place (`prefix_cache.py:1872-1939`). Off-by-one leaks or double-frees blocks. Cover with a leak-detector test (after N requests finish, allocated_blocks back to baseline).
- **H6 — never mix** cache-hit vs miss, or VLM vs text, in one prefill batch (omlx homogeneity gates) — N/A while VLM is out of scope, but keep the cache-hit/miss gate.
- **H7 — explicit bool (de)serialization.** safetensors metadata is string-only; `Bool("False")` is truthy in many languages (upstream bug mlx-lm #1251 corrupted `BatchRotatingKVCache`). Encode/decode bools explicitly (`"1"/"0"` or an int) in any `metaState` round-trip; never round-trip via a naive string→bool.

---

## 6. Test strategy (the whole point of standalone)

- **Invariant gates** (the real gates): batch-invariance (M1), cache-tier-invariance (M3), restart survival (M4), block-leak (H5). **Two kinds of assertion — don't conflate:** (i) genuinely EXACT — safetensors byte round-trip, block-leak accounting (allocated_blocks back to baseline after N finishes); (ii) tol + **margin-gated** token equality — anything comparing model *generations* (batched-vs-serial, reconstructed-vs-fresh), because MLX GPU reductions are non-associative. Curate golden prompts with wide top-1/top-2 margins so token equality is stable.
- **Golden fixtures vs Python omlx/mlx-lm**: fixed (prompt, seed, model) → snapshot logits + token ids to `Tests/Fixtures/*.json` (committed). Assert: exact match on greedy token ids for well-separated logits; tolerance (max-abs-diff/cosine) on raw logit vectors. A fixture-generation script (`scripts/gen_fixtures.py`, runs against `~/Documents/Guest/omlx` or mlx-lm) is committed but fixtures are pre-generated so `swift test` needs no Python.
- **Performance** (M6): separate, statistical, not CI-gated.

## Gate results

- **M0 validation, 2026-07-03 on branch `impl/native`: PASS.** `swift test` with `MLXSERVE_TEST_MODEL=/Users/timapple/Library/Caches/models/mlx-community/Qwen3-0.6B-4bit` executes 4 tests with 0 failures on GPU. The test harness generates/copies the missing `mlx.metallib` into the SwiftPM XCTest bundle before the first MLX operation so command-line `swift test` can load MLX Metal shaders.
- **G1 — bf16 bit-reinterpret view: PASS.** Runtime probe confirms `MLXArray.view(dtype:stream:)` round-trips `uint16 -> bfloat16 -> uint16` byte-exact for `[0x3f80, 0x4000, 0xbf80]`.
- **G2 — safetensors metadata: PASS.** Runtime probe confirms `save(arrays:metadata:url:stream:)` and `loadArraysAndMetadata(url:stream:)` round-trip tensor payloads and string metadata exactly.
- **Model-dependent greedy baseline: PASS.** Local Qwen fixture `qwen3_0_6b_greedy_baseline.json` pins prompt `"The capital of France is"`, `maxTokens=8`, temperature 0, and token IDs `[151667, 198, 32313, 11, 279, 1196, 374, 10161]`.
- **Verify-first for M1:** `mlx-swift-lm` has reusable prompt-cache serialization (`savePromptCache`, `loadPromptCache`) and some left-padding metadata helpers on `ArraysCache`/`MambaCache`, but no reusable `BatchKVCache`, `GenerationBatch`, or `BatchGenerator` equivalent was found in Swift. M1 still needs a Swift batch decode/cache layer, reusing existing serialization/mask helpers where appropriate.

## Blocked

- No active M0 blockers after GPU validation.

---

## Appendix A — Track A porting map (prefix + SSD cache)  [omlx file:line preserved]

| Swift file | omlx source | Notes |
|---|---|---|
| `KVCacheBlock.swift` | `paged_cache.py:126-477` | `CacheBlock` (final class), `BlockHash`=`Data`, `BlockTable` value type |
| `FreeBlockQueue.swift` | `paged_cache.py:194-371` | O(1) intrusive DLL or index-based (dodge ARC cycles) |
| `BlockHashIndex.swift` | `paged_cache.py:378-438` | hash→block, promotes to multimap on collision |
| `BlockHashing.swift` | `paged_cache.py:44-119` | chain hash; **fresh canonical token encoding** |
| `PagedCacheManager.swift` | `paged_cache.py:484-1732` | alloc/ref/fork/CoW/prefix-lookup/elastic grow |
| `SafetensorsBlockIO.swift` | `paged_ssd_cache.py:259-421,1780-2311` | pure-Swift writer/reader, bf16 view, N-tuple markers |
| `PagedSSDCacheManager.swift` | `paged_ssd_cache.py:459-3590` | index + hot cache + budget + async writer + scan-on-start |
| `CacheTypeHandlers.swift` | `type_handlers.py`+`type_registry.py` | `CacheStateAxisInfo` generic slicer; KVCache + Default handlers only |
| `BlockAwarePrefixCache.swift` | `prefix_cache.py` | fetch/reconstruct/store/fork/release; minus rotating/arrays/cachelist |
| `CacheFactory.swift` | `factory.py`,`interface.py` | config-driven wiring |
| *(defer)* Rotating/Arrays/CacheList handlers, `boundary_snapshot_store.py:1-70`, TurboQuant | | with sliding-window phase |

Minimal viable Track A = rows 1–10 minus deferred branches → block paging + chain-hash prefix sharing + metadata-only CoW fork + hot→cold safetensors + restart survival.

Key numbers/paths: block_size **MLXServe default 256** (omlx cache-mgr default 64 `factory.py:42`, but serving wires 256 `scheduler.py:1309`), null block reserved id 0, SSD file = `<dir>/<hex[0]>/<hex>.safetensors` (16 subdirs), format version "3", cache dir env/config.

## Appendix B — Track B porting map (batched decode + scheduler)  [omlx file:line preserved]

| Swift file | omlx source | Notes |
|---|---|---|
| `BatchGenerator.swift` | mlx-lm `GenerationBatch`/`BatchGenerator` (NOT in omlx; consumer contract at `batch_generator.py`) | **net-new**: batched forward, ragged mask, left-pad, insert/next/remove/filter/extend, `Response` |
| `BatchCache.swift` | consumer view `batch_generator.py:688,719-731` | per-layer batch cache; row extract/merge/trim; offset; state/metaState serialize |
| `Request.swift` | `request.py` | Request/status/SamplingParams/RequestOutput; `__lt__` = (priority, arrival) |
| `OutputCollector.swift` | `output_collector.py` | async aggregating buffer + stream interval |
| `Scheduler.swift` | `scheduler.py:1406,9177,7431` | queues, `step()`, `_schedule_waiting`, `_process_batch_responses`, error-recovery envelope |
| `Scheduler+Prefill.swift` | `scheduler.py:2850,3883,3942,4130,4190` | external + chunked prefill; last-token-withheld insert |
| `Scheduler+Cancellation.swift` | `scheduler.py:6713,6731,6791` | enqueue→apply-on-step-thread; sync-before-remove (H3) |
| `SchedulerConfig.swift` | `scheduler.py:1294,1307,1337` | maxNumSeqs, chunkedPrefill flag, SchedulerOutput |
| `Sampling.swift` | `omlx/utils/sampling.py` via `scheduler.py:2507,2518` | temp/top_p/min_p/top_k/xtc + repetition/presence/frequency/suppress |
| `Engine/MLXServeEngine.swift` | `engine/batched.py`,`base.py` | model load, generate/stream, preflight, wraps Scheduler |
| `Seam/PrefixKVStore.swift` | `scheduler.py:5795-5821,8484,6863` consumer methods | fetch→preload→reconstruct at admit; store at finish; release/clear on abort; over serialized `(state,metaState,className)` |
| *(defer)* MTP/ | `batch_generator.py` (all) | speculative decode — out of scope |

Key contract facts: `prefill_batch_size` ALWAYS 1 (prefill external); `completion_batch_size` 32; EOS token NOT emitted; per-request sampler/processors passed at `insert`; capacity rejection = 503 (queue) / 400 (memory), no literal 429 in this layer.
