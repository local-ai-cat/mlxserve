# M8b — Speech engine registry: design of record

Decided 2026-07-05 with the operator. Context: `docs-site/pages/mlxserve-omlx-parity.mdx`
(Local-AI-Chat repo) — audio gets done properly in-engine, not deferred to cat-integration.

## Shape

One registry, engines as peer adapters, consumers on top:

- **`MLXServeSpeech`** (new dependency-free target): the protocol layer —
  `SpeechEngineAdapter`, `SpeechStreamSession`, capabilities, `SpeechEngineRegistry`.
- **Adapter packages/targets** (one per engine, each owning its heavy dependency):
  WhisperKit (ANE) first; Parakeet/FluidAudio (ANE), whisper.cpp (CPU), MLX-ASR (GPU)
  later. App-side engines are **wrapped, not rewritten** — rewrites allowed only after
  the conformance battery is green against the wrapped version.
- **Consumers**: the existing `/v1/audio/transcriptions` route (via a registry-backed
  `AudioTranscriptionBackend` bridge), the app UI picker, the studio `TranscriptionTap`,
  and the M8c consensus dispatcher.

## The studio contract (why streaming is first-class from day one)

The Local-AI-Chat streaming/recording plan (Phase 2: audio lane + session clock)
consumes this registry through a `TranscriptionTap`. Its requirements are protocol
requirements:

1. **Timestamped PCM in, engine-relative word times out** — the tap maps words back
   onto the session clock; a session that returns bare text is useless to it.
2. **Partial → final segments with stable identity** — captions burn partials; the
   recording bundle wants finals; both from one session.
3. **Latency stats** — per-session `arrival − capture` style stats the studio's delay
   budget can consume.
4. **Silicon placement as capability + preference** — during a live stream the GPU is
   compositing; transcription must be steerable to ANE engines.

## Capabilities model

`SpeechEngineCapabilities`: `supportsFileTranscription`, `supportsStreaming`,
`silicon` (.ane/.gpu/.cpu/.multiple), `wordTimestamps`, `confidence`, `languages`
(nil = open). Lane mismatches are solved **inside adapters** (stream-only engines
bridge file→PCM-push; file-only engines buffer), never by consumers.

## Lifecycle (pool citizenship — phase 2)

Adapters expose `loadModel`/`unloadModel`/`loadedFootprint` so speech models can
appear beside LLMs as `audio_stt` model types (omlx's taxonomy) in `/v1/models`
and `/v1/models/status`. Load/unload routes delegate to the owning speech adapter,
but speech models are deliberately not enrolled in the LLM `EnginePool` LRU or
memory-ceiling eviction yet; that needs a dedicated cross-engine admission policy.
`loadedFootprint` is reported as `actual_size`. For CoreML/ANE-backed adapters this
is an approximate adapter working-set signal, not a precise per-model MLX tensor
allocation. When multiple models are loaded inside one adapter, the adapter
footprint is attributed once to the first loaded model in that adapter's status
rows so aggregate status memory does not double count the shared working set.

## Phasing

- **M8b-1 (this branch):** protocol + registry + registry-backed HTTP bridge +
  WhisperKit adapter (file lane real; stream session implemented via WhisperKit's
  streaming transcribe) + conformance test suite with a fake adapter + gated live
  tests. `/v1/audio/transcriptions` stops 501ing when an adapter is registered.
- **M8b-2:** pool citizenship (`audio_stt`), Parakeet adapter (wraps the app's
  FluidAudio engine at cat-integration), `models/status` integration.
- **M8b-3 — scheduling & "batching" (ruling 2026-07-06):** audio batching is
  SCHEDULING, not token batching — Whisper-class/ANE engines can't merge audios
  into one forward pass (fixed-shape CoreML), so short-while-long comes from
  the scheduler. Three mechanisms, layered:
  1. **Window-quantum scheduler** — the pipeline turn gate becomes a priority
     queue whose quantum is ONE decode window (~30s audio, ~1-2s ANE wall),
     not one request. Long jobs yield between windows (the sliding-window
     `trimmedSeconds` machinery already knows how to resume); interactive jobs
     jump the queue. Interactive latency = current window, not whole file.
  2. **Priority classes** — every request carries QoS (`interactive` |
     `background`); surface bindings map to it naturally; HTTP gains a
     `priority` param on `/v1/audio/transcriptions`.
  3. **Busy-aware resolution** — adapters expose queue-depth/busy to
     `resolveCandidates`; interactive jobs route to an idle engine on other
     silicon (Parakeet/SpeechAnalyzer) instead of waiting. Same signal M8c
     consensus needs.
  Per-adapter contract additions: `maxConcurrentSessions` (multi-instance
  within the pool memory budget), `preemptionQuantum`, `supportsTrueBatch`
  (MLX-ASR on GPU later — the only lane where real batch inference exists).
- **M8c:** consensus dispatcher — fan one input to N adapters (ANE+GPU+CPU run
  truly parallel), fuse: confidence pick → ROVER voting → LLM fusion via the
  server's constrained-JSON decode. Post-hoc (raw-first recordings) before live.

## Surface bindings (ruling 2026-07-06)

Engine selection is **per app surface**, layered ABOVE the registry:

- **Registry layer (headless — never changes for this):** always the full
  multi-engine catalog. API callers and services name `engineID:model`
  explicitly or pass capability preferences and get candidate resolution +
  fallback. No pinning lives here — a surface config can never remove options
  from the headless API.
- **Surface-binding layer (app-side):** each surface (translate-live,
  translate-conversation, transcribe-file, global transcription, studio tap, …)
  gets a `SurfaceEngineConfig` — a pinned `engineID` or a preference set
  ("stream-capable, prefer ANE"). Backend/settings storage first; UI exposure
  later is a per-surface picker reading the same config. At call time the
  binding resolves to registry preferences — sugar over `resolveCandidates`,
  never a parallel path. Translate screens are the motivating case: they need
  specific engines per screen without affecting any other consumer.

## Out of scope, permanently

Capture. Mic ownership, AVAudioSession, permissions stay app-side. The registry's
contract starts at "audio in" (file bytes or PCM buffers), ends at "text out".
