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
join the engine pool as an `audio_stt` model type (omlx's taxonomy) — status, LRU,
memory-guard uniform with LLMs. Phase 1 keeps lifecycle internal to the adapter;
the protocol carries the hooks from day one so pool wiring is additive.

## Phasing

- **M8b-1 (this branch):** protocol + registry + registry-backed HTTP bridge +
  WhisperKit adapter (file lane real; stream session implemented via WhisperKit's
  streaming transcribe) + conformance test suite with a fake adapter + gated live
  tests. `/v1/audio/transcriptions` stops 501ing when an adapter is registered.
- **M8b-2:** pool citizenship (`audio_stt`), Parakeet adapter (wraps the app's
  FluidAudio engine at cat-integration), `models/status` integration.
- **M8c:** consensus dispatcher — fan one input to N adapters (ANE+GPU+CPU run
  truly parallel), fuse: confidence pick → ROVER voting → LLM fusion via the
  server's constrained-JSON decode. Post-hoc (raw-first recordings) before live.

## Out of scope, permanently

Capture. Mic ownership, AVAudioSession, permissions stay app-side. The registry's
contract starts at "audio in" (file bytes or PCM buffers), ends at "text out".
