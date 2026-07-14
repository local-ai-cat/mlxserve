# Harness goal — audio axis (differential + concurrency)

You are a delegated worker under an automated orchestrator that will verify
your work. If blocked, emit `NEEDS_INPUT: <question>` and stop — never guess.

Repo: `/Users/timapple/Documents/Github/mlxserve-native-harness` (branch
`feat/omlx-parity-harness`), work in `parity-harness/`. The harness diffs
native MLXServe vs omlx (see `parity/servers.py`, `tests/test_matrix.py` for
patterns). Audio currently has ZERO harness coverage — build the axis.

## Facts you need
- Native binary: env `PARITY_NATIVE_BIN`
  (default `/private/tmp/parity-native-baseline/mlxserve-http`; the current
  tip build is `/Users/timapple/Documents/Github/mlxserve-native-audio/.build/release/mlxserve-http`).
  Native serves `POST /v1/audio/transcriptions` (multipart: `file`, `model`)
  ONLY when launched with `--whisperkit-models-dir PATH`. WhisperKit models
  live at `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml`
  (model id `openai_whisper-tiny`).
- omlx (env `PARITY_OMLX_BIN`, venv has mlx-audio 0.4.4) serves the same route
  with an MLX whisper model. Check the model store
  (`~/Library/Caches/models`) for an mlx whisper model omlx can use; ALSO
  check how omlx resolves STT model ids (read
  `/Users/timapple/Documents/Guest/omlx/omlx/api/audio_routes.py`). If no
  suitable model exists locally, the omlx side of the diff SKIPs loudly
  (pytest.skip with reason) — do NOT download anything large without asking.
- Test fixture: `Tests/MLXServeTests/Fixtures/test_speech.wav` in
  `/Users/timapple/Documents/Github/mlxserve-native-audio` — spoken content
  "The quick brown fox jumps over the lazy dog." Copy it into the harness
  (tests/fixtures/) rather than referencing across repos.

## Required tests (tests/test_audio.py, new axis "6. Audio")
1. **Native solo**: POST the wav to native → 200, non-empty `text`, normalized
   text (lowercase, strip punctuation) contains >=7 of the 9 expected words.
2. **Native 4-way concurrency**: 4 simultaneous transcription requests
   (ThreadPoolExecutor) → all 200 + pass the same normalized-text check, and
   all 4 texts identical to each other (same input → same output).
3. **Error semantics**: unknown model id → 404 with the available-models
   message; garbage (non-audio) file body → clean 4xx/5xx JSON error, server
   still healthy after (GET /v1/models 200).
4. **Differential vs omlx** (skip-loudly if omlx can't serve STT here): same
   wav to both → both 200; word-level agreement between the two normalized
   texts >= 80% (different engines — WhisperKit ANE vs mlx-audio — so demand
   semantic agreement, NOT byte equality). Record the cell in the report like
   other axes.
5. Register results via `parity.report.REPORT.record(...)` consistent with
   existing axes so the markdown report shows the audio axis.

## Server plumbing
Extend `parity/servers.py`: native launch passes
`--whisperkit-models-dir` when env `PARITY_WHISPERKIT_MODELS`
(default `~/Documents/huggingface/models/argmaxinc/whisperkit-coreml`) exists;
keep existing callers working. Use a session fixture like the other axes.

## Gate
`python3 -m pytest tests/test_audio.py -q` green (with the omlx cell either
passing or visibly SKIPped with reason). No changes to existing axes' behavior
(`python3 -m pytest tests/test_errors.py -q` still green as a canary). Commit
in focused units, author `76924051+atlascodesai@users.noreply.github.com`.
Report commit SHAs + the pytest tail. Do NOT run multi-model LLM sweeps.
