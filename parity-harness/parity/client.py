"""HTTP client helpers for both servers, with SSE parsing.

The stream reader stops on `data: [DONE]` instead of waiting for the socket to
close — native keeps the SSE connection open after [DONE] (a real framing gap),
so blocking on close would hang the harness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

from .servers import ServerHandle


@dataclass
class ChatResult:
    status: int
    body: dict | None
    raw: str


def chat(
    server: ServerHandle,
    model_id: str,
    messages: list[dict],
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
    extra: dict | None = None,
    timeout: float = 180.0,
) -> ChatResult:
    """Non-streaming POST /v1/chat/completions."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    resp = requests.post(
        f"{server.base_url}/v1/chat/completions",
        headers=server.headers(),
        data=json.dumps(payload),
        timeout=timeout,
    )
    body: dict | None
    try:
        body = resp.json()
    except ValueError:
        body = None
    return ChatResult(status=resp.status_code, body=body, raw=resp.text)


def raw_post(
    server: ServerHandle, path: str, data: str, *, timeout: float = 30.0
) -> ChatResult:
    """POST an arbitrary (possibly malformed) body — for error-semantics tests."""
    resp = requests.post(
        f"{server.base_url}{path}",
        headers=server.headers(),
        data=data,
        timeout=timeout,
    )
    try:
        body = resp.json()
    except ValueError:
        body = None
    return ChatResult(status=resp.status_code, body=body, raw=resp.text)


def audio_transcription(
    server: ServerHandle,
    model_id: str,
    wav_path: str,
    *,
    timeout: float = 180.0,
) -> ChatResult:
    """Multipart POST /v1/audio/transcriptions."""
    headers: dict[str, str] = {}
    if server.auth:
        headers["Authorization"] = f"Bearer {server.auth}"
    with open(wav_path, "rb") as audio:
        resp = requests.post(
            f"{server.base_url}/v1/audio/transcriptions",
            headers=headers,
            data={"model": model_id},
            files={"file": ("test_speech.wav", audio, "audio/wav")},
            timeout=timeout,
        )
    try:
        body = resp.json()
    except ValueError:
        body = None
    return ChatResult(status=resp.status_code, body=body, raw=resp.text)


def audio_transcription_bytes(
    server: ServerHandle,
    model_id: str,
    data: bytes,
    *,
    filename: str = "garbage.wav",
    timeout: float = 60.0,
) -> ChatResult:
    """Multipart transcription request from bytes, for malformed-audio probes."""
    headers: dict[str, str] = {}
    if server.auth:
        headers["Authorization"] = f"Bearer {server.auth}"
    resp = requests.post(
        f"{server.base_url}/v1/audio/transcriptions",
        headers=headers,
        data={"model": model_id},
        files={"file": (filename, data, "audio/wav")},
        timeout=timeout,
    )
    try:
        body = resp.json()
    except ValueError:
        body = None
    return ChatResult(status=resp.status_code, body=body, raw=resp.text)


@dataclass
class StreamResult:
    status: int
    chunks: list[dict] = field(default_factory=list)  # parsed JSON events (excl. [DONE])
    saw_done: bool = False
    done_is_last: bool = False  # was [DONE] the final payload before we stopped?
    socket_closed: bool = False  # did the server close the socket after [DONE]?
    read_error: str = ""  # non-empty if the read faulted (e.g. mid-stream server fault)
    first_chunk: dict | None = None
    usage_chunk: dict | None = None  # the choices==[] usage-bearing terminal chunk
    content_deltas: int = 0  # count of delta.content pieces
    reasoning_deltas: int = 0  # count of delta.reasoning_content pieces
    text: str = ""  # concatenated delta.content only (visible text)
    reasoning_text: str = ""  # concatenated delta.reasoning_content
    ttft_s: float | None = None  # client-observed time to first token
    gen_s: float | None = None  # client-observed generate window (first->last token)

    def output_tokens(self) -> int | None:
        usage = (self.usage_chunk or {}).get("usage") if self.usage_chunk else None
        if not usage:
            return None
        return usage.get("output_tokens", usage.get("completion_tokens"))


def _ingest_event(result: StreamResult, data: str) -> None:
    try:
        event = json.loads(data)
    except ValueError:
        return
    result.chunks.append(event)
    if result.first_chunk is None:
        result.first_chunk = event
    choices = event.get("choices") or []
    if not choices and event.get("usage"):
        result.usage_chunk = event
    for choice in choices:
        delta = choice.get("delta") or {}
        # Count only non-empty pieces: omlx opens with a keepalive chunk carrying
        # delta.content="" which is not a real generated token.
        if delta.get("content"):
            result.content_deltas += 1
            result.text += delta["content"]
        if delta.get("reasoning_content"):
            result.reasoning_deltas += 1
            result.reasoning_text += delta["reasoning_content"]


def stream_chat(
    server: ServerHandle,
    model_id: str,
    messages: list[dict],
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
    include_usage: bool = True,
    read_timeout: float = 30.0,
) -> StreamResult:
    """Streaming POST; parse SSE, stopping at `data: [DONE]`.

    GOTCHA: native emits [DONE] but does NOT close the socket, and may not flush a
    trailing newline after it — so requests.iter_lines() would block forever
    waiting for a line terminator. We instead scan raw chunks for the [DONE]
    sentinel, break immediately, and then probe whether the socket actually
    closed (native's non-close is a recorded framing gap). A bounded read timeout
    means a mid-stream server fault surfaces as read_error instead of hanging.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}

    result = StreamResult(status=0)
    try:
        with requests.post(
            f"{server.base_url}/v1/chat/completions",
            headers=server.headers(),
            data=json.dumps(payload),
            stream=True,
            timeout=(10, read_timeout),
        ) as resp:
            result.status = resp.status_code
            if resp.status_code != 200:
                return result
            t0 = time.perf_counter()
            first_t: float | None = None
            last_t: float | None = None
            buffer = ""
            # GOTCHA: native sends Connection:close with no Content-Length/chunked
            # encoding, then never closes — so iter_content(chunk_size=None) blocks
            # reading-until-EOF forever. chunk_size=1 yields bytes as they arrive
            # (curl -N behavior); we stop on [DONE] long before the socket matters.
            for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        result.saw_done = True
                        result.done_is_last = True
                        break
                    before = result.content_deltas + result.reasoning_deltas
                    _ingest_event(result, data)
                    after = result.content_deltas + result.reasoning_deltas
                    if after > before:
                        now = time.perf_counter()
                        if first_t is None:
                            first_t = now
                        last_t = now
                if result.saw_done:
                    break
            if first_t is not None:
                result.ttft_s = first_t - t0
                if last_t is not None and last_t > first_t:
                    result.gen_s = last_t - first_t
    except requests.exceptions.RequestException as exc:
        # Mid-stream fault (server hung/closed before [DONE]) — e.g. native
        # rotating-cache fault after 200 headers.
        result.read_error = f"{type(exc).__name__}: {exc}"
    return result


def probe_socket_closes(
    server: ServerHandle,
    model_id: str,
    messages: list[dict],
    *,
    max_tokens: int = 8,
    probe_timeout: float = 4.0,
) -> bool:
    """Does the server close the SSE socket after [DONE]? Reads to [DONE] then
    attempts one more read under a short timeout: a clean close yields EOF fast;
    a server that leaves the socket open (native) trips the read timeout → False."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    with requests.post(
        f"{server.base_url}/v1/chat/completions",
        headers=server.headers(),
        data=json.dumps(payload),
        stream=True,
        timeout=(10, probe_timeout),
    ) as resp:
        if resp.status_code != 200:
            return False
        buffer = ""
        it = resp.iter_content(chunk_size=1, decode_unicode=True)
        saw_done = False
        try:
            for chunk in it:
                if not chunk:
                    continue
                buffer += chunk
                if "[DONE]" in buffer:
                    saw_done = True
                    break
            if not saw_done:
                return False
            # Drain past [DONE]: a closed socket exhausts the iterator fast (EOF);
            # an open socket blocks once the trailing bytes are consumed → the read
            # timeout fires and we conclude "not closed".
            for _ in it:
                pass
            return True  # iterator exhausted cleanly = socket closed
        except requests.exceptions.RequestException:
            return False  # read timed out past [DONE] => socket left open
