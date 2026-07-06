"""Launch, health-check, and tear down the two servers under test.

Post-M3 native MLXServe is multi-model + validating: it is launched ONCE against
a directory of model subdirs, discovers them (id = bare leaf dir name), serves
on demand, and validates the request `model` id (unknown -> 404). omlx is also
multi-model. Both are pointed at the SAME real nested store so their discovered
ids match, and both run on ephemeral ports.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from . import config


def free_port() -> int:
    """Return an OS-assigned free TCP port (closed immediately; small race window)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerHandle:
    """A running server process plus how to talk to it."""

    def __init__(self, name: str, base_url: str, proc: subprocess.Popen, auth: str | None):
        self.name = name
        self.base_url = base_url
        self.proc = proc
        self.auth = auth
        self.model_id: str | None = None
        self._ids_cache: set[str] | None = None

    def headers(self) -> dict[str, str]:
        head = {"Content-Type": "application/json"}
        if self.auth:
            head["Authorization"] = f"Bearer {self.auth}"
        return head

    def discovered_ids(self, refresh: bool = False) -> set[str]:
        """Model ids this server reports from GET /v1/models (cached). Empty set
        on any error, so callers can `model_id in server.discovered_ids()` and
        skip loudly rather than crash."""
        if self._ids_cache is not None and not refresh:
            return self._ids_cache
        ids: set[str] = set()
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models", headers=self.headers(), timeout=10
            )
            if resp.status_code == 200:
                data = resp.json().get("data") or []
                ids = {m.get("id") for m in data if isinstance(m, dict) and m.get("id")}
        except (requests.RequestException, ValueError):
            ids = set()
        self._ids_cache = ids
        return ids

    def wait_listening(self, timeout: float = 60.0) -> None:
        """Block until GET /v1/models returns 200 (process up + routes live)."""
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited early (rc={self.proc.returncode}); last={last}"
                )
            try:
                resp = requests.get(
                    f"{self.base_url}/v1/models", headers=self.headers(), timeout=3
                )
                if resp.status_code == 200:
                    return
                last = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:  # not up yet
                last = str(exc)
            time.sleep(0.5)
        raise TimeoutError(f"{self.name} never became ready ({last})")

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass


def start_native(log_dir: Path) -> ServerHandle:
    """Launch native MLXServe ONCE against the whole model store (multi-model,
    post-M3). No --model-id override — native discovers every subdir and serves
    on demand, validating the request `model`. GOTCHA: mlx.metallib must sit
    beside the binary — the frozen baseline dir guarantees that. GOTCHA: native's
    discovery does NOT follow symlinks, so MODEL_STORE must be the real nested
    store, not the flattened symlink farm."""
    port = free_port()
    log = open(log_dir / f"native-{port}.log", "w")  # noqa: SIM115 (kept open for proc)
    args = [
        config.NATIVE_BIN,
        "--model-dir",
        config.MODEL_STORE,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    whisperkit_models = Path(config.WHISPERKIT_MODELS).expanduser()
    if whisperkit_models.exists():
        args.extend(["--whisperkit-models-dir", str(whisperkit_models)])
    proc = subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(Path(config.NATIVE_BIN).parent),
    )
    handle = ServerHandle("native", f"http://127.0.0.1:{port}", proc, auth=None)
    handle.wait_listening()
    return handle


def start_omlx(log_dir: Path) -> ServerHandle:
    """Launch omlx once against the SAME real store as native, so the ids it
    discovers (bare leaf dir names) match native's and one `model` string selects
    the same model on both."""
    port = free_port()
    log = open(log_dir / f"omlx-{port}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            config.OMLX_BIN,
            "serve",
            "--model-dir",
            config.MODEL_STORE,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            config.OMLX_API_KEY,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    handle = ServerHandle("omlx", f"http://127.0.0.1:{port}", proc, auth=config.OMLX_API_KEY)
    handle.wait_listening()
    return handle
