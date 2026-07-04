"""Launch, health-check, and tear down the two servers under test.

native MLXServe serves ONE model per process (fixed at launch) and must be
handed the resolved snapshot dir. omlx is multi-model and discovers subdirs of
the farm; the `model` field selects. Both are started on ephemeral ports.
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


def resolve_snapshot_dir(model_id: str) -> Path:
    """Resolve a farm symlink to the concrete dir that holds config.json + weights.

    native's --model-dir wants the leaf snapshot dir, not the symlink.
    """
    link = Path(config.MODEL_FARM) / model_id
    real = link.resolve()
    if not (real / "config.json").exists():
        raise FileNotFoundError(f"{real} has no config.json (model_id={model_id})")
    return real


class ServerHandle:
    """A running server process plus how to talk to it."""

    def __init__(self, name: str, base_url: str, proc: subprocess.Popen, auth: str | None):
        self.name = name
        self.base_url = base_url
        self.proc = proc
        self.auth = auth
        self.model_id: str | None = None

    def headers(self) -> dict[str, str]:
        head = {"Content-Type": "application/json"}
        if self.auth:
            head["Authorization"] = f"Bearer {self.auth}"
        return head

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


def start_native(model_id: str, log_dir: Path) -> ServerHandle:
    """Launch native MLXServe pinned to one model. GOTCHA: mlx.metallib must sit
    beside the binary — the frozen baseline dir guarantees that."""
    snapshot = resolve_snapshot_dir(model_id)
    port = free_port()
    log = open(log_dir / f"native-{port}.log", "w")  # noqa: SIM115 (kept open for proc)
    proc = subprocess.Popen(
        [
            config.NATIVE_BIN,
            "--model-dir",
            str(snapshot),
            "--model-id",
            model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(Path(config.NATIVE_BIN).parent),
    )
    handle = ServerHandle("native", f"http://127.0.0.1:{port}", proc, auth=None)
    handle.model_id = model_id
    handle.wait_listening()
    return handle


def start_omlx(log_dir: Path) -> ServerHandle:
    """Launch omlx once; it serves every model in the farm."""
    port = free_port()
    log = open(log_dir / f"omlx-{port}.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            config.OMLX_BIN,
            "serve",
            "--model-dir",
            config.MODEL_FARM,
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
