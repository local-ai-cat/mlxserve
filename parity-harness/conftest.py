"""Pytest fixtures + gating for the parity harness.

Gate: the whole session is skipped unless PARITY_HARNESS=1 (no GPU / CI without
Metal → clean skip). omlx is launched once (session scope, multi-model). native
is launched one process per model on demand; to cap memory we keep at most one
native alive and relaunch when the model changes. Tests are sorted by model so
that costs at most one native launch per model.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from parity import config
from parity.report import REPORT
from parity.servers import ServerHandle, start_native, start_omlx

_LOG_DIR = Path(tempfile.gettempdir()) / "parity-harness-logs"


def pytest_addoption(parser):
    parser.addoption(
        "--tier",
        action="store",
        default="smoke",
        choices=["smoke", "full"],
        help="model-architecture matrix tier: smoke (small, default) or full (heavy).",
    )


def pytest_configure(config):  # noqa: ARG001 (name fixed by pytest hookspec)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def pytest_generate_tests(metafunc):
    """Parametrize `model_id`: matrix tests span the tier set (skipping models not
    present in the farm); conformance tests span the dense smoke models native
    handles. Using one arg name keeps native grouped (one load per model)."""
    if "model_id" not in metafunc.fixturenames:
        return
    tier = metafunc.config.getoption("--tier")
    if metafunc.function.__module__.endswith("test_matrix"):
        specs = config.matrix_models(tier)
        ids = []
        params = []
        for spec in specs:
            present = (Path(config.MODEL_FARM) / spec.model_id).exists()
            marks = () if present else (pytest.mark.skip(reason=f"{spec.model_id} not in farm"),)
            params.append(pytest.param(spec.model_id, marks=marks))
            ids.append(spec.model_id)
        metafunc.parametrize("model_id", params, ids=ids)
    else:
        models = [m.model_id for m in config.CONFORMANCE_MODELS]
        metafunc.parametrize("model_id", models)


def pytest_collection_modifyitems(items):
    """Group tests by their model param so native relaunches at most once/model."""

    def key(item):
        model = ""
        if hasattr(item, "callspec"):
            model = str(item.callspec.params.get("model_id", ""))
        return model

    items.sort(key=key)


def _gated() -> bool:
    return os.environ.get("PARITY_HARNESS") == "1"


@pytest.fixture(scope="session", autouse=True)
def _require_gate():
    if not _gated():
        pytest.skip(
            "PARITY_HARNESS!=1 — parity harness needs a real Apple-Silicon GPU; "
            "set PARITY_HARNESS=1 to run.",
            allow_module_level=False,
        )


@pytest.fixture(scope="session")
def omlx_server():
    if not _gated():
        pytest.skip("gated")
    for binpath in (config.NATIVE_BIN, config.OMLX_BIN):
        if not Path(binpath).exists():
            pytest.skip(f"missing binary: {binpath}")
    try:
        handle = start_omlx(_LOG_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"omlx failed to launch: {exc}")
    yield handle
    handle.stop()


class _NativePool:
    """Keeps at most one native process alive, keyed by model_id."""

    def __init__(self) -> None:
        self.current: ServerHandle | None = None

    def get(self, model_id: str) -> ServerHandle:
        if self.current is not None and self.current.model_id == model_id:
            return self.current
        if self.current is not None:
            self.current.stop()
            self.current = None
        self.current = start_native(model_id, _LOG_DIR)
        return self.current

    def close(self) -> None:
        if self.current is not None:
            self.current.stop()
            self.current = None


@pytest.fixture(scope="session")
def native_pool():
    pool = _NativePool()
    yield pool
    pool.close()


@pytest.fixture
def spec(model_id):
    return config.spec_for(model_id)


@pytest.fixture
def native_server(native_pool, model_id):
    """Native server pinned to the current test's model. May raise if native
    can't even launch — matrix tests catch that and record a GAP/FAIL."""
    return native_pool.get(model_id)


def _native_version() -> str:
    """Best-effort native build id: git short SHA of the native source repo."""
    repo = os.environ.get(
        "PARITY_NATIVE_REPO", "/Users/timapple/Documents/Github/mlxserve-native"
    )
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"git {sha}"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _omlx_version() -> str:
    try:
        out = subprocess.check_output(
            [config.OMLX_BIN, "--version"], stderr=subprocess.STDOUT, text=True, timeout=15
        ).strip()
        return out.splitlines()[-1][:60] if out else "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Emit the conformance matrix (md + html) once, after all cells recorded."""
    if not _gated():
        return
    tier = session.config.getoption("--tier")
    REPORT.meta.generated_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    REPORT.meta.native_version = _native_version()
    REPORT.meta.omlx_version = _omlx_version()
    REPORT.meta.tier = tier
    REPORT.meta.model_count = len(config.matrix_models(tier))

    here = Path(__file__).parent
    REPORT.write_report(here / "report.md")
    REPORT.write_html(here / "report.html")

    # Snapshot server logs next to the report for debugging.
    dest = here / "logs"
    if _LOG_DIR.exists():
        dest.mkdir(exist_ok=True)
        for log in _LOG_DIR.glob("*.log"):
            try:
                shutil.copy(log, dest / log.name)
            except OSError:
                pass
