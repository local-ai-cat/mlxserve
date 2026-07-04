#!/usr/bin/env bash
# Differential parity harness runner: native MLXServe vs Python omlx.
#
# Freezes the native binary (so a later Swift rebuild can't disturb the run),
# then invokes pytest, which launches both servers itself and diffs them across
# five axes. Emits report.md (conformance matrix) + report-junit.xml (CI).
#
# Requires a real Apple-Silicon GPU. Without PARITY_HARNESS=1 the suite skips.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Freeze native binary + its colocated metallib to a stable dir.
NATIVE_SRC_DIR="${NATIVE_SRC_DIR:-/Users/timapple/Documents/Github/mlxserve-native/.build/release}"
FROZEN_DIR="${PARITY_FROZEN_DIR:-/private/tmp/parity-native-baseline}"
if [[ -x "${NATIVE_SRC_DIR}/mlxserve-http" ]]; then
  mkdir -p "${FROZEN_DIR}"
  cp -f "${NATIVE_SRC_DIR}/mlxserve-http" "${NATIVE_SRC_DIR}/mlx.metallib" "${FROZEN_DIR}/"
fi
export PARITY_NATIVE_BIN="${PARITY_NATIVE_BIN:-${FROZEN_DIR}/mlxserve-http}"

# 2. Run the harness.
export PARITY_HARNESS="${PARITY_HARNESS:-1}"
cd "${HERE}"
exec python3 -m pytest "$@"
