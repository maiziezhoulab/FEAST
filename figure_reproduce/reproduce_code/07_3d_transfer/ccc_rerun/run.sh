#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CCC_PYTHON="${CCC_PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/feast-ccc-mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/feast-ccc-numba}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=2
export MPLBACKEND=Agg

# Calling the interpreter directly works in noninteractive shells, without conda init.
if [[ $# -eq 0 ]]; then set -- check; fi
cd "$SCRIPT_DIR/../.."
exec "$CCC_PYTHON" -u "$SCRIPT_DIR/LOT.py" "$@"
