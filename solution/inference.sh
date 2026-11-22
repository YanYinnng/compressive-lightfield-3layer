#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${LIGHTFIELD_MINICONDA_DIR:-$HOME/miniconda3}"
[[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]] || { echo 'Run bash create_env.sh first.' >&2; exit 1; }
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="$MINICONDA_DIR/envs"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
conda activate light_310
python -c 'import pathlib,sys; assert sys.version_info[:2]==(3,10) and pathlib.Path(sys.prefix).name=="light_310"'
exec python -u "$ROOT/inference.py" "$@"
