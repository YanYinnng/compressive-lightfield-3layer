#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${LIGHTFIELD_MINICONDA_DIR:-$HOME/miniconda3}"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Use Linux x86_64.' >&2; exit 1; }
if [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
  mkdir -p "$ROOT/.bootstrap"
  installer="$ROOT/.bootstrap/Miniconda3-latest-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
  else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
  fi
  bash "$installer" -b -p "$MINICONDA_DIR"
fi
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="$MINICONDA_DIR/envs"
if [[ ! -f "$MINICONDA_DIR/envs/light_310/conda-meta/history" ]]; then
  conda create --yes --name light_310 --override-channels --channel conda-forge python=3.10 pip
fi
unset PYTHONHOME PYTHONPATH PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
conda activate light_310
python -m pip install --index-url https://pypi.org/simple -r "$ROOT/requirements.txt"
python -c 'import pathlib,sys,numpy,PIL; assert sys.version_info[:2]==(3,10) and pathlib.Path(sys.prefix).name=="light_310"; print("light_310 ready")'
