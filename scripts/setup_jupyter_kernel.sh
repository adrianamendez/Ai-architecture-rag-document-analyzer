#!/usr/bin/env bash
set -euo pipefail

# Setup a Python 3.12 virtual environment, install dependencies with pip3,
# and register a Jupyter kernel for this project.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv312"
KERNEL_NAME="rag312"
KERNEL_DISPLAY_NAME="Python 3.12 (rag)"

echo "Project root: ${PROJECT_ROOT}"

if command -v python3.12 >/dev/null 2>&1; then
  PY312_BIN="$(command -v python3.12)"
elif [[ -x "/opt/homebrew/bin/python3.12" ]]; then
  PY312_BIN="/opt/homebrew/bin/python3.12"
elif [[ -x "/usr/local/bin/python3.12" ]]; then
  PY312_BIN="/usr/local/bin/python3.12"
else
  echo "ERROR: Python 3.12 not found."
  echo "Install it first, e.g.: brew install python@3.12"
  exit 1
fi

echo "Using Python 3.12 at: ${PY312_BIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PY312_BIN}" -m venv "${VENV_DIR}"
  echo "Created virtualenv: ${VENV_DIR}"
else
  echo "Virtualenv already exists: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python --version
python -m pip install --upgrade pip setuptools wheel

if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  pip3 install -r "${PROJECT_ROOT}/requirements.txt"
else
  echo "WARNING: requirements.txt not found. Installing base notebook packages only."
fi

# Ensure common notebook/runtime deps are present in this kernel.
pip3 install ipykernel matplotlib ipywidgets jupyterlab_widgets

python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "Done."
echo "Now open Jupyter and select kernel: ${KERNEL_DISPLAY_NAME}"
