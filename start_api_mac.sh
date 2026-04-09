#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_DIR="${SCRIPT_DIR}/py312"
ENV_PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Error: missing Conda environment at ${ENV_DIR}" >&2
  echo "Create it in the current project directory with:" >&2
  echo "  cd \"${SCRIPT_DIR}\"" >&2
  echo "  conda create --prefix \"${ENV_DIR}\" python=3.12 -y" >&2
  echo "  conda activate \"${ENV_DIR}\"" >&2
  echo "  # install a matching PyTorch build from https://pytorch.org/get-started/locally/" >&2
  echo "  pip install -e \".[runtime,api]\"" >&2
  echo "  # optional: if you want FlashAttention, also install:" >&2
  echo "  \"${ENV_PYTHON}\" -m pip install flash-attn --no-build-isolation" >&2
  exit 1
fi

flash_mode="prompt"
flash_arg_present=0

for arg in "$@"; do
  case "${arg}" in
    --flash-attn)
      flash_mode="on"
      flash_arg_present=1
      ;;
    --no-flash-attn)
      flash_mode="off"
      flash_arg_present=1
      ;;
  esac
done

if [[ "${flash_mode}" == "prompt" ]]; then
  if [[ -t 0 ]]; then
    echo
    read -r -p "Enable FlashAttention? [y/N, N uses --no-flash-attn]: " flash_choice
    case "${flash_choice}" in
      y|Y|yes|YES|Yes)
        flash_mode="on"
        ;;
      *)
        flash_mode="off"
        ;;
    esac
  else
    echo "No interactive input detected; starting with --no-flash-attn." >&2
    flash_mode="off"
  fi
fi

extra_args=()
if [[ "${flash_mode}" == "on" ]]; then
  if ! "${ENV_PYTHON}" -c "import flash_attn" >/dev/null 2>&1; then
    echo "\`flash_attn\` is not installed in ${ENV_DIR}." >&2
    echo "Install it in the current environment with:" >&2
    echo "  \"${ENV_PYTHON}\" -m pip install flash-attn --no-build-isolation" >&2
    echo "If you do not want to install it, run this script again and choose N." >&2
    exit 1
  fi
  if [[ "${flash_arg_present}" -eq 0 ]]; then
    extra_args+=(--flash-attn)
  fi
else
  if [[ "${flash_arg_present}" -eq 0 ]]; then
    extra_args+=(--no-flash-attn)
  fi
fi

exec "${ENV_PYTHON}" api/main.py "${extra_args[@]}" "$@"
