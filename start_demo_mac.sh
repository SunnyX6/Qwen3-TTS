#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_DIR="${SCRIPT_DIR}/py312"
ENV_PYTHON="${ENV_DIR}/bin/python"
DEFAULT_CHECKPOINT="Qwen3-TTS-12Hz-1.7B-Base"
CHECKPOINT="${DEFAULT_CHECKPOINT}"
DISPLAY_IP="127.0.0.1"
DISPLAY_PORT="7860"
FLASH_MODE="prompt"
FLASH_ARG_PRESENT=0
PORT_ARG_PRESENT=0
IP_ARG_PRESENT=0
DEVICE_ARG_PRESENT=0
POSITIONAL_CHECKPOINT_PRESENT=0
EXPECT_VALUE=""
SELECTED_DEVICE=""
CUDA_AVAILABLE=0

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Error: missing Conda environment at ${ENV_DIR}" >&2
  echo "Create it in the current project directory with:" >&2
  echo "  cd \"${SCRIPT_DIR}\"" >&2
  echo "  conda create --prefix \"${ENV_DIR}\" python=3.12 -y" >&2
  echo "  conda activate \"${ENV_DIR}\"" >&2
  echo "  # install a matching PyTorch build from https://pytorch.org/get-started/locally/" >&2
  echo "  pip install -e \".[runtime]\"" >&2
  echo "  # optional: if you want FlashAttention, also install:" >&2
  echo "  \"${ENV_PYTHON}\" -m pip install flash-attn --no-build-isolation" >&2
  exit 1
fi

if [[ -d "${SCRIPT_DIR}/models/${DEFAULT_CHECKPOINT}" ]]; then
  CHECKPOINT="${SCRIPT_DIR}/models/${DEFAULT_CHECKPOINT}"
fi

for arg in "$@"; do
  if [[ -n "${EXPECT_VALUE}" ]]; then
    case "${EXPECT_VALUE}" in
      checkpoint)
        CHECKPOINT="${arg}"
        ;;
      port)
        DISPLAY_PORT="${arg}"
        ;;
      ip)
        DISPLAY_IP="${arg}"
        ;;
      device)
        SELECTED_DEVICE="${arg}"
        ;;
    esac
    EXPECT_VALUE=""
    continue
  fi

  case "${arg}" in
    --checkpoint|-c)
      EXPECT_VALUE="checkpoint"
      ;;
    --flash-attn)
      FLASH_MODE="on"
      FLASH_ARG_PRESENT=1
      ;;
    --no-flash-attn)
      FLASH_MODE="off"
      FLASH_ARG_PRESENT=1
      ;;
    --port)
      PORT_ARG_PRESENT=1
      EXPECT_VALUE="port"
      ;;
    --ip)
      IP_ARG_PRESENT=1
      EXPECT_VALUE="ip"
      ;;
    --device)
      DEVICE_ARG_PRESENT=1
      EXPECT_VALUE="device"
      ;;
    -*)
      ;;
    *)
      if [[ "${POSITIONAL_CHECKPOINT_PRESENT}" -eq 0 ]]; then
        CHECKPOINT="${arg}"
        POSITIONAL_CHECKPOINT_PRESENT=1
      fi
      ;;
  esac
done

if "${ENV_PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  CUDA_AVAILABLE=1
fi

if [[ "${FLASH_MODE}" == "prompt" ]]; then
  if [[ "${CUDA_AVAILABLE}" -eq 0 ]]; then
    echo "CUDA is not available; starting with --no-flash-attn." >&2
    FLASH_MODE="off"
  elif [[ -t 0 ]]; then
    echo
    read -r -p "Enable FlashAttention for Gradio demo? [y/N, N uses --no-flash-attn]: " flash_choice
    case "${flash_choice}" in
      y|Y|yes|YES|Yes)
        FLASH_MODE="on"
        ;;
      *)
        FLASH_MODE="off"
        ;;
    esac
  else
    echo "No interactive input detected; starting with --no-flash-attn." >&2
    FLASH_MODE="off"
  fi
fi

EXTRA_ARGS=()

if [[ "${DEVICE_ARG_PRESENT}" -eq 0 ]]; then
  if "${ENV_PYTHON}" -c "import torch; m=getattr(torch.backends, 'mps', None); ok=bool(m and getattr(m, 'is_built', lambda: True)() and getattr(m, 'is_available', lambda: False)()); raise SystemExit(0 if ok else 1)" >/dev/null 2>&1; then
    SELECTED_DEVICE="mps"
  else
    SELECTED_DEVICE="cpu"
  fi
  EXTRA_ARGS+=(--device "${SELECTED_DEVICE}")
fi

if [[ "${FLASH_MODE}" == "on" ]]; then
  if [[ "${CUDA_AVAILABLE}" -eq 0 ]]; then
    echo "\`--flash-attn\` was requested, but FlashAttention requires CUDA." >&2
    echo "Please switch to a CUDA device or run again with --no-flash-attn." >&2
    exit 1
  fi
  if ! "${ENV_PYTHON}" -c "import flash_attn" >/dev/null 2>&1; then
    echo "\`flash_attn\` is not installed in ${ENV_DIR}." >&2
    echo "Install it in the current environment with:" >&2
    echo "  \"${ENV_PYTHON}\" -m pip install flash-attn --no-build-isolation" >&2
    echo "If you do not want to install it, run this script again and choose N." >&2
    exit 1
  fi
  if [[ "${FLASH_ARG_PRESENT}" -eq 0 ]]; then
    EXTRA_ARGS+=(--flash-attn)
  fi
else
  if [[ "${FLASH_ARG_PRESENT}" -eq 0 ]]; then
    EXTRA_ARGS+=(--no-flash-attn)
  fi
fi

if [[ "${PORT_ARG_PRESENT}" -eq 0 ]]; then
  EXTRA_ARGS+=(--port 7860)
fi
if [[ "${IP_ARG_PRESENT}" -eq 0 ]]; then
  EXTRA_ARGS+=(--ip 127.0.0.1)
fi

echo "Starting Gradio demo with checkpoint:"
echo "  ${CHECKPOINT}"
if [[ "${DEVICE_ARG_PRESENT}" -eq 0 ]]; then
  echo "Device: ${SELECTED_DEVICE} (auto-selected for macOS)"
fi
echo "Open http://${DISPLAY_IP}:${DISPLAY_PORT} after startup."
echo

CMD=("${ENV_PYTHON}" -m qwen_tts.cli.demo --checkpoint "${CHECKPOINT}")
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi
CMD+=("$@")

exec "${CMD[@]}"
