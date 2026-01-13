#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Defaults
# ============================================================
TASK="Isaac-Velocity-Rough-Kuroko-v0"

# If empty -> auto-detect by nvidia-smi
NPROC_PER_NODE=""

MAX_ITERATIONS=100000
HANG_TIMEOUT_S=15
RESTART_SLEEP_S=15

EXTRA_ARGS=()

# If set, start the *first* launch from scratch (auto-resume disabled once).
# After the first launch, auto-resume is enabled so restarts resume from the latest checkpoint.
FRESH_RUN=0
FIRST_LAUNCH=1

# ============================================================
# Helpers
# ============================================================
usage() {
  cat <<'EOF'
Usage:
  ./train_loop.sh [options] [-- <extra args for train.py>]

Options:
  -t, --task <name>           Task name
  -n, --nproc <int>           nproc_per_node (auto-detect if omitted)
      --max_iterations <int>  max_iterations (default: 100000)
      --hang_timeout_s <int>  hang timeout seconds (default: 15)
      --restart_sleep_s <int> Restart delay after crash (default: 15)
      --fresh                 Start from scratch on the first launch (no auto-resume once)
  -h, --help                  Show this help

Notes:
  - Headless is always enabled
  - If nproc_per_node == 1:
      * Single-process launch (no torch.distributed)
      * Do NOT pass --nnodes / --nproc_per_node / --distributed
  - If nproc_per_node >= 2:
      * Multi-process launch via torch.distributed.run
      * Pass --distributed to train.py
EOF
}

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$count" -ge 1 ]]; then
      echo "$count"
      return
    fi
  fi
  echo 1
}

# ============================================================
# Argument parsing
# ============================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--task)
      TASK="$2"; shift 2 ;;
    -n|--nproc|--nproc_per_node)
      NPROC_PER_NODE="$2"; shift 2 ;;
    --max_iterations)
      MAX_ITERATIONS="$2"; shift 2 ;;
    --hang_timeout_s)
      HANG_TIMEOUT_S="$2"; shift 2 ;;
    --restart_sleep_s)
      RESTART_SLEEP_S="$2"; shift 2 ;;
    --fresh)
      FRESH_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

# ============================================================
# Validation
# ============================================================
if [[ -z "$NPROC_PER_NODE" ]]; then
  NPROC_PER_NODE="$(detect_gpu_count)"
fi

if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "[ERROR] nproc_per_node must be >= 1 (got: $NPROC_PER_NODE)" >&2
  exit 1
fi

# ============================================================
# Environment
# ============================================================
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=15
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# ============================================================
# Command setup
# ============================================================
TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"

BASE_ARGS=(
  --task "$TASK"
  --headless
  --max_iterations "$MAX_ITERATIONS"
  --hang_timeout_s "$HANG_TIMEOUT_S"
)

if [[ "$FRESH_RUN" -eq 1 ]]; then
  echo "[INFO] --fresh enabled: first launch will start from scratch (no auto-resume)"
fi

echo "[INFO] task=$TASK headless=true nproc_per_node=$NPROC_PER_NODE"

# ============================================================
# Training loop
# ============================================================
while true; do
  # Compose args per launch.
  # - Default: always enable auto-resume.
  # - With --fresh: disable auto-resume only for the very first launch, then enable it for restarts.
  RUN_ARGS=("${BASE_ARGS[@]}")
  if [[ "$FRESH_RUN" -eq 1 ]]; then
    if [[ "$FIRST_LAUNCH" -eq 0 ]]; then
      RUN_ARGS+=(--auto_resume)
    else
      echo "[INFO] --fresh: first launch starts from scratch (auto_resume disabled for this launch)"
    fi
  else
    RUN_ARGS+=(--auto_resume)
  fi

  set +e
  if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    ./isaaclab.sh -p "$TRAIN_SCRIPT" \
      "${RUN_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  else
    ./isaaclab.sh -p -m torch.distributed.run \
      --nnodes=1 \
      --nproc_per_node="$NPROC_PER_NODE" \
      "$TRAIN_SCRIPT" \
      "${RUN_ARGS[@]}" \
      --distributed \
      "${EXTRA_ARGS[@]}"
  fi
  exit_code=$?
  set -e

  FIRST_LAUNCH=0

  echo "[INFO] Training exited with code: $exit_code"

  if [[ "$exit_code" -eq 0 ]]; then
    echo "[INFO] Training completed successfully. Exiting."
    break
  fi

  echo "[INFO] Training crashed. Restarting in ${RESTART_SLEEP_S} seconds..."
  sleep "$RESTART_SLEEP_S"
done
