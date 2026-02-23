#!/usr/bin/env bash
set -eEuo pipefail

# =============================================================================
# Task 2 + Task 3 Agent Benchmark Runner
# Full 6 Tools + Critic + Image Reinjection
# =============================================================================
#
# Pipeline settings UNIFIED with:
#   scripts/run_task1_ham10000_500_agent_full_critic.sh
#
# Runs sequentially:
#   [1/3] Task 2 Derm7pt  (100 samples) - Concept Annotation
#   [2/3] Task 2 SkinCon  (100 samples) - Concept Annotation
#   [3/3] Task 3 SkinCAP  (100 samples) - Clinical Image Captioning
#
# Configuration (same as HAM10000 full_critic):
#   - 6 Tools: panderm, make, dermogpt_vqa, rag, text_rag, ontology
#     (Task 2/3 prompts use dermogpt_vqa; Task 1 uses qwen_vqa)
#   - Critic: Enabled (retry on low confidence / tool conflicts)
#   - Image Reinjection: Enabled (re-inject image when critic triggers retry)
#   - Max Critic Retries: 2
#   - Output: results/agent_critic/ (isolated from results/agent/)
#
# Usage:
#   bash scripts/run_task2_task3_agent_critic.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_task2_task3_agent_critic.sh
#   MAX_SAMPLES=10 bash scripts/run_task2_task3_agent_critic.sh
#
#   # Resume a specific sub-task
#   RESUME_TASK2_DERM7PT=20260214_120000 bash scripts/run_task2_task3_agent_critic.sh
#   RESUME_TASK2_SKINCON=20260214_130000 bash scripts/run_task2_task3_agent_critic.sh
#   RESUME_TASK3_SKINCAP=20260214_140000 bash scripts/run_task2_task3_agent_critic.sh
#
#   # Skip completed sub-tasks
#   SKIP_TASK2_DERM7PT=true SKIP_TASK2_SKINCON=true bash scripts/run_task2_task3_agent_critic.sh
#
#   # Override critic settings
#   MAX_CRITIC_RETRIES=3 CRITIC_LLM_SUMMARY=true bash scripts/run_task2_task3_agent_critic.sh
#
# Notes:
# - Output is under results/agent_critic/ (NOT results/agent/) to isolate from original runs
# - ENABLE_CRITIC: Always true (override with ENABLE_CRITIC=false to disable)
# - CRITIC_IMAGE_REINJECT: Always true (override with CRITIC_IMAGE_REINJECT=false)
# - CRITIC_LLM_SUMMARY: LLM evidence summary (default: false)
# - MAX_CRITIC_RETRIES: Max critic retries per sample (default: 2)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============== Load .env file (if exists) ==============
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============== Override LangSmith Project ==============
export LANGSMITH_PROJECT="DermAgent"

# ============== Notification (optional) ==============
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

send_telegram() {
  local message="$1"
  if [[ -z "${TELEGRAM_BOT_TOKEN}" || -z "${TELEGRAM_CHAT_ID}" ]]; then
    return 0
  fi
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${message}" \
    --data-urlencode "parse_mode=HTML" > /dev/null 2>&1 || true
}

# =============================================================================
# Unified Pipeline Settings (aligned with run_task1_ham10000_500_agent_full_critic.sh)
# =============================================================================
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
DEVICE="${DEVICE:-cuda}"
# All 6 tools: panderm, make, dermogpt_vqa, rag, text_rag, ontology
# Note: Task 2/3 prompts use dermogpt_vqa; Task 1 (HAM10000) uses qwen_vqa
ENABLED_TOOLS="${ENABLED_TOOLS:-panderm,make,dermogpt_vqa,rag,text_rag,ontology}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 means no limit
HOSTNAME="$(hostname)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Critic configuration (defaults match HAM10000 full_critic)
ENABLE_CRITIC="${ENABLE_CRITIC:-true}"
CRITIC_LLM_SUMMARY="${CRITIC_LLM_SUMMARY:-false}"
CRITIC_IMAGE_REINJECT="${CRITIC_IMAGE_REINJECT:-true}"
MAX_CRITIC_RETRIES="${MAX_CRITIC_RETRIES:-2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

# =============================================================================
# Resume / Skip configuration (per sub-task)
# =============================================================================
RESUME_TASK2_DERM7PT="${RESUME_TASK2_DERM7PT:-}"
RESUME_TASK2_SKINCON="${RESUME_TASK2_SKINCON:-}"
RESUME_TASK3_SKINCAP="${RESUME_TASK3_SKINCAP:-}"
RETRY_ERRORS="${RETRY_ERRORS:-false}"

SKIP_TASK2_DERM7PT="${SKIP_TASK2_DERM7PT:-false}"
SKIP_TASK2_SKINCON="${SKIP_TASK2_SKINCON:-false}"
SKIP_TASK3_SKINCAP="${SKIP_TASK3_SKINCAP:-false}"

# =============================================================================
# Output directories (isolated under results/agent_critic/)
# =============================================================================
OUTPUT_BASE="${PROJECT_ROOT}/results/agent_critic"
OUTPUT_TASK2_DERM7PT="${OUTPUT_BASE}/task2_concept/derm7pt_100"
OUTPUT_TASK2_SKINCON="${OUTPUT_BASE}/task2_concept/skincon_100"
OUTPUT_TASK3_SKINCAP="${OUTPUT_BASE}/task3_captioning/skincap_100"

echo "================================================================================"
echo "Task 2 + Task 3 Agent Benchmark"
echo "Full 6 Tools + Critic + Image Reinjection"
echo "================================================================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "DEVICE: ${DEVICE}"
echo "ENABLED_TOOLS: ${ENABLED_TOOLS}"
echo "QWEN_MODEL_ID: ${QWEN_MODEL_ID}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "ENABLE_CRITIC: ${ENABLE_CRITIC}"
echo "CRITIC_LLM_SUMMARY: ${CRITIC_LLM_SUMMARY}"
echo "CRITIC_IMAGE_REINJECT: ${CRITIC_IMAGE_REINJECT}"
echo "MAX_CRITIC_RETRIES: ${MAX_CRITIC_RETRIES}"
echo "OUTPUT_BASE: ${OUTPUT_BASE}"
echo ""
echo "Sub-tasks:"
echo "  [1/3] Task2 Derm7pt_100  -> ${OUTPUT_TASK2_DERM7PT}  (skip: ${SKIP_TASK2_DERM7PT})"
echo "  [2/3] Task2 SkinCon_100  -> ${OUTPUT_TASK2_SKINCON}  (skip: ${SKIP_TASK2_SKINCON})"
echo "  [3/3] Task3 SkinCAP_100  -> ${OUTPUT_TASK3_SKINCAP}  (skip: ${SKIP_TASK3_SKINCAP})"
echo "================================================================================"

echo ""
echo "[Pre-run checks]"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found (ok if CPU-only)"
free -h || true
df -h /mnt/hdd/sdb/ || true

# =============================================================================
# Helper: build critic flags (same logic as HAM10000 full_critic)
# =============================================================================
build_critic_args() {
  local -n arr=$1
  if [[ "${ENABLE_CRITIC}" == "true" ]]; then
    arr+=(--enable-critic)
  fi
  if [[ "${CRITIC_LLM_SUMMARY}" == "true" ]]; then
    arr+=(--critic-llm-summary)
  fi
  if [[ "${CRITIC_IMAGE_REINJECT}" == "true" ]]; then
    arr+=(--critic-image-reinject)
  fi
  if [[ -n "${MAX_CRITIC_RETRIES}" && "${MAX_CRITIC_RETRIES}" != "2" ]]; then
    arr+=(--max-critic-retries "${MAX_CRITIC_RETRIES}")
  fi
}

overall_start="$(date '+%Y-%m-%d %H:%M:%S')"
send_telegram "🚀 <b>Task2+3 Agent Full+Critic 开始</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
🧰 工具: <code>${ENABLED_TOOLS}</code>
🔍 Critic: ON | Summary: ${CRITIC_LLM_SUMMARY} | Reinject: ${CRITIC_IMAGE_REINJECT} | MaxRetries: ${MAX_CRITIC_RETRIES}
🧪 MAX_SAMPLES: <code>${MAX_SAMPLES}</code>
📁 输出: <code>${OUTPUT_BASE}</code>"

FAILED_TASKS=()

# =============================================================================
# [1/3] Task 2 - Derm7pt 100 samples
# =============================================================================
if [[ "${SKIP_TASK2_DERM7PT}" != "true" ]]; then
  echo ""
  echo "================================================================================"
  echo "[1/3] Task 2 Concept Annotation - Derm7pt 100 Samples (Critic)"
  echo "================================================================================"

  mkdir -p "${OUTPUT_TASK2_DERM7PT}"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  send_telegram "▶️ <b>[1/3] Task2 Derm7pt_100 Critic 开始</b>
🕐 ${start_time}"

  CMD_ARGS=(
    "${PROJECT_ROOT}/scripts/run_task2_concept_agent.py"
    --dataset derm7pt
    --sample-size "100"
    --device "${DEVICE}"
    --model-name "${MODEL_NAME}"
    --enabled-tools "${ENABLED_TOOLS}"
    --qwen-model-id "${QWEN_MODEL_ID}"
    --output-dir "${OUTPUT_TASK2_DERM7PT}"
  )

  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${RESUME_TASK2_DERM7PT}" ]]; then
    CMD_ARGS+=(--resume "${OUTPUT_TASK2_DERM7PT}/${RESUME_TASK2_DERM7PT}")
  fi
  if [[ "${RETRY_ERRORS}" == "true" ]]; then
    CMD_ARGS+=(--retry-errors)
  fi

  build_critic_args CMD_ARGS

  echo "Executing: ${PYTHON_BIN} ${CMD_ARGS[*]}"

  if "${PYTHON_BIN}" "${CMD_ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_TASK2_DERM7PT}/run_$(date +%Y%m%d_%H%M%S).log"; then
    send_telegram "✅ <b>[1/3] Task2 Derm7pt_100 Critic 完成</b>"
  else
    send_telegram "❌ <b>[1/3] Task2 Derm7pt_100 Critic 失败</b>"
    FAILED_TASKS+=("Task2_Derm7pt")
  fi
else
  echo ""
  echo "[1/3] Task2 Derm7pt_100 - SKIPPED"
fi

# =============================================================================
# [2/3] Task 2 - SkinCon 100 samples
# =============================================================================
if [[ "${SKIP_TASK2_SKINCON}" != "true" ]]; then
  echo ""
  echo "================================================================================"
  echo "[2/3] Task 2 Concept Annotation - SkinCon 100 Samples (Critic)"
  echo "================================================================================"

  mkdir -p "${OUTPUT_TASK2_SKINCON}"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  send_telegram "▶️ <b>[2/3] Task2 SkinCon_100 Critic 开始</b>
🕐 ${start_time}"

  CMD_ARGS=(
    "${PROJECT_ROOT}/scripts/run_task2_concept_agent.py"
    --dataset skincon
    --sample-size "100"
    --device "${DEVICE}"
    --model-name "${MODEL_NAME}"
    --enabled-tools "${ENABLED_TOOLS}"
    --qwen-model-id "${QWEN_MODEL_ID}"
    --output-dir "${OUTPUT_TASK2_SKINCON}"
  )

  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${RESUME_TASK2_SKINCON}" ]]; then
    CMD_ARGS+=(--resume "${OUTPUT_TASK2_SKINCON}/${RESUME_TASK2_SKINCON}")
  fi
  if [[ "${RETRY_ERRORS}" == "true" ]]; then
    CMD_ARGS+=(--retry-errors)
  fi

  build_critic_args CMD_ARGS

  echo "Executing: ${PYTHON_BIN} ${CMD_ARGS[*]}"

  if "${PYTHON_BIN}" "${CMD_ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_TASK2_SKINCON}/run_$(date +%Y%m%d_%H%M%S).log"; then
    send_telegram "✅ <b>[2/3] Task2 SkinCon_100 Critic 完成</b>"
  else
    send_telegram "❌ <b>[2/3] Task2 SkinCon_100 Critic 失败</b>"
    FAILED_TASKS+=("Task2_SkinCon")
  fi
else
  echo ""
  echo "[2/3] Task2 SkinCon_100 - SKIPPED"
fi

# =============================================================================
# [3/3] Task 3 - SkinCAP 100 samples
# =============================================================================
if [[ "${SKIP_TASK3_SKINCAP}" != "true" ]]; then
  echo ""
  echo "================================================================================"
  echo "[3/3] Task 3 Captioning - SkinCAP 100 Samples (Critic)"
  echo "================================================================================"

  mkdir -p "${OUTPUT_TASK3_SKINCAP}"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  send_telegram "▶️ <b>[3/3] Task3 SkinCAP_100 Critic 开始</b>
🕐 ${start_time}"

  CMD_ARGS=(
    "${PROJECT_ROOT}/scripts/run_task3_captioning_agent.py"
    --device "${DEVICE}"
    --model-name "${MODEL_NAME}"
    --enabled-tools "${ENABLED_TOOLS}"
    --qwen-model-id "${QWEN_MODEL_ID}"
    --output-dir "${OUTPUT_TASK3_SKINCAP}"
  )

  if [[ "${MAX_SAMPLES}" != "0" ]]; then
    CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${RESUME_TASK3_SKINCAP}" ]]; then
    CMD_ARGS+=(--resume "${OUTPUT_TASK3_SKINCAP}/${RESUME_TASK3_SKINCAP}")
  fi
  if [[ "${RETRY_ERRORS}" == "true" ]]; then
    CMD_ARGS+=(--retry-errors)
  fi

  build_critic_args CMD_ARGS

  echo "Executing: ${PYTHON_BIN} ${CMD_ARGS[*]}"

  if "${PYTHON_BIN}" "${CMD_ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_TASK3_SKINCAP}/run_$(date +%Y%m%d_%H%M%S).log"; then
    send_telegram "✅ <b>[3/3] Task3 SkinCAP_100 Critic 完成</b>"
  else
    send_telegram "❌ <b>[3/3] Task3 SkinCAP_100 Critic 失败</b>"
    FAILED_TASKS+=("Task3_SkinCAP")
  fi
else
  echo ""
  echo "[3/3] Task3 SkinCAP_100 - SKIPPED"
fi

# =============================================================================
# Summary
# =============================================================================
overall_end="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "================================================================================"
echo "ALL DONE"
echo "================================================================================"
echo "Started: ${overall_start}"
echo "Ended:   ${overall_end}"
echo "Output:  ${OUTPUT_BASE}"

if [[ ${#FAILED_TASKS[@]} -eq 0 ]]; then
  echo "Status:  ALL 3 sub-tasks completed successfully"
  send_telegram "🎉 <b>Task2+3 Agent Full+Critic 全部完成</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
🧰 工具: <code>${ENABLED_TOOLS}</code>
🔍 Critic: ON | Reinject: ${CRITIC_IMAGE_REINJECT}
🕐 开始: ${overall_start}
🕑 结束: ${overall_end}
📁 输出: <code>${OUTPUT_BASE}</code>"
else
  echo "Status:  ${#FAILED_TASKS[@]} sub-task(s) FAILED: ${FAILED_TASKS[*]}"
  send_telegram "⚠️ <b>Task2+3 Agent Full+Critic 部分失败</b>
🖥 主机: ${HOSTNAME}
❌ 失败: ${FAILED_TASKS[*]}
🕐 开始: ${overall_start}
🕑 结束: ${overall_end}
📁 输出: <code>${OUTPUT_BASE}</code>"
fi

echo ""
echo "Output structure:"
echo "  results/agent_critic/"
echo "    task2_concept/derm7pt_100/{timestamp}/"
echo "    task2_concept/skincon_100/{timestamp}/"
echo "    task3_captioning/skincap_100/{timestamp}/"
