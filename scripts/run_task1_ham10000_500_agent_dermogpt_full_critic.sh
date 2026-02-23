#!/usr/bin/env bash
set -eEuo pipefail

# =============================================================================
# HAM10000 500-Sample Agent Benchmark Runner
# Full 6 Tools (VQA=DermoGPT-RL) + Critic + Image Reinjection
# =============================================================================
#
# Configuration:
#   - 6 Tools: panderm, make, dermogpt_vqa, rag, text_rag, ontology
#   - VQA: DermoGPT-RL (fine-tuned from Qwen3-VL-8B)
#   - Critic: Enabled (retry on low confidence / tool conflicts)
#   - Image Reinjection: Enabled (re-inject image when critic triggers retry)
#
# Usage:
#   bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#   MAX_SAMPLES=10 bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#
#   # Resume a previous run
#   RESUME=20260222_120000 bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#
#   # Resume and retry errors
#   RESUME=20260222_120000 RETRY_ERRORS=true bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#
#   # Override critic settings
#   MAX_CRITIC_RETRIES=3 CRITIC_LLM_SUMMARY=true bash scripts/run_task1_ham10000_500_agent_dermogpt_full_critic.sh
#
# Notes:
# - Output: results/agent_dermogpt_full_critic/task1_diagnosis/HAM10000_500/{timestamp}/
# - RESUME: Set to timestamp directory name to resume
# - RETRY_ERRORS: Set to true to only retry failed samples
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

# ============== Configuration ==============
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
DEVICE="${DEVICE:-cuda}"
# All 6 tools: panderm, make, dermogpt_vqa (DermoGPT-RL), rag, text_rag, ontology
ENABLED_TOOLS="${ENABLED_TOOLS:-panderm,make,dermogpt_vqa,rag,text_rag,ontology}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 means no limit
HOSTNAME="$(hostname)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Resume configuration
RESUME="${RESUME:-}"
RETRY_ERRORS="${RETRY_ERRORS:-false}"

# Critic configuration
ENABLE_CRITIC="${ENABLE_CRITIC:-true}"
CRITIC_LLM_SUMMARY="${CRITIC_LLM_SUMMARY:-false}"
CRITIC_IMAGE_REINJECT="${CRITIC_IMAGE_REINJECT:-true}"
MAX_CRITIC_RETRIES="${MAX_CRITIC_RETRIES:-2}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

OUTPUT_DIR="${PROJECT_ROOT}/results/agent_dermogpt_full_critic/task1_diagnosis/HAM10000_500"

echo "================================================================================"
echo "HAM10000 500-Sample Agent Benchmark"
echo "Full 6 Tools (DermoGPT-RL) + Critic + Image Reinjection"
echo "================================================================================"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "DEVICE: ${DEVICE}"
echo "ENABLED_TOOLS: ${ENABLED_TOOLS}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "ENABLE_CRITIC: ${ENABLE_CRITIC}"
echo "CRITIC_LLM_SUMMARY: ${CRITIC_LLM_SUMMARY}"
echo "CRITIC_IMAGE_REINJECT: ${CRITIC_IMAGE_REINJECT}"
echo "MAX_CRITIC_RETRIES: ${MAX_CRITIC_RETRIES}"
if [[ -n "${RESUME}" ]]; then
  echo "RESUME: ${RESUME}"
  echo "RETRY_ERRORS: ${RETRY_ERRORS}"
fi
echo "--------------------------------------------------------------------------------"

echo "[Pre-run checks]"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found (ok if CPU-only)"
free -h || true
df -h /mnt/hdd/sdb/ || true

echo "--------------------------------------------------------------------------------"
echo "[Running]"

start_time="$(date '+%Y-%m-%d %H:%M:%S')"
send_telegram "🚀 <b>Task1 Agent (HAM10000_500) DermoGPT Full 6T+Critic 开始</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
🧰 工具: <code>${ENABLED_TOOLS}</code>
🔍 Critic: ON | Summary: ${CRITIC_LLM_SUMMARY} | Reinject: ${CRITIC_IMAGE_REINJECT} | MaxRetries: ${MAX_CRITIC_RETRIES}
🧪 MAX_SAMPLES: <code>${MAX_SAMPLES}</code>
🕐 开始: ${start_time}
📁 输出: <code>${OUTPUT_DIR}</code>"

mkdir -p "${OUTPUT_DIR}"

# Build command arguments
CMD_ARGS=(
  "${PROJECT_ROOT}/scripts/run_task1_ham10000_500_agent.py"
  --device "${DEVICE}"
  --model-name "${MODEL_NAME}"
  --enabled-tools "${ENABLED_TOOLS}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${RESUME}" ]]; then
  CMD_ARGS+=(--resume "${OUTPUT_DIR}/${RESUME}")
fi

if [[ "${RETRY_ERRORS}" == "true" ]]; then
  CMD_ARGS+=(--retry-errors)
fi

# Critic flags
if [[ "${ENABLE_CRITIC}" == "true" ]]; then
  CMD_ARGS+=(--enable-critic)
fi

if [[ "${CRITIC_LLM_SUMMARY}" == "true" ]]; then
  CMD_ARGS+=(--critic-llm-summary)
fi

if [[ "${CRITIC_IMAGE_REINJECT}" == "true" ]]; then
  CMD_ARGS+=(--critic-image-reinject)
fi

if [[ -n "${MAX_CRITIC_RETRIES}" && "${MAX_CRITIC_RETRIES}" != "2" ]]; then
  CMD_ARGS+=(--max-critic-retries "${MAX_CRITIC_RETRIES}")
fi

if "${PYTHON_BIN}" "${CMD_ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"; then
  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  send_telegram "✅ <b>Task1 Agent (HAM10000_500) DermoGPT Full 6T+Critic 完成</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
🧰 工具: <code>${ENABLED_TOOLS}</code>
🔍 Critic: ON | Reinject: ${CRITIC_IMAGE_REINJECT}
🕐 开始: ${start_time}
🕑 结束: ${end_time}
📁 输出: <code>${OUTPUT_DIR}</code>"
else
  exit_code=$?
  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  send_telegram "❌ <b>Task1 Agent (HAM10000_500) DermoGPT Full 6T+Critic 失败</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
⚠️ 退出码: ${exit_code}
🕐 开始: ${start_time}
🕑 失败时间: ${end_time}
📁 输出: <code>${OUTPUT_DIR}</code>"
  exit "${exit_code}"
fi

echo "--------------------------------------------------------------------------------"
echo "DONE. Outputs saved to: ${OUTPUT_DIR}"
echo "  - {timestamp}/test_results_{timestamp}.txt"
echo "  - {timestamp}/predictions_{timestamp}.csv"
echo "  - {timestamp}/traces_summary_{timestamp}.json"
echo "  - {timestamp}/traces/"
