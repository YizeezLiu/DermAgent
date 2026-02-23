#!/usr/bin/env bash
set -eEuo pipefail

# =============================================================================
# Task 3 Captioning - Leave-One-Out (LOO) Ablation Study
# =============================================================================
#
# Removes one tool at a time from the full 6-tool set to measure each tool's
# individual contribution. Each experiment runs serially using 2 GPUs.
#
# Full tool set: dermogpt_vqa, make, panderm, rag, text_rag, ontology
#
# Usage:
#   bash scripts/run_task3_loo_ablation.sh
#   CUDA_VISIBLE_DEVICES="2,3" bash scripts/run_task3_loo_ablation.sh
#   MAX_SAMPLES=10 bash scripts/run_task3_loo_ablation.sh
#

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
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 means no limit
USE_FLASH_ATTN="${USE_FLASH_ATTN:-false}"
HOSTNAME="$(hostname)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Use 2 GPUs (default: 0,1)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

BASE_OUTPUT_DIR="${PROJECT_ROOT}/results/agent_ablation/task3_captioning/skincap_100"

# =============================================================================
# Leave-One-Out Experiments
# Each entry: "REMOVED_TOOL_NAME  ENABLED_TOOLS_SORTED"
# =============================================================================
declare -A LOO_EXPERIMENTS=(
  ["dermogpt_vqa"]="make,ontology,panderm,rag,text_rag"
  ["make"]="dermogpt_vqa,ontology,panderm,rag,text_rag"
  ["panderm"]="dermogpt_vqa,make,ontology,rag,text_rag"
  ["rag"]="dermogpt_vqa,make,ontology,panderm,text_rag"
  ["text_rag"]="dermogpt_vqa,make,ontology,panderm,rag"
  ["ontology"]="dermogpt_vqa,make,panderm,rag,text_rag"
)

# Execution order (deterministic)
LOO_ORDER=("dermogpt_vqa" "make" "panderm" "rag" "text_rag" "ontology")

echo "================================================================================"
echo "Task 3 Captioning - Leave-One-Out Ablation Study"
echo "================================================================================"
echo "PROJECT_ROOT:         ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME:           ${MODEL_NAME}"
echo "DEVICE:               ${DEVICE}"
echo "USE_FLASH_ATTN:       ${USE_FLASH_ATTN}"
echo "MAX_SAMPLES:          ${MAX_SAMPLES:-all}"
echo "BASE_OUTPUT_DIR:      ${BASE_OUTPUT_DIR}"
echo "Experiments:          ${#LOO_ORDER[@]} (remove one tool at a time)"
echo "--------------------------------------------------------------------------------"
for removed in "${LOO_ORDER[@]}"; do
  echo "  w/o ${removed} -> enabled: ${LOO_EXPERIMENTS[$removed]}"
done
echo "--------------------------------------------------------------------------------"

echo "[Pre-run checks]"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found (ok if CPU-only)"
free -h || true
df -h /mnt/hdd/sdb/ || true

# ============== Helper: wait for GPU availability ==============
wait_for_gpu() {
  local gpu_id=0
  local threshold_pct=10
  local interval=300
  
  echo "Monitoring GPU ${gpu_id} memory (target: < ${threshold_pct}% usage)..."
  while true; do
    # Get used and total memory in MiB
    local mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i ${gpu_id})
    local used_mem=$(echo ${mem_info} | cut -d',' -f1 | xargs)
    local total_mem=$(echo ${mem_info} | cut -d',' -f2 | xargs)
    
    # Calculate percentage usage
    local usage_pct=$(( used_mem * 100 / total_mem ))
    
    if [[ ${usage_pct} -lt ${threshold_pct} ]]; then
      echo "✅ GPU ${gpu_id} memory usage is ${usage_pct}% (${used_mem}/${total_mem} MiB), starting experiment."
      break
    fi
    echo "⏳ GPU ${gpu_id} memory usage is ${usage_pct}% (${used_mem}/${total_mem} MiB), waiting ${interval}s..."
    sleep ${interval}
  done
}

echo "--------------------------------------------------------------------------------"
echo "[Starting Leave-One-Out Ablation]"

# Wait for GPU 0 availability before starting
wait_for_gpu

start_time_total="$(date '+%Y-%m-%d %H:%M:%S')"
send_telegram "🚀 <b>Task3 LOO Ablation 开始 (串行, ${CUDA_VISIBLE_DEVICES})</b>
🖥 主机: ${HOSTNAME}
🤖 模型: <code>${MODEL_NAME}</code>
🎮 GPUs: <code>${CUDA_VISIBLE_DEVICES}</code>
🧪 实验: ${#LOO_ORDER[@]} 个 (leave-one-out)
📁 输出: <code>${BASE_OUTPUT_DIR}</code>"

SUCCESSFUL=()
FAILED=()
TOTAL=${#LOO_ORDER[@]}
CURRENT=0

for REMOVED in "${LOO_ORDER[@]}"; do
    CURRENT=$((CURRENT + 1))
    ENABLED="${LOO_EXPERIMENTS[$REMOVED]}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/loo_wo_${REMOVED}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "================================================================================"
    echo "[${CURRENT}/${TOTAL}] w/o ${REMOVED}"
    echo "  Enabled tools: ${ENABLED}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "================================================================================"

    start_time_exp="$(date '+%Y-%m-%d %H:%M:%S')"

    # Build command
    CMD_ARGS=(
      "${PROJECT_ROOT}/scripts/run_task3_captioning_agent.py"
      --device "${DEVICE}"
      --model-name "${MODEL_NAME}"
      --enabled-tools "${ENABLED}"
      --qwen-model-id "${QWEN_MODEL_ID}"
      --output-dir "${OUTPUT_DIR}"
    )

    if [[ "${MAX_SAMPLES}" != "0" ]]; then
      CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
    fi

    if [[ "${USE_FLASH_ATTN}" == "true" ]]; then
      CMD_ARGS+=(--use-flash-attn)
    fi

    LOG_FILE="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    echo "Executing: ${PYTHON_BIN} ${CMD_ARGS[*]}"

    if "${PYTHON_BIN}" "${CMD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"; then
        echo "✅ w/o ${REMOVED} finished successfully."
        SUCCESSFUL+=("${REMOVED}")

        send_telegram "✅ <b>Task3 LOO [${CURRENT}/${TOTAL}] 完成: w/o ${REMOVED}</b>
🖥 主机: ${HOSTNAME}
🕐 开始: ${start_time_exp}
🕑 结束: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "❌ w/o ${REMOVED} failed."
        FAILED+=("${REMOVED}")

        send_telegram "❌ <b>Task3 LOO [${CURRENT}/${TOTAL}] 失败: w/o ${REMOVED}</b>
🖥 主机: ${HOSTNAME}
🕐 开始: ${start_time_exp}
🕑 失败: $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # GPU cleanup between runs
    if [[ ${CURRENT} -lt ${TOTAL} ]]; then
        echo "Waiting 10s for GPU memory cleanup..."
        sleep 10
    fi
done

# =============================================================================
# Summary
# =============================================================================
end_time_total="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "================================================================================"
echo "Leave-One-Out Ablation Results"
echo "================================================================================"
for REMOVED in "${LOO_ORDER[@]}"; do
    if printf '%s\n' "${SUCCESSFUL[@]}" | grep -qx "${REMOVED}"; then
        echo "  ✅ w/o ${REMOVED}"
    else
        echo "  ❌ w/o ${REMOVED}"
    fi
done
echo ""
echo "Outputs: ${BASE_OUTPUT_DIR}/loo_wo_*/"
echo "  Successful: ${#SUCCESSFUL[@]}/${TOTAL}"
echo "  Failed:     ${#FAILED[@]}/${TOTAL}"
echo "================================================================================"

if [[ ${#FAILED[@]} -eq 0 ]]; then
  send_telegram "🎉 <b>Task3 LOO Ablation 全部完成!</b>
🖥 主机: ${HOSTNAME}
🕐 开始: ${start_time_total}
🕑 结束: ${end_time_total}
✅ 全部 ${TOTAL} 个实验成功
📁 输出: <code>${BASE_OUTPUT_DIR}/loo_wo_*/</code>"
else
  send_telegram "⚠️ <b>Task3 LOO Ablation 完成 (有失败)</b>
🖥 主机: ${HOSTNAME}
🕐 开始: ${start_time_total}
🕑 结束: ${end_time_total}
✅ 成功: ${#SUCCESSFUL[@]}/${TOTAL}
❌ 失败: ${#FAILED[@]}/${TOTAL} (${FAILED[*]})
📁 输出: <code>${BASE_OUTPUT_DIR}/loo_wo_*/</code>"
fi

# Exit with failure if any experiment failed
[[ ${#FAILED[@]} -eq 0 ]]
