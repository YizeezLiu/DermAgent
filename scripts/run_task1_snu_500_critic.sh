#!/usr/bin/env bash
set -eEuo pipefail

# =============================================================================
# SNU 500 Critic — Narrowed Re-diagnosis Benchmark
# =============================================================================
#
# Waits for GPU 0 memory to drop below threshold, then runs SNU 500
# full-sample diagnosis with critic + narrowed re-diagnosis enabled,
# using 2 GPUs (0,1).
#
# Usage:
#   bash scripts/run_task1_snu_500_critic.sh
#
#   # Skip GPU wait (GPUs already free)
#   SKIP_WAIT=true bash scripts/run_task1_snu_500_critic.sh
#
#   # Override samples (quick test)
#   MAX_SAMPLES=100 bash scripts/run_task1_snu_500_critic.sh
#
#   # Resume a previous run
#   RESUME=YYYYMMDD_HHMMSS bash scripts/run_task1_snu_500_critic.sh
#
#   # Only retry errors from a previous run
#   RESUME=YYYYMMDD_HHMMSS RETRY_ERRORS=true bash scripts/run_task1_snu_500_critic.sh
#
# After the run, analyze narrowed re-diagnosis:
#   LOG=results/agent_critic/task1_diagnosis/SNU_500/run_*.log
#   grep '\[NarrowedDx\]' $LOG
#   grep -c '\[NarrowedDx\] Shortlist built' $LOG   # shortlists built
#   grep -c '\[NarrowedDx\] Blocked tool'    $LOG   # tools blocked in Round 2
#   grep -c '\[NarrowedDx\] Injecting'       $LOG   # feedback injections
#   grep -c '\[NarrowedDx\] Chatbot'         $LOG   # narrowed chatbot rounds
#

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============== Load .env file (if exists) ==============
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

# ============== GPU Monitor Configuration ==============
MONITOR_GPU="${MONITOR_GPU:-0}"           # GPU to monitor for availability
THRESHOLD="${THRESHOLD:-10}"               # Memory % threshold to start
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"    # Check every N seconds (5 min)
SKIP_WAIT="${SKIP_WAIT:-false}"            # Skip GPU wait

# ============== Agent Configuration ==============
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
DEVICE="${DEVICE:-cuda}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"            # 0 = all 500 samples
USE_FLASH_ATTN="${USE_FLASH_ATTN:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Resume
RESUME="${RESUME:-}"
RETRY_ERRORS="${RETRY_ERRORS:-false}"

# Critic: ALWAYS enabled
ENABLE_CRITIC="true"
CRITIC_LLM_SUMMARY="${CRITIC_LLM_SUMMARY:-false}"
CRITIC_IMAGE_REINJECT="${CRITIC_IMAGE_REINJECT:-false}"
MAX_CRITIC_RETRIES="${MAX_CRITIC_RETRIES:-2}"

# Prompt variant
PROMPT_VARIANT="${PROMPT_VARIANT:-v2}"

# Two GPUs
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/results/agent_critic/task1_diagnosis/SNU_500"
ENABLED_TOOLS="${ENABLED_TOOLS:-panderm,make,dermogpt_vqa,rag,text_rag}"

# Log level: INFO to see [NarrowedDx] messages
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "================================================================================"
echo "SNU 500 Critic — Narrowed Re-diagnosis Benchmark"
echo "================================================================================"
echo "PROJECT_ROOT:          ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME:            ${MODEL_NAME}"
echo "DEVICE:                ${DEVICE}"
echo "USE_FLASH_ATTN:        ${USE_FLASH_ATTN}"
echo "MAX_SAMPLES:           ${MAX_SAMPLES} (0=all)"
echo "ENABLED_TOOLS:         ${ENABLED_TOOLS}"
echo "ENABLE_CRITIC:         ${ENABLE_CRITIC}"
echo "CRITIC_LLM_SUMMARY:    ${CRITIC_LLM_SUMMARY}"
echo "CRITIC_IMAGE_REINJECT:  ${CRITIC_IMAGE_REINJECT}"
echo "MAX_CRITIC_RETRIES:    ${MAX_CRITIC_RETRIES}"
echo "PROMPT_VARIANT:        ${PROMPT_VARIANT}"
echo "OUTPUT_DIR:            ${OUTPUT_DIR}"
echo "LOG_LEVEL:             ${LOG_LEVEL}"
echo "MONITOR_GPU:           ${MONITOR_GPU}"
echo "THRESHOLD:             ${THRESHOLD}%"
echo "SKIP_WAIT:             ${SKIP_WAIT}"
if [[ -n "${RESUME}" ]]; then
  echo "RESUME:                ${RESUME}"
  echo "RETRY_ERRORS:          ${RETRY_ERRORS}"
fi
echo "================================================================================"

# ============== GPU Monitor ==============
get_gpu_memory_percent() {
    local gpu_id="$1"
    local mem_info
    mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null)
    if [[ -z "${mem_info}" ]]; then
        echo "100"
        return
    fi
    local mem_used mem_total
    mem_used=$(echo "${mem_info}" | cut -d',' -f1 | tr -d ' ')
    mem_total=$(echo "${mem_info}" | cut -d',' -f2 | tr -d ' ')
    if [[ "${mem_total}" -eq 0 ]]; then
        echo "100"
        return
    fi
    echo $(( mem_used * 100 / mem_total ))
}

if [[ "${SKIP_WAIT}" != "true" ]]; then
    echo ""
    echo "[GPU Monitor] Watching GPU ${MONITOR_GPU}, waiting for memory < ${THRESHOLD}%..."
    echo "[GPU Monitor] Checking every ${CHECK_INTERVAL}s. Set SKIP_WAIT=true to skip."
    echo ""

    while true; do
        mem_percent=$(get_gpu_memory_percent "${MONITOR_GPU}")
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[${timestamp}] GPU ${MONITOR_GPU} memory: ${mem_percent}%"

        if [[ "${mem_percent}" -lt "${THRESHOLD}" ]]; then
            echo ""
            echo "[${timestamp}] GPU ${MONITOR_GPU} memory < ${THRESHOLD}%! Starting benchmark..."
            break
        fi
        sleep "${CHECK_INTERVAL}"
    done
else
    echo ""
    echo "[GPU Monitor] Skipped (SKIP_WAIT=true)"
fi

# ============== Pre-run checks ==============
echo ""
echo "[Pre-run checks]"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
free -h || true
df -h . || true

# ============== Run Benchmark ==============
echo ""
echo "================================================================================"
echo "[Starting SNU 500 Critic Benchmark]"
echo "================================================================================"

start_time="$(date '+%Y-%m-%d %H:%M:%S')"
RUN_LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${OUTPUT_DIR}"

# Build command
CMD_ARGS=(
  "${PROJECT_ROOT}/scripts/run_task1_snu_500_agent.py"
  --device "${DEVICE}"
  --model-name "${MODEL_NAME}"
  --enabled-tools "${ENABLED_TOOLS}"
  --qwen-model-id "${QWEN_MODEL_ID}"
  --output-dir "${OUTPUT_DIR}"
  --enable-critic
  --max-critic-retries "${MAX_CRITIC_RETRIES}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  CMD_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${USE_FLASH_ATTN}" == "true" ]]; then
  CMD_ARGS+=(--use-flash-attn)
fi

if [[ -n "${RESUME}" ]]; then
  CMD_ARGS+=(--resume "${OUTPUT_DIR}/${RESUME}")
fi

if [[ "${RETRY_ERRORS}" == "true" ]]; then
  CMD_ARGS+=(--retry-errors)
fi

if [[ "${CRITIC_LLM_SUMMARY}" == "true" ]]; then
  CMD_ARGS+=(--critic-llm-summary)
fi

if [[ "${CRITIC_IMAGE_REINJECT}" == "true" ]]; then
  CMD_ARGS+=(--critic-image-reinject)
fi

if [[ "${PROMPT_VARIANT}" != "default" ]]; then
  CMD_ARGS+=(--prompt "${PROMPT_VARIANT}")
fi

echo "Executing: ${PYTHON_BIN} ${CMD_ARGS[*]}"
echo "Log file:  ${RUN_LOG}"
echo "================================================================================"

if "${PYTHON_BIN}" "${CMD_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"; then
    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "================================================================================"
    echo "Critic Benchmark finished successfully."
    echo "================================================================================"

    # ============== Post-run: Narrowed Re-diagnosis Analysis ==============
    echo ""
    echo "===== Narrowed Re-diagnosis Summary ====="
    echo ""

    n_shortlist=$(grep -c '\[NarrowedDx\] Shortlist built' "${RUN_LOG}" 2>/dev/null || echo "0")
    n_blocked=$(grep -c '\[NarrowedDx\] Blocked tool' "${RUN_LOG}" 2>/dev/null || echo "0")
    n_inject=$(grep -c '\[NarrowedDx\] Injecting' "${RUN_LOG}" 2>/dev/null || echo "0")
    n_chatbot=$(grep -c '\[NarrowedDx\] Chatbot' "${RUN_LOG}" 2>/dev/null || echo "0")

    echo "Shortlists built:           ${n_shortlist}"
    echo "Tools blocked (Round 2):    ${n_blocked}"
    echo "Feedback injections:        ${n_inject}"
    echo "Chatbot narrowed rounds:    ${n_chatbot}"
    echo ""

    echo "===== Sample Shortlists (first 20) ====="
    grep '\[NarrowedDx\] Shortlist built' "${RUN_LOG}" 2>/dev/null | head -20 || echo "(none)"
    echo ""

    echo "===== All [NarrowedDx] Events ====="
    grep '\[NarrowedDx\]' "${RUN_LOG}" 2>/dev/null | head -50 || echo "(none)"
    echo ""

    echo "[SUCCESS] Benchmark completed."
else
    echo "[FAILED] Critic Benchmark failed."
    exit 1
fi

echo "================================================================================"
echo "Log: ${RUN_LOG}"
echo "Analyze: grep '\\[NarrowedDx\\]' ${RUN_LOG}"
echo "================================================================================"
