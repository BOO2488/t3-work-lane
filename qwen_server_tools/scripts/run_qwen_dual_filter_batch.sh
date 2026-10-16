#!/usr/bin/env bash
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

# Qwen-VL 双任务初筛：一次读取/抽帧，同一张 center_camera_fov30 contact sheet 分别判断：
# 1) 城市施工道路是否影响自车路径
# 2) 潮汐车道/可逆车道

INROOT=${INROOT:-/mnt/data_clip/t3_origin/dlp_data_analysis}
OUT=${OUT:-${BUNDLE_ROOT}/outputs/.state}
PY=${PY:-python3}
SCRIPT=${SCRIPT:-${BUNDLE_ROOT}/scripts/qwen_dual_filter_batch.py}
EAS_ENV_SCRIPT=${EAS_ENV_SCRIPT:-${BUNDLE_ROOT}/model/run_qwen_generate_df_qa_q35_eas_onefile_semantic_nobehind.sh}
WORKERS=${WORKERS:-128}
STRIDE=${STRIDE:-200}
CAMERA=${CAMERA:-camera/center_camera_fov30}
THUMB_WIDTH=${THUMB_WIDTH:-384}
TIMEOUT=${TIMEOUT:-30}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-360}
MODEL=${MODEL:-Qwen3.6-35B-A3B}
ROUTE=${ROUTE:-chat_completions}
LIMIT=${LIMIT:-0}
START=${START:-0}
DONE_JSONL=${DONE_JSONL:-""}
EXCLUDE_PREFIX=${EXCLUDE_PREFIX:-""}
LANE_OUTPUT_DIR=${LANE_OUTPUT_DIR:-${BUNDLE_ROOT}/outputs/lane}
WORK_OUTPUT_DIR=${WORK_OUTPUT_DIR:-${BUNDLE_ROOT}/outputs/work}

if [[ ! -d "${INROOT}" ]]; then
  echo "[ERROR] INROOT does not exist: ${INROOT}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] SCRIPT not found: ${SCRIPT}" >&2
  exit 2
fi
mkdir -p "${OUT}" "${OUT}/logs" "${LANE_OUTPUT_DIR}" "${WORK_OUTPUT_DIR}"

if [[ -z "${EAS_ENDPOINT:-}" ]]; then
  EAS_ENDPOINT=$(sed -n "s/^export EAS_ENDPOINT='\(.*\)'/\1/p" "${EAS_ENV_SCRIPT}" | head -n1)
  export EAS_ENDPOINT
fi
if [[ -z "${EAS_TOKEN:-}" ]]; then
  EAS_TOKEN=$(sed -n "s/^export EAS_TOKEN='\(.*\)'/\1/p" "${EAS_ENV_SCRIPT}" | head -n1)
  export EAS_TOKEN
fi
if [[ -z "${EAS_ENDPOINT:-}" || -z "${EAS_TOKEN:-}" ]]; then
  echo "[ERROR] Failed to load EAS_ENDPOINT/EAS_TOKEN from ${EAS_ENV_SCRIPT}" >&2
  exit 2
fi

export EAS_MODEL="${MODEL}"
export EAS_PREFER_ROUTE="${ROUTE}"
export EAS_TIMEOUT="${TIMEOUT}"

JSONL=${JSONL:-${OUT}/qwen_dual_filter.jsonl}
JSON=${JSON:-${OUT}/qwen_dual_filter.json}
TXT=${TXT:-${OUT}/qwen_dual_filter.txt}
ERROR_JSONL=${ERROR_JSONL:-${OUT}/qwen_dual_filter_errors.jsonl}
RUN_LOG=${RUN_LOG:-${OUT}/logs/run.log}

ARGS=(
  "${SCRIPT}"
  --root "${INROOT}"
  --jsonl "${JSONL}"
  --json "${JSON}"
  --txt "${TXT}"
  --error-jsonl "${ERROR_JSONL}"
  --camera "${CAMERA}"
  --stride "${STRIDE}"
  --workers "${WORKERS}"
  --thumb-width "${THUMB_WIDTH}"
  --timeout "${TIMEOUT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --lane-output-dir "${LANE_OUTPUT_DIR}"
  --work-output-dir "${WORK_OUTPUT_DIR}"
)
[[ "${LIMIT}" != "0" ]] && ARGS+=(--limit "${LIMIT}")
[[ "${START}" != "0" ]] && ARGS+=(--start "${START}")
[[ -n "${DONE_JSONL}" ]] && ARGS+=(--done-jsonl "${DONE_JSONL}")
[[ -n "${EXCLUDE_PREFIX}" ]] && ARGS+=(--exclude-prefix "${EXCLUDE_PREFIX}")

echo "[INFO] start: $(date '+%F %T')" | tee -a "${RUN_LOG}"
echo "[INFO] inroot=${INROOT}" | tee -a "${RUN_LOG}"
echo "[INFO] out=${OUT}" | tee -a "${RUN_LOG}"
echo "[INFO] jsonl=${JSONL}" | tee -a "${RUN_LOG}"
echo "[INFO] json=${JSON}" | tee -a "${RUN_LOG}"
echo "[INFO] txt=${TXT}" | tee -a "${RUN_LOG}"
echo "[INFO] error_jsonl=${ERROR_JSONL}" | tee -a "${RUN_LOG}"
echo "[INFO] workers=${WORKERS} stride=${STRIDE} limit=${LIMIT} start=${START}" | tee -a "${RUN_LOG}"
echo "[INFO] camera=${CAMERA}" | tee -a "${RUN_LOG}"
echo "[INFO] thumb_width=${THUMB_WIDTH}" | tee -a "${RUN_LOG}"
echo "[INFO] lane_output_dir=${LANE_OUTPUT_DIR}" | tee -a "${RUN_LOG}"
echo "[INFO] work_output_dir=${WORK_OUTPUT_DIR}" | tee -a "${RUN_LOG}"
echo "[INFO] exclude_prefix=${EXCLUDE_PREFIX}" | tee -a "${RUN_LOG}"
echo "[INFO] eas_endpoint=${EAS_ENDPOINT}" | tee -a "${RUN_LOG}"
echo "[INFO] eas_token=***" | tee -a "${RUN_LOG}"

nohup "${PY}" "${ARGS[@]}" >> "${RUN_LOG}" 2>&1 &
PID=$!
if ! printf "%s\n" "${PID}" > "${OUT}/run.pid"; then
  echo "[WARN] failed to write ${OUT}/run.pid" | tee -a "${RUN_LOG}"
fi
echo "[INFO] pid=${PID}" | tee -a "${RUN_LOG}"
echo "[INFO] tail -f ${RUN_LOG}"
