#!/usr/bin/env bash
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

# Qwen-VL 潮汐车道/可逆车道初筛启动脚本
# 不覆盖道路施工结果，默认输出到 /mnt/wzb11/qwen_tidal_lane_run

INROOT=${INROOT:-""}
OUT=${OUT:-/mnt/wzb11/qwen_tidal_lane_run}
PY=${PY:-python3}
SCRIPT=${SCRIPT:-${BUNDLE_ROOT}/scripts/qwen_tidal_lane_batch.py}
EAS_ENV_SCRIPT=${EAS_ENV_SCRIPT:-${BUNDLE_ROOT}/model/run_qwen_generate_df_qa_q35_eas_onefile_semantic_nobehind.sh}
WORKERS=${WORKERS:-64}
STRIDE=${STRIDE:-100}
CAMERA=${CAMERA:-camera/center_camera_fov30}
TIMEOUT=${TIMEOUT:-600}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-260}
MODEL=${MODEL:-Qwen3.6-35B-A3B}
ROUTE=${ROUTE:-chat_completions}
LIMIT=${LIMIT:-0}
START=${START:-0}
DONE_JSONL=${DONE_JSONL:-""}
EXCLUDE_PREFIX=${EXCLUDE_PREFIX:-""}
REVIEW_DIR=${REVIEW_DIR:-${OUT}/review_frames}

if [[ -z "${INROOT}" ]]; then
  echo "[ERROR] INROOT is empty. Example:" >&2
  echo "  INROOT=/mnt/data_clip/t3_origin/dlp_data_analysis bash $0" >&2
  exit 2
fi
if [[ ! -d "${INROOT}" ]]; then
  echo "[ERROR] INROOT does not exist: ${INROOT}" >&2
  exit 2
fi
mkdir -p "${OUT}" "${OUT}/logs"

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

JSONL=${JSONL:-${OUT}/qwen_tidal_lane.jsonl}
JSON=${JSON:-${OUT}/qwen_tidal_lane.json}
TXT=${TXT:-${OUT}/qwen_tidal_lane.txt}
RUN_LOG=${RUN_LOG:-${OUT}/logs/run.log}

ARGS=(
  "${SCRIPT}"
  --root "${INROOT}"
  --jsonl "${JSONL}"
  --json "${JSON}"
  --txt "${TXT}"
  --camera "${CAMERA}"
  --stride "${STRIDE}"
  --workers "${WORKERS}"
  --timeout "${TIMEOUT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --review-dir "${REVIEW_DIR}"
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
echo "[INFO] workers=${WORKERS} stride=${STRIDE} limit=${LIMIT} start=${START}" | tee -a "${RUN_LOG}"
echo "[INFO] camera=${CAMERA}" | tee -a "${RUN_LOG}"
echo "[INFO] review_dir=${REVIEW_DIR}" | tee -a "${RUN_LOG}"
echo "[INFO] exclude_prefix=${EXCLUDE_PREFIX}" | tee -a "${RUN_LOG}"
echo "[INFO] eas_endpoint=${EAS_ENDPOINT}" | tee -a "${RUN_LOG}"
echo "[INFO] eas_token=***" | tee -a "${RUN_LOG}"

nohup "${PY}" "${ARGS[@]}" >> "${RUN_LOG}" 2>&1 &
PID=$!
echo "${PID}" > "${OUT}/run.pid"
echo "[INFO] pid=${PID}" | tee -a "${RUN_LOG}"
echo "[INFO] tail -f ${RUN_LOG}"
