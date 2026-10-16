#!/usr/bin/env bash
set -Eeuo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EAS_TOKEN="${EAS_TOKEN:-}"
export EAS_ENDPOINT="${EAS_ENDPOINT:-http://1393074507312057.cn-wulanchabu.pai-eas.aliyuncs.com/api/predict/vqa_test_qwen3_6}"
export EAS_MODEL="${EAS_MODEL:-Qwen3.6-35B-A3B}"
export EAS_TIMEOUT="${EAS_TIMEOUT:-600}"
export EAS_PREFER_ROUTE="${EAS_PREFER_ROUTE:-chat_completions}"

ROOT="${ROOT:-/mnt/data_clip/t3_origin/dlp_data_analysis}"
EXCLUDE="${EXCLUDE:-/mnt/data_clip/t3_origin/dlp_data_analysis/data_collection/gt_data/pilotGtParser/dongfeng_t3/20260522}"
KEEP_DIR="${KEEP_DIR:-/mnt/wzb11/道路施工-all}"
RESULT_DIR="${RESULT_DIR:-/mnt/wzb11/mihomo/qwen_construction_filter_results}"
WORKERS="${WORKERS:-32}"
MAX_CLIPS="${MAX_CLIPS:-0}"
SAMPLES_PER_CLIP="${SAMPLES_PER_CLIP:-0}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-100}"

LOG_DIR="${LOG_DIR:-${RESULT_DIR}/run_logs}"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/construction_filter_${TS}.log"
LATEST_LOG="${LOG_DIR}/latest.log"
: > "${LATEST_LOG}"

exec > >(tee -a "${RUN_LOG}" "${LATEST_LOG}") 2>&1
echo "[INFO] run_log=${RUN_LOG}"
echo "[INFO] latest_log=${LATEST_LOG}"

python "${SCRIPT_DIR}/qwen_construction_filter_stream.py" \
  --root "${ROOT}" \
  --exclude "${EXCLUDE}" \
  --keep-dir "${KEEP_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --workers "${WORKERS}" \
  --max-clips "${MAX_CLIPS}" \
  --samples-per-clip "${SAMPLES_PER_CLIP}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --eas-endpoint "${EAS_ENDPOINT}" \
  --eas-token "${EAS_TOKEN}" \
  --eas-model "${EAS_MODEL}" \
  --eas-timeout "${EAS_TIMEOUT}" \
  --eas-prefer-route "${EAS_PREFER_ROUTE}"
