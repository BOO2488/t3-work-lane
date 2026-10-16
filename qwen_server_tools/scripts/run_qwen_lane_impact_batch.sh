#!/usr/bin/env bash
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

# =========================
# Qwen-VL 城市施工道路初筛启动脚本
# 目标：筛选“近中距离内影响自车行驶路径”的城市施工道路场景
# =========================

# 必填：待筛数据根目录。目录下需符合 */*/*/*_dlp_pilotGtParser 结构。
# 示例：INROOT=/mnt/data_clip/t3_origin/dlp_data_analysis/data_collection/gt_data/pilotGtParser/dongfeng_t3/20260523
INROOT=${INROOT:-""}

# 输出目录
OUT=${OUT:-/mnt/wzb11/qwen_lane_impact_run}

# Python 和脚本
PY=${PY:-python3}
SCRIPT=${SCRIPT:-${BUNDLE_ROOT}/scripts/qwen_lane_impact_batch.py}

# 自动复用上午 Qwen EAS 配置，不在日志中打印 token
EAS_ENV_SCRIPT=${EAS_ENV_SCRIPT:-${BUNDLE_ROOT}/model/run_qwen_generate_df_qa_q35_eas_onefile_semantic_nobehind.sh}

# 并发设置：这里是客户端并发请求数，不是 GPU 数。32 卡服务可先用 64，稳定后可调 96/128。
WORKERS=${WORKERS:-64}
STRIDE=${STRIDE:-100}
CAMERA=${CAMERA:-camera/center_camera_fov30}
TIMEOUT=${TIMEOUT:-600}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-260}
MODEL=${MODEL:-Qwen3.6-35B-A3B}
ROUTE=${ROUTE:-chat_completions}

# 可选：只跑一部分做验证。0 表示不限制。
LIMIT=${LIMIT:-0}
START=${START:-0}

# 可选：额外跳过已有 jsonl。若“不跑以前的”且 INROOT 是新目录，一般不用设置。
DONE_JSONL=${DONE_JSONL:-""}
EXCLUDE_PREFIX=${EXCLUDE_PREFIX:-""}
REVIEW_DIR=${REVIEW_DIR:-${OUT}/review_frames}

if [[ -z "${INROOT}" ]]; then
  echo "[ERROR] INROOT is empty. Example:" >&2
  echo "  INROOT=/path/to/new/pilotGtParser/root bash $0" >&2
  exit 2
fi

if [[ ! -d "${INROOT}" ]]; then
  echo "[ERROR] INROOT does not exist: ${INROOT}" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] SCRIPT not found: ${SCRIPT}" >&2
  exit 2
fi

mkdir -p "${OUT}" "${OUT}/logs"

# 如果当前环境没有 EAS_ENDPOINT/EAS_TOKEN，就从上午脚本中提取。
if [[ -z "${EAS_ENDPOINT:-}" ]]; then
  EAS_ENDPOINT=$(sed -n "s/^export EAS_ENDPOINT='\(.*\)'/\1/p" "${EAS_ENV_SCRIPT}" | head -n1)
  export EAS_ENDPOINT
fi
if [[ -z "${EAS_TOKEN:-}" ]]; then
  EAS_TOKEN=$(sed -n "s/^export EAS_TOKEN='\(.*\)'/\1/p" "${EAS_ENV_SCRIPT}" | head -n1)
  export EAS_TOKEN
fi

if [[ -z "${EAS_ENDPOINT:-}" || -z "${EAS_TOKEN:-}" ]]; then
  echo "[ERROR] Failed to load EAS_ENDPOINT/EAS_TOKEN." >&2
  echo "        Check EAS_ENV_SCRIPT=${EAS_ENV_SCRIPT}" >&2
  exit 2
fi

export EAS_MODEL="${MODEL}"
export EAS_PREFER_ROUTE="${ROUTE}"
export EAS_TIMEOUT="${TIMEOUT}"

JSONL=${JSONL:-${OUT}/qwen_lane_impact.jsonl}
JSON=${JSON:-${OUT}/qwen_lane_impact.json}
TXT=${TXT:-${OUT}/qwen_lane_impact.txt}
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

if [[ "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${START}" != "0" ]]; then
  ARGS+=(--start "${START}")
fi
if [[ -n "${DONE_JSONL}" ]]; then
  ARGS+=(--done-jsonl "${DONE_JSONL}")
fi
if [[ -n "${EXCLUDE_PREFIX}" ]]; then
  ARGS+=(--exclude-prefix "${EXCLUDE_PREFIX}")
fi

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
