#!/bin/bash
# =============================================================================
# Russell OHLCV + LLM A/B Pipeline Runner
#
# End-to-end flow:
#   1. Build Russell signal dumps
#   2. Build deterministic composite score CSVs
#   3. Run LLM A/B comparison pipeline
#
# Usage:
#   chmod +x run_russell_llm_pipeline.sh
#   ./run_russell_llm_pipeline.sh 2000-01-01 2025-12-31 10
#
# Optional env vars:
#   DATA_START_DATE=1997-01-01
#   COMPOSITE_JOBS=2
#   RUN_SIGNAL_DUMP=1
#   RUN_COMPOSITES=1
#   RUN_AB=1
#   REVIEW_MODE=static
#   SKIP_PLOTS=1
#   LLM_MODEL=deepseek-reasoner
# =============================================================================

set -euo pipefail

BACKTEST_START_DATE="${1:-2000-01-01}"
END_DATE="${2:-2025-12-31}"
FORWARD_DAYS="${3:-10}"

DATA_START_DATE="${DATA_START_DATE:-1997-01-01}"
RUN_SIGNAL_DUMP="${RUN_SIGNAL_DUMP:-1}"
RUN_COMPOSITES="${RUN_COMPOSITES:-1}"
RUN_AB="${RUN_AB:-1}"
REVIEW_MODE="${REVIEW_MODE:-static}"

if [ -z "${SKIP_PLOTS+x}" ] && [ -n "${KIP_PLOTS+x}" ]; then
  SKIP_PLOTS="${KIP_PLOTS}"
  echo "⚠️  Using KIP_PLOTS as an alias for SKIP_PLOTS"
else
  SKIP_PLOTS="${SKIP_PLOTS:-1}"
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [ -x "./.venv/bin/python" ]; then
  PYTHON_CMD="./.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

if [ -n "${COMPOSITE_JOBS:-}" ]; then
  JOBS="${COMPOSITE_JOBS}"
else
  CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
  if [ "${CPU_COUNT}" -lt 2 ]; then
    JOBS=1
  else
    JOBS=2
  fi
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-backtest_results/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

TEMP_SIGNAL_DIR=""
TEMP_COMPOSITE_DIR=""

run_cmd() {
  echo "+ $*"
  "$@"
}

make_temp_dir() {
  local prefix="$1"
  mkdir -p backtest_results
  mktemp -d "backtest_results/${prefix}.tmp.XXXXXX"
}

count_matching_csvs() {
  local search_dir="$1"
  local pattern="$2"
  if [ ! -d "${search_dir}" ]; then
    echo "0"
    return
  fi
  find "${search_dir}" -type f -name "${pattern}" | wc -l | tr -d ' '
}

promote_dir() {
  local staged_dir="$1"
  local final_dir="$2"
  local backup_dir="${final_dir}.__previous__"

  if [ -e "${backup_dir}" ]; then
    rm -rf "${backup_dir}"
  fi

  if [ -e "${final_dir}" ]; then
    mv "${final_dir}" "${backup_dir}"
  fi

  if mv "${staged_dir}" "${final_dir}"; then
    if [ -e "${backup_dir}" ]; then
      rm -rf "${backup_dir}"
    fi
    return
  fi

  if [ -e "${backup_dir}" ] && [ ! -e "${final_dir}" ]; then
    mv "${backup_dir}" "${final_dir}"
  fi

  echo "❌ Failed to promote ${staged_dir} to ${final_dir}"
  exit 1
}

cleanup_temp_dirs() {
  if [ -n "${TEMP_SIGNAL_DIR}" ] && [ -d "${TEMP_SIGNAL_DIR}" ]; then
    rm -rf "${TEMP_SIGNAL_DIR}"
  fi
  if [ -n "${TEMP_COMPOSITE_DIR}" ] && [ -d "${TEMP_COMPOSITE_DIR}" ]; then
    rm -rf "${TEMP_COMPOSITE_DIR}"
  fi
}

trap cleanup_temp_dirs EXIT

echo ""
echo "=================================================================="
echo "  RUSSELL LLM PIPELINE RUNNER"
echo "=================================================================="
echo "  Data Start:      ${DATA_START_DATE}"
echo "  Backtest Start:  ${BACKTEST_START_DATE}"
echo "  End Date:        ${END_DATE}"
echo "  Forward Horizon: ${FORWARD_DAYS} days"
echo "  Python:          ${PYTHON_CMD}"
echo "  Composite Jobs:  ${JOBS}"
echo "  Review Mode:     ${REVIEW_MODE}"
echo "  Skip Plots:      ${SKIP_PLOTS}"
echo "=================================================================="
echo ""

if [ "${RUN_SIGNAL_DUMP}" = "1" ]; then
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  STAGE 1: Russell Signal Dump                                  ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""

  TEMP_SIGNAL_DIR="$(make_temp_dir russell_signal_dump)"

  run_cmd "${PYTHON_CMD}" russell_signal_dump.py \
    --start "${DATA_START_DATE}" \
    --end "${END_DATE}" \
    --output-dir "${TEMP_SIGNAL_DIR}"

  SIGNAL_COUNT="$(count_matching_csvs "${TEMP_SIGNAL_DIR}" 'russell_signal_dump_*.csv')"
  if [ "${SIGNAL_COUNT}" -eq 0 ]; then
    echo "❌ Stage 1 completed without writing any Russell signal dump CSVs"
    exit 1
  fi

  promote_dir "${TEMP_SIGNAL_DIR}" "backtest_results/russell_signal_dump"
  TEMP_SIGNAL_DIR=""

  echo ""
  echo "✓ Stage 1 complete: Russell signal dumps built (${SIGNAL_COUNT} files)"
  echo ""
else
  echo "STAGE 1: Skipped (RUN_SIGNAL_DUMP=${RUN_SIGNAL_DUMP})"
  echo ""
fi

if [ "${RUN_COMPOSITES}" = "1" ]; then
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  STAGE 2: Deterministic Composite Scores                       ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""

  if [ ! -d "backtest_results/russell_signal_dump" ]; then
    echo "❌ Missing directory: backtest_results/russell_signal_dump"
    echo "   Run Stage 1 first or set RUN_SIGNAL_DUMP=1"
    exit 1
  fi

  SIGNAL_COUNT="$(count_matching_csvs "backtest_results/russell_signal_dump" 'russell_signal_dump_*.csv')"
  if [ "${SIGNAL_COUNT}" -eq 0 ]; then
    echo "❌ No Russell signal dump CSVs found in backtest_results/russell_signal_dump"
    echo "   Stage 1 may have failed or produced only skipped tickers"
    exit 1
  fi

  TEMP_COMPOSITE_DIR="$(make_temp_dir composite_scores_russell)"
  COMPOSITE_CALC_CMD=(
    "${PYTHON_CMD}" composite_score_calculator.py
    --output-dir "${TEMP_COMPOSITE_DIR}"
  )
  if [ "${SKIP_PLOTS}" = "1" ]; then
    COMPOSITE_CALC_CMD+=(--skip-plots)
  fi

  find backtest_results/russell_signal_dump -name 'russell_signal_dump_*.csv' -print0 | \
    xargs -0 -n 1 -P "${JOBS}" "${COMPOSITE_CALC_CMD[@]}"

  COMPOSITE_COUNT="$(count_matching_csvs "${TEMP_COMPOSITE_DIR}" 'composite_scores_*.csv')"
  if [ "${COMPOSITE_COUNT}" -eq 0 ]; then
    echo "❌ Stage 2 completed without writing any deterministic composite CSVs"
    exit 1
  fi

  promote_dir "${TEMP_COMPOSITE_DIR}" "backtest_results/composite_scores_russell"
  TEMP_COMPOSITE_DIR=""

  echo ""
  echo "✓ Stage 2 complete: Deterministic composites built (${COMPOSITE_COUNT} files)"
  echo ""
else
  echo "STAGE 2: Skipped (RUN_COMPOSITES=${RUN_COMPOSITES})"
  echo ""
fi

if [ "${RUN_AB}" = "1" ]; then
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  STAGE 3: LLM A/B Pipeline                                     ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""

  if [ ! -d "backtest_results/composite_scores_russell" ]; then
    echo "❌ Missing directory: backtest_results/composite_scores_russell"
    echo "   Run Stage 2 first or set RUN_COMPOSITES=1"
    exit 1
  fi

  COMPOSITE_COUNT="$(count_matching_csvs "backtest_results/composite_scores_russell" 'composite_scores_*.csv')"
  if [ "${COMPOSITE_COUNT}" -eq 0 ]; then
    echo "❌ No composite score CSVs found in backtest_results/composite_scores_russell"
    echo "   Run Stage 2 first or set RUN_COMPOSITES=1"
    exit 1
  fi

  export REVIEW_MODE
  export SKIP_PLOTS

  run_cmd ./llm_ab_test_runner.sh \
    "${BACKTEST_START_DATE}" \
    "${END_DATE}" \
    "${FORWARD_DAYS}"

  echo ""
  echo "✓ Stage 3 complete: LLM A/B pipeline finished"
  echo ""
else
  echo "STAGE 3: Skipped (RUN_AB=${RUN_AB})"
  echo ""
fi

echo "=================================================================="
echo "  PIPELINE COMPLETE"
echo "=================================================================="
echo ""
echo "  Russell signal dumps:     backtest_results/russell_signal_dump/"
echo "  Deterministic composites: backtest_results/composite_scores_russell/"
echo "  LLM cache:                backtest_results/llm_portfolio_cache/"
echo "  Backtest outputs:         backtest_results/factor_backtest/"
echo ""
echo "=================================================================="
