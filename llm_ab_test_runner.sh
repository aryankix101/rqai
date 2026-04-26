#!/bin/bash
# =============================================================================
# LLM A/B Test Runner
# 
# Runs a full A/B comparison: Deterministic vs LLM portfolio construction
#
# Usage:
#   chmod +x llm_ab_test_runner.sh
#   ./llm_ab_test_runner.sh                          # defaults: 2019, 10d horizon
#   ./llm_ab_test_runner.sh 2017-01-01 2017-12-31    # custom date range
#   ./llm_ab_test_runner.sh 2019-01-01 2019-12-31 10 # custom date range + horizon
# =============================================================================

set -euo pipefail

START_DATE="${1:-2019-01-01}"
END_DATE="${2:-2019-12-31}"
FORWARD_DAYS="${3:-10}"
LLM_DECISION_DAYS="${LLM_DECISION_DAYS:-10}"
LLM_MODEL="${LLM_MODEL:-deepseek-reasoner}"
REVIEW_MODE="${REVIEW_MODE:-adaptive}"
HYSTERESIS_MODE="${HYSTERESIS_MODE:-combined}"
REVIEW_NORMAL_DAYS="${REVIEW_NORMAL_DAYS:-20}"
REVIEW_GRIND_UP_DAYS="${REVIEW_GRIND_UP_DAYS:-60}"
REVIEW_SLOW_BLEED_DAYS="${REVIEW_SLOW_BLEED_DAYS:-20}"
REVIEW_SQUEEZE_DAYS="${REVIEW_SQUEEZE_DAYS:-5}"
REVIEW_CRISIS_DAYS="${REVIEW_CRISIS_DAYS:-5}"
REVIEW_UNKNOWN_DAYS="${REVIEW_UNKNOWN_DAYS:-20}"
DRY_RUN="${DRY_RUN:-0}"
LLM_TURNOVER_AWARE="${LLM_TURNOVER_AWARE:-0}"
ENABLE_LOSS_MEMORY="${ENABLE_LOSS_MEMORY:-0}"
LOSS_MEMORY_TOP_K="${LOSS_MEMORY_TOP_K:-3}"
LOSS_MEMORY_MAX_TARGETS_PER_SIDE="${LOSS_MEMORY_MAX_TARGETS_PER_SIDE:-4}"
DATASET_VERSION_TAG="${DATASET_VERSION_TAG:-russell_ohlcv_v1}"

if [ -z "${SKIP_PLOTS+x}" ] && [ -n "${KIP_PLOTS+x}" ]; then
  SKIP_PLOTS="${KIP_PLOTS}"
  echo "⚠️  Using KIP_PLOTS as an alias for SKIP_PLOTS"
else
  SKIP_PLOTS="${SKIP_PLOTS:-0}"
fi

RUN_ID="${REVIEW_MODE}_${HYSTERESIS_MODE}_n${REVIEW_NORMAL_DAYS}_g${REVIEW_GRIND_UP_DAYS}_sb${REVIEW_SLOW_BLEED_DAYS}_sq${REVIEW_SQUEEZE_DAYS}_cr${REVIEW_CRISIS_DAYS}_u${REVIEW_UNKNOWN_DAYS}_${FORWARD_DAYS}d_${START_DATE}_${END_DATE}"
DET_VARIANT="det_${RUN_ID}"
if [ "${LLM_TURNOVER_AWARE}" = "1" ]; then
  LLM_PROMPT_TAG="turnover_aware"
else
  LLM_PROMPT_TAG="signal_only"
fi
LOSS_MEMORY_TAG=""
if [ "${ENABLE_LOSS_MEMORY}" = "1" ]; then
  LOSS_MEMORY_TAG="_lossmem_k${LOSS_MEMORY_TOP_K}_t${LOSS_MEMORY_MAX_TARGETS_PER_SIDE}"
fi
LLM_VARIANT="llm_${RUN_ID}_${LLM_PROMPT_TAG}${LOSS_MEMORY_TAG}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [ -x "./.venv/bin/python" ]; then
  PYTHON_CMD="./.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

# Paths
CACHE_DIR="backtest_results/llm_portfolio_cache"
LLM_SCORES_DIR="backtest_results/composite_scores_llm_${RUN_ID}_${LLM_PROMPT_TAG}${LOSS_MEMORY_TAG}"
CACHE_FILE="${CACHE_DIR}/llm_portfolio_${START_DATE}_${END_DATE}_${LLM_DECISION_DAYS}d_equal_weight_${LLM_PROMPT_TAG}_${DATASET_VERSION_TAG}${LOSS_MEMORY_TAG}.json"

run_cmd() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

echo ""
echo "=================================================================="
echo "  LLM A/B TEST RUNNER"
echo "=================================================================="
echo "  Test Window:     ${START_DATE} to ${END_DATE}"
echo "  Forward Horizon: ${FORWARD_DAYS} days"
echo "  LLM Decisions:   every ${LLM_DECISION_DAYS} trading days"
echo "  LLM Model:       ${LLM_MODEL}"
echo "  LLM Prompt:      ${LLM_PROMPT_TAG}"
if [ "${ENABLE_LOSS_MEMORY}" = "1" ]; then
  echo "  Loss Memory:     on (top_k=${LOSS_MEMORY_TOP_K}, targets/side=${LOSS_MEMORY_MAX_TARGETS_PER_SIDE})"
else
  echo "  Loss Memory:     off"
fi
echo "  Review Policy:   ${REVIEW_MODE} / ${HYSTERESIS_MODE}"
echo "  Regime Days:     N=${REVIEW_NORMAL_DAYS} G=${REVIEW_GRIND_UP_DAYS} SB=${REVIEW_SLOW_BLEED_DAYS} SQ=${REVIEW_SQUEEZE_DAYS} CR=${REVIEW_CRISIS_DAYS} U=${REVIEW_UNKNOWN_DAYS}"
echo "  Python:          ${PYTHON_CMD}"
if [ "${DRY_RUN}" = "1" ]; then
  echo "  Mode:            DRY RUN"
fi
echo "=================================================================="
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 1: Generate LLM Portfolio Decisions
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 1: LLM Portfolio Construction                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

LLM_CONSTRUCTOR_CMD=(
  "${PYTHON_CMD}" llm_portfolio_constructor.py
  --start "${START_DATE}"
  --end "${END_DATE}"
  --rebalance-days "${LLM_DECISION_DAYS}"
  --model "${LLM_MODEL}"
  --resume
  --equal-weight
  --temperature 0.6
  --top-n 60
  --rate-limit 2.0
)
if [ "${LLM_TURNOVER_AWARE}" = "1" ]; then
  LLM_CONSTRUCTOR_CMD+=(--turnover-aware)
fi
if [ "${ENABLE_LOSS_MEMORY}" = "1" ]; then
  LLM_CONSTRUCTOR_CMD+=(
    --enable-loss-memory
    --loss-memory-top-k "${LOSS_MEMORY_TOP_K}"
    --loss-memory-max-targets-per-side "${LOSS_MEMORY_MAX_TARGETS_PER_SIDE}"
  )
fi
run_cmd "${LLM_CONSTRUCTOR_CMD[@]}"

echo ""
echo "✓ Stage 1 complete: LLM portfolio decisions cached"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 2: Convert LLM Decisions to Composite Score CSVs
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 2: Write LLM Composite CSVs                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

run_cmd "${PYTHON_CMD}" llm_composite_writer.py \
    --cache "${CACHE_FILE}" \
    --output-dir "${LLM_SCORES_DIR}"

DET_SCORE_COUNT="$(find backtest_results/composite_scores_russell -type f -name 'composite_scores_*.csv' | wc -l | tr -d ' ')"
LLM_SCORE_COUNT="$(find "${LLM_SCORES_DIR}" -type f -name 'composite_scores_*.csv' | wc -l | tr -d ' ')"
if [ "${LLM_SCORE_COUNT}" -eq 0 ]; then
  echo "❌ LLM composite writer produced zero score files"
  exit 1
fi
if [ "${DET_SCORE_COUNT}" -gt 0 ] && [ "${LLM_SCORE_COUNT}" -lt $(( DET_SCORE_COUNT / 2 )) ]; then
  echo "❌ LLM composite directory looks incomplete: ${LLM_SCORE_COUNT} files vs ${DET_SCORE_COUNT} deterministic files"
  exit 1
fi

echo ""
echo "✓ Stage 2 complete: LLM composite CSVs written"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 3: Run Deterministic Backtest (Control)
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 3: Deterministic Backtest (Control Arm)                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

DET_CMD=(
  "${PYTHON_CMD}" factor_backtest_engine.py
  "${FORWARD_DAYS}"
  "${START_DATE}"
  "${END_DATE}"
  --review-mode "${REVIEW_MODE}"
  --hysteresis-mode "${HYSTERESIS_MODE}"
  --review-normal-days "${REVIEW_NORMAL_DAYS}"
  --review-grind-up-days "${REVIEW_GRIND_UP_DAYS}"
  --review-slow-bleed-days "${REVIEW_SLOW_BLEED_DAYS}"
  --review-squeeze-days "${REVIEW_SQUEEZE_DAYS}"
  --review-crisis-days "${REVIEW_CRISIS_DAYS}"
  --review-unknown-days "${REVIEW_UNKNOWN_DAYS}"
  --variant-label "${DET_VARIANT}"
)
if [ "${SKIP_PLOTS}" = "1" ]; then
  DET_CMD+=(--skip-plots)
fi
run_cmd "${DET_CMD[@]}"

echo ""
echo "✓ Stage 3 complete: Deterministic backtest done"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 4: Run LLM Backtest (Treatment)
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 4: LLM Backtest (Treatment Arm)                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

LLM_CMD=(
  "${PYTHON_CMD}" factor_backtest_engine.py
  "${FORWARD_DAYS}"
  "${START_DATE}"
  "${END_DATE}"
  --scores-dir "${LLM_SCORES_DIR}"
  --review-mode "${REVIEW_MODE}"
  --hysteresis-mode "${HYSTERESIS_MODE}"
  --review-normal-days "${REVIEW_NORMAL_DAYS}"
  --review-grind-up-days "${REVIEW_GRIND_UP_DAYS}"
  --review-slow-bleed-days "${REVIEW_SLOW_BLEED_DAYS}"
  --review-squeeze-days "${REVIEW_SQUEEZE_DAYS}"
  --review-crisis-days "${REVIEW_CRISIS_DAYS}"
  --review-unknown-days "${REVIEW_UNKNOWN_DAYS}"
  --variant-label "${LLM_VARIANT}"
)
if [ "${SKIP_PLOTS}" = "1" ]; then
  LLM_CMD+=(--skip-plots)
fi
run_cmd "${LLM_CMD[@]}"

echo ""
echo "✓ Stage 4 complete: LLM backtest done"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 5: Generate A/B Comparison Dashboard
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 5: A/B Comparison Dashboard                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Determine output paths based on naming convention
DET_RESULTS="backtest_results/factor_backtest/${DET_VARIANT}/${FORWARD_DAYS}d_horizon"
LLM_RESULTS="backtest_results/factor_backtest/${LLM_VARIANT}/${FORWARD_DAYS}d_horizon"
AB_OUTPUT="backtest_results/factor_backtest/ab_comparison_${RUN_ID}_${LLM_PROMPT_TAG}${LOSS_MEMORY_TAG}"

run_cmd "${PYTHON_CMD}" llm_ab_comparison.py \
    --det-dir "${DET_RESULTS}" \
    --llm-dir "${LLM_RESULTS}" \
    --output-dir "${AB_OUTPUT}" \
    --start "${START_DATE}" \
    --end "${END_DATE}"

echo ""
echo "✓ Stage 5 complete: Comparison dashboard generated"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "  A/B TEST COMPLETE"
echo "=================================================================="
echo ""
echo "  Deterministic results: ${DET_RESULTS}/"
echo "  LLM results:           ${LLM_RESULTS}/"
echo "  Comparison:            ${AB_OUTPUT}/"
echo "  LLM cache:             ${CACHE_FILE}"
echo ""
echo "=================================================================="
