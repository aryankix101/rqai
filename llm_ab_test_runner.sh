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
REBALANCE_DAYS="${FORWARD_DAYS}"

# Paths
CACHE_DIR="backtest_results/llm_portfolio_cache"
LLM_SCORES_DIR="backtest_results/composite_scores_llm"
CACHE_FILE="${CACHE_DIR}/llm_portfolio_${START_DATE}_${END_DATE}_${REBALANCE_DAYS}d.json"

echo ""
echo "=================================================================="
echo "  LLM A/B TEST RUNNER"
echo "=================================================================="
echo "  Test Window:     ${START_DATE} to ${END_DATE}"
echo "  Forward Horizon: ${FORWARD_DAYS} days"
echo "  Rebalance:       every ${REBALANCE_DAYS} trading days"
echo "=================================================================="
echo ""

# ─────────────────────────────────────────────────────────────────────
# Stage 1: Generate LLM Portfolio Decisions
# ─────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  STAGE 1: LLM Portfolio Construction                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

python3 llm_portfolio_constructor.py \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --rebalance-days "${REBALANCE_DAYS}" \
    --resume \
    --temperature 0.6 \
    --top-n 60 \
    --rate-limit 2.0

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

python3 llm_composite_writer.py \
    --cache "${CACHE_FILE}" \
    --output-dir "${LLM_SCORES_DIR}"

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

python3 factor_backtest_engine.py \
    "${FORWARD_DAYS}" \
    "${START_DATE}" \
    "${END_DATE}"

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

python3 factor_backtest_engine.py \
    "${FORWARD_DAYS}" \
    "${START_DATE}" \
    "${END_DATE}" \
    --scores-dir "${LLM_SCORES_DIR}"

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
DET_RESULTS="backtest_results/factor_backtest/${FORWARD_DAYS}d_horizon"
LLM_RESULTS="backtest_results/factor_backtest/composite_scores_llm_${FORWARD_DAYS}d_horizon"

python3 llm_ab_comparison.py \
    --det-dir "${DET_RESULTS}" \
    --llm-dir "${LLM_RESULTS}" \
    --output-dir "backtest_results/factor_backtest/ab_comparison_${FORWARD_DAYS}d" \
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
echo "  Comparison:            backtest_results/factor_backtest/ab_comparison_${FORWARD_DAYS}d/"
echo "  LLM cache:             ${CACHE_FILE}"
echo ""
echo "=================================================================="
