composite_scores.py

Purpose: Calculates composite scores for momentum, trend, relative strength (RS), volume, and risk. Combines them into Tier-1 scores and alpha signals for next-day predictions.
Dependencies: technical_indicators_calculator.py (for indicators), regime_classifier.py (for regime-aware weighting).

technical_indicators_calculator.py

Purpose: Computes technical indicators from OHLCV data.
Inputs: Pandas DataFrame with OHLCV columns (Open, High, Low, Close, Volume). Foundation for all scoring. Warmup period (252 days) required for accurate indicators. 

deterministic_scoring.py

Purpose: Combines composite scores into regime-aware Tier-1 master scores. Adjusts weights based on market conditions (e.g., more risk in downtrends).
Key Functions/Outputs:
calculate_tier1_master_score() → Tier-1 score (-1 to +1), quality (average of composites), and regime.
Inputs: Composites dict from composite_scores.py, regime from regime_classifier.py.

regime_classifier.py

Purpose: Classifies market regimes and volatility levels to inform signal weighting and risk.
Inputs: Indicators from technical_indicators_calculator.py.
Note: Uses MA20/75/200 and volatility thresholds. "Unknown" regimes indicate missing data. 

hit_rate_validator.py

Purpose: Validates signal quality by simulating trades and calculating hit rates, returns, and diagnostics across stocks.
uns on historical data. Compares Tier-1 vs. Alpha methods. Target: 52%+ hit rate. Use for pre-live validation.

backtest_single_stock.py

Purpose: Backtests a single stock using live-like signal generation and position sizing.
Simulates real trading with slippage/commission. Use for stock-specific tuning.

feature_diagnostics.py

Purpose: Analyzes feature distributions, saturation, correlations, and non-linear relationships to identify issues (e.g., weak signals).
Notes: Flags problems like "alpha score always 0". Run after signal changes.

signal_feature_pipeline.py

Purpose: Orchestrates multi-stock signal generation, ranking, and universe-wide analysis.
generate_universe_signals() → Ranked signals for all stocks.
Outputs: JSON, regime classifications
