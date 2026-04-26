"""
Shared signal-schema contract for score generation.
"""

from __future__ import annotations


FORWARD_HORIZONS = (1, 5, 10, 20, 30, 60)


REQUIRED_INDICATORS = {
    "vix_percentile_252d",
    "vol_percentile_504d",
    "vix_level",
    "vol_imbalance_20d",
    "m2_m1_ratio",
    "atr_14d_pct",
    "realized_vol_10d",
    "vol_adjusted_trend_ma100",
    "range_pos_55d",
    "pct_from_52w_high",
    "range_pct",
    "autocorr_5d",
    "mom_3m",
    "rsi_5d",
    "vel_10d_over_vol",
    "days_since_52w_high",
    "vol_adjusted_trend_ma100_slope",
    "pct_from_ma100",
}

