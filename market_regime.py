"""Provide selector-neutral, point-in-time SPY market-regime features.

The helpers classify each signal date using only market information available
by that date so strategy prompts cannot acquire look-ahead information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_source_config import SPY_SIGNAL_DATA_PATH


SPY_DATA_PATH = SPY_SIGNAL_DATA_PATH


def identify_regime(spy_df: pd.DataFrame, date: pd.Timestamp) -> str:
    """Classify the market regime using only SPY/VIX data available by ``date``."""
    frame = spy_df.sort_values("date").reset_index(drop=True).copy()
    frame["spy_200d_ma"] = frame["adj_close"].rolling(200, min_periods=200).mean()

    row = frame[frame["date"] == date]
    if row.empty:
        prior = frame[frame["date"] <= date]
        if prior.empty:
            return "unknown"
        row = prior.iloc[[-1]]

    observation = row.iloc[0]
    close = observation.get("adj_close", np.nan)
    ma200 = observation.get("spy_200d_ma", np.nan)
    vix_pct = observation.get("vix_percentile_252d", np.nan)
    if pd.isna(close) or pd.isna(ma200):
        return "unknown"

    trend = "uptrend" if close > ma200 else "downtrend"
    if pd.isna(vix_pct):
        return "normal"

    prior_vix = frame.loc[frame["date"] <= date, "vix_percentile_252d"].dropna()
    if len(prior_vix) < 50:
        return "normal"

    thresh_high = prior_vix.quantile(0.80)
    thresh_low = prior_vix.quantile(0.30)
    if vix_pct > thresh_high:
        volatility = "high_vol"
    elif vix_pct < thresh_low:
        volatility = "low_vol"
    else:
        volatility = "mid_vol"

    if volatility == "high_vol" and trend == "downtrend":
        return "crisis"
    if volatility == "high_vol" and trend == "uptrend":
        return "squeeze"
    if volatility == "low_vol" and trend == "uptrend":
        return "grind_up"
    if volatility == "low_vol" and trend == "downtrend":
        return "slow_bleed"
    return "normal"
