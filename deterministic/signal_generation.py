"""
Shared signal-frame generation from OHLCV history.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deterministic.hit_rate_validator import CANONICAL_EXECUTION_MODEL, calculate_forward_return_label
from deterministic.signal_contract import FORWARD_HORIZONS, REQUIRED_INDICATORS
from deterministic.technical_indicators_calculator import TechnicalIndicatorCalculator


_COLUMN_RENAMES = {
    "date": "date",
    "Date": "date",
    "open": "Open",
    "Open": "Open",
    "high": "High",
    "High": "High",
    "low": "Low",
    "Low": "Low",
    "close": "Close",
    "Close": "Close",
    "volume": "Volume",
    "Volume": "Volume",
    "adj_open": "Adj_Open",
    "Adj_Open": "Adj_Open",
    "adj_high": "Adj_High",
    "Adj_High": "Adj_High",
    "adj_low": "Adj_Low",
    "Adj_Low": "Adj_Low",
    "adj_close": "Adj_Close",
    "Adj_Close": "Adj_Close",
    "adj_volume": "Adj_Volume",
    "Adj_Volume": "Adj_Volume",
}

SIGNAL_WARMUP_ROWS = 504
MIN_SIGNAL_HISTORY_ROWS = SIGNAL_WARMUP_ROWS + 1
REQUIRED_INDICATOR_COLUMNS = tuple(sorted(REQUIRED_INDICATORS))
SIGNAL_OUTPUT_COLUMNS = (
    ["date", "adj_open", "adj_close"]
    + [f"fwd_{h}d_ret" for h in FORWARD_HORIZONS]
    + [f"fwd_abs_{h}d_ret" for h in FORWARD_HORIZONS if h != 1]
    + list(REQUIRED_INDICATOR_COLUMNS)
)


def normalize_ohlcv_history(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    rename_map = {col: _COLUMN_RENAMES[col] for col in df.columns if col in _COLUMN_RENAMES}
    df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    else:
        raise ValueError("History must contain a date column or DatetimeIndex")

    if "Open" not in df.columns and "Adj_Open" in df.columns:
        df["Open"] = df["Adj_Open"]
    if "High" not in df.columns and "Adj_High" in df.columns:
        df["High"] = df["Adj_High"]
    if "Low" not in df.columns and "Adj_Low" in df.columns:
        df["Low"] = df["Adj_Low"]
    if "Close" not in df.columns and "Adj_Close" in df.columns:
        df["Close"] = df["Adj_Close"]
    if "Volume" not in df.columns and "Adj_Volume" in df.columns:
        df["Volume"] = df["Adj_Volume"]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    for col in ["Open", "High", "Low", "Close", "Volume", "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


def _coerce_indicator_value(ind_value: Any) -> float:
    if ind_value is None or isinstance(ind_value, (bool, np.bool_)):
        return np.nan
    try:
        fval = float(ind_value)
        return fval if np.isfinite(fval) else np.nan
    except (TypeError, ValueError):
        return np.nan


def build_signal_frame_from_history(
    hist: pd.DataFrame,
    execution_model: str = CANONICAL_EXECUTION_MODEL,
) -> pd.DataFrame:
    normalized_hist = normalize_ohlcv_history(hist)
    records: list[dict[str, Any]] = []

    for idx in range(SIGNAL_WARMUP_ROWS, len(normalized_hist)):
        date = normalized_hist.index[idx]
        hist_subset = normalized_hist.iloc[: idx + 1]
        calc = TechnicalIndicatorCalculator(hist_subset)

        all_indicators = {}
        all_indicators.update(calc.calculate_momentum() or {})
        all_indicators.update(calc.calculate_trend_indicators() or {})
        all_indicators.update(calc.calculate_mean_reversion_indicators() or {})
        all_indicators.update(calc.calculate_trend_quality_indicators() or {})
        all_indicators.update(calc.calculate_range_structure_indicators() or {})
        all_indicators.update(calc.calculate_volatility_indicators() or {})
        all_indicators.update(calc.calculate_volume_liquidity_indicators() or {})
        all_indicators.update(calc.calculate_volume_participation_indicators() or {})
        all_indicators.update(calc.calculate_price_events() or {})
        all_indicators.update(calc.calculate_vix_indicators(date) or {})
        all_indicators.update(calc.calculate_term_structure_indicators(date) or {})

        fwd_returns = {
            f"fwd_{h}d_ret": calculate_forward_return_label(
                normalized_hist, date, h, execution_model=execution_model
            )
            for h in FORWARD_HORIZONS
        }

        record = {
            "date": date,
            "adj_open": float(hist_subset["Adj_Open"].iloc[-1]) if "Adj_Open" in hist_subset.columns else np.nan,
            "adj_close": float(hist_subset["Adj_Close"].iloc[-1]) if "Adj_Close" in hist_subset.columns else float(hist_subset["Close"].iloc[-1]),
            **fwd_returns,
        }
        for h in FORWARD_HORIZONS:
            if h == 1:
                continue
            ret_val = record[f"fwd_{h}d_ret"]
            record[f"fwd_abs_{h}d_ret"] = abs(ret_val) if pd.notna(ret_val) else np.nan

        for ind_name in REQUIRED_INDICATOR_COLUMNS:
            record[ind_name] = _coerce_indicator_value(all_indicators.get(ind_name))

        records.append(record)

    return pd.DataFrame.from_records(records, columns=SIGNAL_OUTPUT_COLUMNS)
