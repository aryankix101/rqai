"""
Shared trade-timing helpers for forward-return labeling.

Canonical convention:
- Signal is observed at the close of signal_date
- Enter at the next trading day's open
- Hold for N full trading days
- Exit at the open after those N holding days
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


CANONICAL_EXECUTION_MODEL = "next_open_to_open_hold"

_OPEN_COL_CANDIDATES = ("Adj_Open", "adj_open", "Open", "open")
_CLOSE_COL_CANDIDATES = ("Adj_Close", "adj_close", "Close", "close")


@dataclass(frozen=True)
class TradeWindow:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float


def _normalize_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Return a date-indexed DataFrame without duplicate dates."""
    if isinstance(hist.index, pd.DatetimeIndex):
        df = hist if hist.index.is_monotonic_increasing else hist.sort_index()
    elif "date" in hist.columns:
        df = hist.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        raise ValueError("History must be indexed by date or contain a 'date' column")

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


def _resolve_price_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _canonicalize_execution_model(execution_model: Optional[str]) -> str:
    if execution_model in (None, "", CANONICAL_EXECUTION_MODEL, "next_open_to_open"):
        return CANONICAL_EXECUTION_MODEL
    if execution_model in {"next_open_to_close", "next_open_to_close_hold"}:
        return execution_model
    raise ValueError(f"Unsupported execution_model: {execution_model}")


def get_trade_window(
    hist: pd.DataFrame,
    signal_date: pd.Timestamp,
    holding_days: int,
    execution_model: str = CANONICAL_EXECUTION_MODEL,
) -> Optional[TradeWindow]:
    """
    Resolve entry/exit dates and prices for a signal date.

    Supported execution models:
    - next_open_to_open_hold: canonical no-lookahead hold
    - next_open_to_close: next-day intraday only
    - next_open_to_close_hold: legacy next-open to target-day close
    """
    if holding_days < 1:
        return None

    model = _canonicalize_execution_model(execution_model)
    df = _normalize_history(hist)
    signal_ts = pd.Timestamp(signal_date)

    if signal_ts not in df.index:
        return None

    signal_idx = df.index.get_loc(signal_ts)
    if not isinstance(signal_idx, (int, np.integer)):
        return None

    open_col = _resolve_price_column(df, _OPEN_COL_CANDIDATES)
    close_col = _resolve_price_column(df, _CLOSE_COL_CANDIDATES)
    if open_col is None or close_col is None:
        return None

    entry_idx = signal_idx + 1
    if model == "next_open_to_close":
        exit_idx = entry_idx
        exit_col = close_col
    elif model == "next_open_to_close_hold":
        exit_idx = signal_idx + holding_days
        exit_col = close_col
    else:
        exit_idx = signal_idx + holding_days + 1
        exit_col = open_col

    if entry_idx >= len(df) or exit_idx >= len(df):
        return None

    entry_price = df.iloc[entry_idx][open_col]
    exit_price = df.iloc[exit_idx][exit_col]

    if (
        pd.isna(entry_price)
        or pd.isna(exit_price)
        or float(entry_price) <= 0
        or float(exit_price) <= 0
    ):
        return None

    return TradeWindow(
        signal_date=signal_ts,
        entry_date=pd.Timestamp(df.index[entry_idx]),
        exit_date=pd.Timestamp(df.index[exit_idx]),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
    )


def calculate_forward_return_label(
    hist: pd.DataFrame,
    current_date: pd.Timestamp,
    days_ahead: int,
    execution_model: str = CANONICAL_EXECUTION_MODEL,
) -> float:
    """Calculate forward return in percent using a consistent execution convention."""
    trade_window = get_trade_window(
        hist=hist,
        signal_date=current_date,
        holding_days=days_ahead,
        execution_model=execution_model,
    )
    if trade_window is None:
        return np.nan

    return ((trade_window.exit_price / trade_window.entry_price) - 1.0) * 100.0
