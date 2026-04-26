"""Shared portfolio-level trading guardrails."""

import math

import pandas as pd


MIN_TRADE_PRICE = 5.0


def has_min_trade_price(price) -> bool:
    """Return True when the candidate price clears the trading floor."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > MIN_TRADE_PRICE


def _is_truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def row_has_clean_symbol_metadata(row) -> bool:
    """Reject ambiguous or recycled-symbol Russell rows when metadata is present.

    Russell composite CSVs carry provenance fields such as `is_historical_suffix`
    and `collision_count`. When a base ticker is reused or multiple source columns
    map to the same base ticker on a date, those rows can create exactly the kind
    of stale-symbol / corporate-action artifacts we saw with MAXX-style names.
    """
    if row is None:
        return False

    historical_flag = row.get("is_historical_suffix")
    if historical_flag is not None and pd.notna(historical_flag) and _is_truthy(historical_flag):
        return False

    collision_count = row.get("collision_count")
    if collision_count is not None and pd.notna(collision_count):
        try:
            if int(float(collision_count)) > 1:
                return False
        except (TypeError, ValueError):
            return False

    return True


def filter_symbol_hygiene_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop Russell rows that fail the shared symbol-hygiene checks."""
    if df is None:
        return None
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index, dtype=bool)
    if "is_historical_suffix" in df.columns:
        hist = df["is_historical_suffix"].map(lambda value: _is_truthy(value) if pd.notna(value) else False)
        mask &= ~hist
    if "collision_count" in df.columns:
        collisions = pd.to_numeric(df["collision_count"], errors="coerce")
        mask &= collisions.fillna(1).le(1)

    return df.loc[mask].copy()
