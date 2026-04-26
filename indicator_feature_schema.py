from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd


# Canonical indicator names used throughout the LLM pipeline.
# Composite score CSVs now typically persist percentile-normalized versions
# of these features as `<feature>_percentile`, so callers should resolve
# through the helper below rather than reading columns directly.
CANONICAL_INDICATOR_FEATURES = [
    "vix_percentile_252d",
    "vol_percentile_504d",
    "vix_level",
    "adx_14d",
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
]


def candidate_indicator_columns(feature_name: str) -> tuple[str, ...]:
    """Return likely source columns for a canonical feature.

    We prefer percentile-normalized exports when available because Russell
    composite CSVs currently persist those columns and omit many raw values.
    """
    return (f"{feature_name}_percentile", feature_name)


def extract_canonical_indicator_features(
    row: Any,
    feature_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Read indicator values from a row into canonical feature names.

    The returned dict always uses the canonical names above, regardless of
    whether the source row provided raw values or percentile-normalized ones.
    """
    features: Dict[str, float] = {}
    for feature_name in feature_names or CANONICAL_INDICATOR_FEATURES:
        for source_col in candidate_indicator_columns(feature_name):
            value = _safe_float(row.get(source_col))
            if value is None:
                continue
            features[feature_name] = value
            break
    return features


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
