import numpy as np
import pandas as pd

from deterministic.signal_contract import FORWARD_HORIZONS, REQUIRED_INDICATORS
from deterministic.signal_generation import MIN_SIGNAL_HISTORY_ROWS, SIGNAL_OUTPUT_COLUMNS, build_signal_frame_from_history


def test_build_signal_frame_from_history_emits_composite_contract():
    n = 620
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100.0, 180.0, n) + 2.0 * np.sin(np.arange(n) / 12.0)
    open_ = close * (1.0 - 0.002)
    high = close * 1.01
    low = close * 0.99
    volume = 1_000_000 + (np.arange(n) % 50) * 10_000

    hist = pd.DataFrame(
        {
            "date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj_Open": open_,
            "Adj_High": high,
            "Adj_Low": low,
            "Adj_Close": close,
        }
    )

    signal_df = build_signal_frame_from_history(hist)

    assert not signal_df.empty
    missing = REQUIRED_INDICATORS - set(signal_df.columns)
    assert not missing, f"missing required indicators: {sorted(missing)}"
    assert "fwd_10d_ret" in signal_df.columns
    assert "fwd_abs_10d_ret" in signal_df.columns


def test_build_signal_frame_from_history_returns_empty_contract_for_short_history():
    n = MIN_SIGNAL_HISTORY_ROWS - 1
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(100.0, 120.0, n)

    hist = pd.DataFrame(
        {
            "date": dates,
            "Open": close * 0.999,
            "High": close * 1.001,
            "Low": close * 0.998,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
            "Adj_Open": close * 0.999,
            "Adj_High": close * 1.001,
            "Adj_Low": close * 0.998,
            "Adj_Close": close,
        }
    )

    signal_df = build_signal_frame_from_history(hist)

    assert signal_df.empty
    assert list(signal_df.columns) == SIGNAL_OUTPUT_COLUMNS
    assert set(REQUIRED_INDICATORS).issubset(signal_df.columns)
    for horizon in FORWARD_HORIZONS:
        assert f"fwd_{horizon}d_ret" in signal_df.columns
