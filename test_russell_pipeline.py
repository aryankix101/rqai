import numpy as np
import pandas as pd

from factor_backtest_engine import (
    CandidatePoolConfig,
    _candidate_snapshot,
    _execution_price_column,
    _is_close_only_dataset,
    _prepare_ticker_data,
)
from hit_rate_validator import CLOSE_TO_CLOSE_EXECUTION_MODEL, calculate_forward_return_label
from portfolio_constraints import filter_symbol_hygiene_frame
from russell_signal_dump import write_russell_signal_dumps
from russell_data_loader import (
    _collapse_wide_ohlcv_histories,
    collapse_to_base_ticker_histories,
    parse_source_column,
)
from signal_generation import MIN_SIGNAL_HISTORY_ROWS


def test_parse_source_column_preserves_suffix_metadata():
    current = parse_source_column("AAPL")
    historical = parse_source_column("AAPL-202212")

    assert current.ticker == "AAPL"
    assert current.asset_id == "AAPL"
    assert current.is_historical_suffix is False
    assert current.suffix_yyyymm is None

    assert historical.ticker == "AAPL"
    assert historical.asset_id == "AAPL-202212"
    assert historical.is_historical_suffix is True
    assert historical.suffix_yyyymm == 202212


def test_collapse_to_base_ticker_histories_prefers_unsuffixed_then_latest_suffix():
    df_wide = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "A": [10.0, np.nan, np.nan],
            "A-199901": [9.0, 9.5, 9.8],
            "A-200012": [11.0, 11.5, 11.8],
        }
    )

    histories = collapse_to_base_ticker_histories(df_wide)
    a_hist = histories["A"]

    assert list(a_hist["adj_close"]) == [10.0, 11.5, 11.8]
    assert list(a_hist["asset_id"]) == ["A", "A-200012", "A-200012"]
    assert list(a_hist["collision_count"]) == [3, 2, 2]


def test_close_to_close_forward_return_label_uses_close_prices():
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "adj_close": [100.0, 110.0, 121.0],
        }
    )

    ret_1d = calculate_forward_return_label(
        hist=hist,
        current_date=pd.Timestamp("2024-01-02"),
        days_ahead=1,
        execution_model=CLOSE_TO_CLOSE_EXECUTION_MODEL,
    )
    ret_2d = calculate_forward_return_label(
        hist=hist,
        current_date=pd.Timestamp("2024-01-02"),
        days_ahead=2,
        execution_model=CLOSE_TO_CLOSE_EXECUTION_MODEL,
    )

    assert np.isclose(ret_1d, 10.0)
    assert np.isclose(ret_2d, 21.0)


def test_filter_symbol_hygiene_frame_drops_historical_and_collision_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["SAFE", "HIST", "COLL"],
            "adj_close": [10.0, 11.0, 12.0],
            "direction_score": [1.0, 2.0, 3.0],
            "is_historical_suffix": [False, True, False],
            "collision_count": [1, 1, 2],
        }
    )

    cleaned = filter_symbol_hygiene_frame(df)

    assert list(cleaned["ticker"]) == ["SAFE"]


def test_candidate_snapshot_excludes_dirty_symbol_metadata_rows():
    dirty = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "adj_close": [10.0, 999.0, 11.0],
            "direction_score": [5.0, 90.0, 6.0],
            "is_historical_suffix": [False, True, False],
            "collision_count": [1, 1, 1],
        }
    )

    prepared = {
        "MAXX": _prepare_ticker_data(
            dirty,
            "direction_score",
            preferred_price_column="adj_close",
        )
    }

    snapshot = _candidate_snapshot(
        "direction_score",
        prepared,
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03"),
        require_exact_dates=True,
    )

    assert snapshot["date_scores"] == {}
    assert snapshot["long_candidates"] == []
    assert snapshot["short_candidates"] == []


def test_collapse_wide_ohlcv_histories_supports_uploaded_parquet_shape():
    df_wide = pd.DataFrame(
        {
            "A_Open": [10.0, np.nan, 30.0],
            "A_High": [11.0, np.nan, 31.0],
            "A_Low": [9.0, np.nan, 29.0],
            "A_Close": [10.5, np.nan, 30.5],
            "A_Volume": [100.0, np.nan, 300.0],
            "A_Unadjusted Close": [105.0, np.nan, 305.0],
            "A-202312_Open": [9.0, 19.0, 29.0],
            "A-202312_High": [10.0, 20.0, 30.0],
            "A-202312_Low": [8.0, 18.0, 28.0],
            "A-202312_Close": [9.5, 19.5, 29.5],
            "A-202312_Volume": [90.0, 190.0, 290.0],
            "A-202312_Unadjusted Close": [95.0, 195.0, 295.0],
        },
        index=pd.Index([45292.0, 45293.0, 45294.0], name="Date"),
    )

    histories = _collapse_wide_ohlcv_histories(df_wide)
    a_hist = histories["A"]

    assert list(a_hist["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert list(a_hist["asset_id"]) == ["A", "A-202312", "A"]
    assert list(a_hist["collision_count"]) == [2, 1, 2]
    assert list(a_hist["close"]) == [105.0, 195.0, 305.0]
    assert list(a_hist["adj_close"]) == [10.5, 19.5, 30.5]
    assert list(a_hist["open"]) == [10.0, 19.0, 30.0]


def test_factor_backtest_close_only_helpers_detect_adj_close_mode():
    close_only = {
        "A": pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "adj_close": [10.0]}),
        "B": pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "adj_close": [20.0]}),
    }
    legacy = {
        "A": pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "adj_open": [9.5], "adj_close": [10.0]}),
    }

    assert _is_close_only_dataset(close_only) is True
    assert _is_close_only_dataset(legacy) is False
    assert _execution_price_column(close_only["A"]) == "adj_close"
    assert _execution_price_column(legacy["A"]) == "adj_open"


def test_candidate_snapshot_can_require_exact_russell_dates():
    stale_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-03", "2000-01-04"]),
            "adj_close": [10.0, 10.5],
            "direction_score": [1.0, 2.0],
        }
    )
    live_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]),
            "adj_close": [20.0, 21.0, 22.0],
            "direction_score": [3.0, 4.0, 5.0],
        }
    )
    prepared = {
        "STALE": _prepare_ticker_data(stale_df, "direction_score", preferred_price_column="adj_close"),
        "LIVE": _prepare_ticker_data(live_df, "direction_score", preferred_price_column="adj_close"),
    }

    permissive = _candidate_snapshot(
        "direction_score",
        prepared,
        pd.Timestamp("2000-01-05"),
        pd.Timestamp("2000-01-05"),
        require_exact_dates=False,
    )
    exact = _candidate_snapshot(
        "direction_score",
        prepared,
        pd.Timestamp("2000-01-05"),
        pd.Timestamp("2000-01-05"),
        require_exact_dates=True,
    )

    assert set(permissive["date_scores"].keys()) == {"STALE", "LIVE"}
    assert set(exact["date_scores"].keys()) == {"LIVE"}


def test_candidate_snapshot_can_mirror_llm_candidate_pools():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05"]),
            "adj_close": [20.0],
            "direction_score": [10.0],
            "magnitude_score": [90.0],
        }
    )
    data = {
        "A": rows.assign(direction_score=10.0, magnitude_score=90.0),
        "B": rows.assign(direction_score=12.0, magnitude_score=70.0),
        "C": rows.assign(direction_score=4.0, magnitude_score=95.0),
        "D": rows.assign(direction_score=-8.0, magnitude_score=80.0),
        "E": rows.assign(direction_score=-20.0, magnitude_score=60.0),
        "F": rows.assign(direction_score=-3.0, magnitude_score=99.0),
    }
    prepared = {
        ticker: _prepare_ticker_data(df, "direction_score", preferred_price_column="adj_close")
        for ticker, df in data.items()
    }

    snapshot = _candidate_snapshot(
        "direction_score",
        prepared,
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-05"),
        require_exact_dates=True,
        candidate_pool_config=CandidatePoolConfig(
            mode="llm_mirror",
            direction_threshold=5.0,
            top_n_magnitude_per_side=2,
        ),
    )

    assert snapshot["long_candidates"] == ["B", "A"]
    assert snapshot["short_candidates"] == ["E", "D"]
    assert set(snapshot["date_scores"].keys()) == {"A", "B", "D", "E"}


def _make_ohlcv_history(ticker: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = np.linspace(100.0, 180.0, periods) + 1.5 * np.sin(np.arange(periods) / 7.0)
    open_ = close * 0.999
    high = close * 1.01
    low = close * 0.99
    volume = np.full(periods, 1_000_000, dtype=float)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "asset_id": ticker,
            "source_column": ticker,
            "is_historical_suffix": False,
            "collision_count": 1,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_open": open_,
            "adj_high": high,
            "adj_low": low,
            "adj_close": close,
        }
    )


def test_write_russell_signal_dumps_skips_short_histories_without_crashing(monkeypatch, tmp_path):
    histories = {
        "GOOD": _make_ohlcv_history("GOOD", MIN_SIGNAL_HISTORY_ROWS + 40),
        "SHORT": _make_ohlcv_history("SHORT", MIN_SIGNAL_HISTORY_ROWS - 10),
    }
    monkeypatch.setattr("russell_signal_dump.load_russell_ohlcv_histories", lambda tickers=None: histories)

    output_dir = tmp_path / "russell_signal_dump"
    write_russell_signal_dumps(
        start_date="2020-01-01",
        end_date="2025-12-31",
        output_dir=output_dir,
    )

    csv_files = sorted(output_dir.glob("russell_signal_dump_*.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].name.startswith("russell_signal_dump_GOOD_")

    manifest = pd.read_csv(output_dir / "_russell_signal_dump_manifest.csv")
    status_map = dict(zip(manifest["ticker"], manifest["status"]))
    assert status_map["GOOD"] == "written"
    assert status_map["SHORT"] == "skipped_short_history"
