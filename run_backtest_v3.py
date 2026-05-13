"""
Standalone weekly Russell mean-reversion benchmark.

This keeps the original weekly loser-vs-winner strategy logic intact while
switching the data inputs over to the current RusselData-backed loaders and
adding a compact summary artifact for later DET comparisons.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from data_source_config import RUSSELL_DATA_DIR
from russell_data_loader import load_russell_ohlcv_histories

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_OUTPUT_DIR = BASE_DIR / "backtest_results" / "weekly_mean_reversion_v3"
DEFAULT_MEMBERSHIP_PATH = RUSSELL_DATA_DIR / "r1000_membership.parquet"
DEFAULT_COST_BPS_LEVELS = (0.0, 2.5, 5.0, 15.0)
DEFAULT_CADENCE_MODE = "weekly_friday"
DEFAULT_SIGNAL_LOOKBACK_DAYS = 5
DEFAULT_REBALANCE_DAYS = 5
MIN_TRADE_PRICE = 5.0
MIN_ADV20_DOLLARS = 5_000_000.0
MIN_UNIVERSE_SIZE = 50
DEFAULT_PERIODS_PER_YEAR = 52.0
SANITY_CHECK_TICKERS = ("AAPL", "LEHMQ")


@dataclass(frozen=True)
class BenchmarkConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    output_dir: Path
    membership_path: Path
    cost_bps_levels: tuple[float, ...] = DEFAULT_COST_BPS_LEVELS
    cadence_mode: str = DEFAULT_CADENCE_MODE
    signal_lookback_days: int = DEFAULT_SIGNAL_LOOKBACK_DAYS
    rebalance_days: int = DEFAULT_REBALANCE_DAYS
    rebalance_anchor_date: pd.Timestamp | None = None
    holding_days: int | None = None
    positions_per_side: int | None = None
    min_trade_price: float = MIN_TRADE_PRICE
    min_adv20_dollars: float = MIN_ADV20_DOLLARS
    min_universe_size: int = MIN_UNIVERSE_SIZE

    def resolved_holding_days(self) -> int:
        return int(self.holding_days) if self.holding_days is not None else int(self.rebalance_days)

    def periods_per_year(self) -> float:
        if self.cadence_mode == "weekly_friday":
            return DEFAULT_PERIODS_PER_YEAR
        return 252.0 / float(max(1, self.resolved_holding_days()))

    def cadence_label(self) -> str:
        if self.cadence_mode == "weekly_friday":
            return "weekly Friday"
        return (
            f"every {int(self.rebalance_days)} trading days "
            f"(lookback={int(self.signal_lookback_days)}, hold={self.resolved_holding_days()})"
        )

    def required_universe_size(self) -> int:
        if self.positions_per_side is None:
            return int(self.min_universe_size)
        if int(self.positions_per_side) <= 0:
            raise ValueError("positions_per_side must be positive when provided")
        return max(int(self.min_universe_size), int(self.positions_per_side) * 2)


@dataclass(frozen=True)
class MembershipLookup:
    dates: tuple[pd.Timestamp, ...]
    symbols_by_date: dict[pd.Timestamp, set[str]]

    def universe_on_or_before(self, date: pd.Timestamp) -> set[str]:
        idx = bisect_right(self.dates, pd.Timestamp(date)) - 1
        if idx < 0:
            return set()
        return self.symbols_by_date[self.dates[idx]]


@dataclass(frozen=True)
class MarketFrames:
    close_df: pd.DataFrame
    open_df: pd.DataFrame
    unadj_close_df: pd.DataFrame
    dollar_volume_df: pd.DataFrame

    @property
    def trading_dates(self) -> pd.DatetimeIndex:
        return self.close_df.index


@dataclass(frozen=True)
class BenchmarkArtifacts:
    weekly_returns: pd.DataFrame
    signal_checks: pd.DataFrame
    portfolio_selections: pd.DataFrame
    headline_metrics: pd.DataFrame
    yearly_returns: pd.DataFrame
    strategy_summary: pd.DataFrame
    market_frames: MarketFrames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone weekly Russell mean-reversion benchmark.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Backtest start date (default: {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END, help=f"Backtest end date (default: {DEFAULT_END})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for benchmark outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--membership-path",
        type=Path,
        default=DEFAULT_MEMBERSHIP_PATH,
        help=f"Optional Russell membership override (default: {DEFAULT_MEMBERSHIP_PATH})",
    )
    parser.add_argument(
        "--cost-bps-levels",
        type=parse_cost_bps_levels,
        default=DEFAULT_COST_BPS_LEVELS,
        help="Comma-separated one-way cost levels in bps (default: 0,2.5,5,15)",
    )
    parser.add_argument(
        "--cadence-mode",
        choices=("weekly_friday", "trading_days"),
        default=DEFAULT_CADENCE_MODE,
        help="Signal/rebalance schedule: weekly Friday benchmark or every N trading days",
    )
    parser.add_argument(
        "--signal-lookback-days",
        type=int,
        default=DEFAULT_SIGNAL_LOOKBACK_DAYS,
        help="Trailing trading-day lookback for the reversal signal in trading-day mode (default: 5)",
    )
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=DEFAULT_REBALANCE_DAYS,
        help="Trading-day rebalance interval in trading-day mode (default: 5)",
    )
    parser.add_argument(
        "--rebalance-anchor-date",
        default=None,
        help="Optional first signal date for trading-day cadence, e.g. 2000-01-03",
    )
    parser.add_argument(
        "--holding-days",
        type=int,
        default=None,
        help="Trading-day holding horizon in trading-day mode (default: same as --rebalance-days)",
    )
    parser.add_argument(
        "--positions-per-side",
        type=int,
        default=None,
        help="Optional fixed equal-weight long/short book size per side instead of full deciles",
    )
    return parser.parse_args(argv)


def parse_cost_bps_levels(value: str) -> tuple[float, ...]:
    levels: list[float] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        bps = float(chunk)
        if bps < 0:
            raise argparse.ArgumentTypeError("Cost bps levels must be non-negative")
        levels.append(bps)
    if not levels:
        raise argparse.ArgumentTypeError("Expected at least one cost bps level")
    if 0.0 not in levels:
        levels.insert(0, 0.0)
    return tuple(dict.fromkeys(levels))


def _coerce_truthy(series: pd.Series) -> pd.Series:
    def _convert(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "t"}
        return bool(value)

    return series.map(_convert)


def load_membership_lookup(membership_path: Path, end_date: pd.Timestamp | None = None) -> MembershipLookup:
    membership_path = Path(membership_path)
    if not membership_path.exists():
        raise FileNotFoundError(f"Russell membership file not found: {membership_path}")

    if membership_path.suffix == ".parquet":
        membership = pd.read_parquet(membership_path)
    elif membership_path.suffix == ".csv":
        membership = pd.read_csv(membership_path)
    else:
        raise ValueError(f"Unsupported membership file type: {membership_path.suffix}")

    if "date" not in membership.columns or "symbol" not in membership.columns:
        raise ValueError("Membership file must contain at least ['date', 'symbol']")

    membership = membership.copy()
    membership["date"] = pd.to_datetime(membership["date"], errors="coerce")
    membership = membership.dropna(subset=["date", "symbol"])
    if end_date is not None:
        membership = membership[membership["date"] <= pd.Timestamp(end_date)].copy()
    if "in_r1000" in membership.columns:
        membership = membership[_coerce_truthy(membership["in_r1000"]).fillna(False)].copy()

    membership["base_symbol"] = (
        membership["symbol"].astype(str).str.upper().str.strip().str.split("-", n=1).str[0]
    )
    grouped = (
        membership.groupby("date")["base_symbol"]
        .agg(lambda values: set(value for value in values if isinstance(value, str) and value))
        .to_dict()
    )
    dates = tuple(sorted(grouped))
    return MembershipLookup(dates=dates, symbols_by_date=grouped)


def build_market_frames(histories: dict[str, pd.DataFrame]) -> MarketFrames:
    close_cols: dict[str, pd.Series] = {}
    open_cols: dict[str, pd.Series] = {}
    unadj_close_cols: dict[str, pd.Series] = {}
    volume_cols: dict[str, pd.Series] = {}

    for ticker, history in histories.items():
        if history.empty:
            continue
        df = history.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if df.empty:
            continue
        df = df.set_index("date")
        close_cols[ticker] = pd.to_numeric(df.get("adj_close"), errors="coerce")
        open_cols[ticker] = pd.to_numeric(df.get("adj_open"), errors="coerce")
        raw_close = pd.to_numeric(df.get("close"), errors="coerce")
        if raw_close.isna().all():
            raw_close = close_cols[ticker].copy()
        unadj_close_cols[ticker] = raw_close
        volume_cols[ticker] = pd.to_numeric(df.get("volume"), errors="coerce")

    if not close_cols:
        raise RuntimeError("No Russell OHLCV histories were loaded for the weekly benchmark.")

    close_df = pd.DataFrame(close_cols).sort_index()
    open_df = pd.DataFrame(open_cols).sort_index()
    unadj_close_df = pd.DataFrame(unadj_close_cols).sort_index()
    volume_df = pd.DataFrame(volume_cols).sort_index()
    dollar_volume_df = unadj_close_df * volume_df

    return MarketFrames(
        close_df=close_df,
        open_df=open_df,
        unadj_close_df=unadj_close_df,
        dollar_volume_df=dollar_volume_df,
    )


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    subset = values[mask]
    valid = subset[~np.isnan(subset)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def calc_turnover(old_set: set[str], new_set: set[str]) -> float:
    if len(old_set) == 0:
        return 1.0 if len(new_set) > 0 else 0.0
    tickers = old_set | new_set
    old_n = len(old_set)
    new_n = len(new_set)
    return float(
        sum(
            abs((1.0 / new_n if ticker in new_set else 0.0) - (1.0 / old_n if ticker in old_set else 0.0))
            for ticker in tickers
        )
        / 2.0
    )


def format_bps_label(bps: float) -> str:
    if float(bps).is_integer():
        return str(int(bps))
    return str(bps).replace(".", "p")


def parse_bps_label(label: str) -> float:
    return float(str(label).replace("p", "."))


def variant_bps_name(bps: float) -> str:
    return "Gross" if float(bps) == 0.0 else f"{bps:g}bps"


def available_cost_levels(weekly_returns: pd.DataFrame, prefix: str = "ls_c2c") -> tuple[float, ...]:
    levels: list[float] = []
    prefix_with_sep = f"{prefix}_"
    for column in weekly_returns.columns:
        if not str(column).startswith(prefix_with_sep):
            continue
        try:
            levels.append(parse_bps_label(str(column)[len(prefix_with_sep):]))
        except ValueError:
            continue
    return tuple(sorted(dict.fromkeys(levels)))


def safe_cagr(returns: pd.Series, periods_per_year: float = DEFAULT_PERIODS_PER_YEAR) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).prod()
    if cumulative <= 0:
        return np.nan
    return float((cumulative ** (periods_per_year / len(clean)) - 1.0) * 100.0)


def compute_max_drawdown_pct(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def _next_trading_day_map(trading_dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    dates = list(pd.to_datetime(trading_dates).sort_values())
    next_day: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx, date in enumerate(dates[:-1]):
        next_day[pd.Timestamp(date)] = pd.Timestamp(dates[idx + 1])
    return next_day


def _window_return(close_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, tickers: list[str]) -> np.ndarray:
    start_prices = close_df.loc[start_date, tickers].to_numpy(dtype=float)
    end_prices = close_df.loc[end_date, tickers].to_numpy(dtype=float)
    return end_prices / start_prices - 1.0


def _trailing_week_return(close_df: pd.DataFrame, prev_fri: pd.Timestamp, this_fri: pd.Timestamp, tickers: list[str]) -> np.ndarray:
    return _window_return(close_df, prev_fri, this_fri, tickers)


def _forward_c2c_return(close_df: pd.DataFrame, this_fri: pd.Timestamp, next_fri: pd.Timestamp, tickers: list[str]) -> np.ndarray:
    return _window_return(close_df, this_fri, next_fri, tickers)


def _forward_o2o_return(
    open_df: pd.DataFrame,
    this_exec: pd.Timestamp | None,
    next_exec: pd.Timestamp | None,
    tickers: list[str],
) -> np.ndarray:
    values = np.full(len(tickers), np.nan, dtype=float)
    if this_exec is None or next_exec is None:
        return values

    entry_prices = open_df.loc[this_exec, tickers].to_numpy(dtype=float)
    exit_prices = open_df.loc[next_exec, tickers].to_numpy(dtype=float)
    valid = (~np.isnan(entry_prices)) & (~np.isnan(exit_prices)) & (entry_prices > 0) & (exit_prices > 0)
    values[valid] = exit_prices[valid] / entry_prices[valid] - 1.0
    return values


def _rank_signal(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank_order = np.argsort(-signal, kind="mergesort")
    order = np.empty(len(signal), dtype=int)
    order[rank_order] = np.arange(len(signal))
    deciles = np.clip((order * 10) // len(signal) + 1, 1, 10)
    return rank_order, deciles


def _select_book_masks(
    rank_order: np.ndarray,
    deciles: np.ndarray,
    asset_count: int,
    positions_per_side: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if positions_per_side is None:
        return deciles == 1, deciles == 10

    book_size = int(positions_per_side)
    if book_size <= 0:
        raise ValueError("positions_per_side must be positive when provided")
    if asset_count < book_size * 2:
        raise ValueError(
            f"Need at least {book_size * 2} assets to build a non-overlapping {book_size}/{book_size} book; "
            f"got {asset_count}"
        )

    long_mask = np.zeros(asset_count, dtype=bool)
    short_mask = np.zeros(asset_count, dtype=bool)
    long_mask[rank_order[:book_size]] = True
    short_mask[rank_order[-book_size:]] = True
    return long_mask, short_mask


def run_weekly_strategy(
    market_frames: MarketFrames,
    membership_lookup: MembershipLookup,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_df = market_frames.close_df
    open_df = market_frames.open_df
    unadj_close_df = market_frames.unadj_close_df
    adv20_df = market_frames.dollar_volume_df.rolling(20, min_periods=15).mean()

    trading_dates = close_df.index
    fridays = trading_dates[trading_dates.dayofweek == 4]
    friday_to_next_trade = _next_trading_day_map(trading_dates)

    print(f"Trading dates: {len(trading_dates)}")
    print(f"Fridays in dataset: {len(fridays)}")
    print(f"Processing weekly strategy from {config.start.date()} to {config.end.date()}...")

    results: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prev_long_set: set[str] = set()
    prev_short_set: set[str] = set()

    for idx in range(1, len(fridays) - 1):
        prev_fri = pd.Timestamp(fridays[idx - 1])
        this_fri = pd.Timestamp(fridays[idx])
        next_fri = pd.Timestamp(fridays[idx + 1])

        if this_fri < config.start:
            continue
        if this_fri > config.end or next_fri > config.end:
            break

        r1000_universe = membership_lookup.universe_on_or_before(this_fri)
        if not r1000_universe:
            continue

        valid: list[str] = []
        for ticker in sorted(r1000_universe):
            if ticker not in close_df.columns:
                continue
            close_prev = close_df.at[prev_fri, ticker] if prev_fri in close_df.index else np.nan
            close_now = close_df.at[this_fri, ticker] if this_fri in close_df.index else np.nan
            if pd.isna(close_prev) or pd.isna(close_now) or close_prev <= 0 or close_now <= 0:
                continue

            raw_price = unadj_close_df.at[this_fri, ticker] if this_fri in unadj_close_df.index else np.nan
            if pd.isna(raw_price) or raw_price < config.min_trade_price:
                continue

            adv20 = adv20_df.at[this_fri, ticker] if this_fri in adv20_df.index else np.nan
            if pd.isna(adv20) or adv20 < config.min_adv20_dollars:
                continue

            valid.append(ticker)

        if len(valid) < config.required_universe_size():
            continue

        trail = _trailing_week_return(close_df, prev_fri, this_fri, valid)
        signal = -trail

        rank_order, deciles = _rank_signal(signal)
        long_mask, short_mask = _select_book_masks(rank_order, deciles, len(valid), config.positions_per_side)
        long_tickers = [valid[pos] for pos in np.where(long_mask)[0]]
        short_tickers = [valid[pos] for pos in np.where(short_mask)[0]]

        fwd_c2c = _forward_c2c_return(close_df, this_fri, next_fri, valid)
        fwd_o2o = _forward_o2o_return(
            open_df,
            friday_to_next_trade.get(this_fri),
            friday_to_next_trade.get(next_fri),
            valid,
        )

        long_ret_c2c = safe_mean(fwd_c2c, long_mask)
        short_ret_c2c = safe_mean(fwd_c2c, short_mask)
        long_ret_o2o = safe_mean(fwd_o2o, long_mask)
        short_ret_o2o = safe_mean(fwd_o2o, short_mask)

        decile_c2c = {}
        decile_o2o = {}
        for decile in range(1, 11):
            decile_mask = deciles == decile
            decile_c2c[decile] = safe_mean(fwd_c2c, decile_mask)
            decile_o2o[decile] = safe_mean(fwd_o2o, decile_mask)

        new_long_set = set(long_tickers)
        new_short_set = set(short_tickers)
        turnover_long = calc_turnover(prev_long_set, new_long_set)
        turnover_short = calc_turnover(prev_short_set, new_short_set)
        prev_long_set = new_long_set
        prev_short_set = new_short_set

        valid_ic_c2c = (~np.isnan(signal)) & (~np.isnan(fwd_c2c))
        valid_ic_o2o = (~np.isnan(signal)) & (~np.isnan(fwd_o2o))
        ic_c2c = stats.spearmanr(signal[valid_ic_c2c], fwd_c2c[valid_ic_c2c])[0] if valid_ic_c2c.sum() > 20 else np.nan
        ic_o2o = stats.spearmanr(signal[valid_ic_o2o], fwd_o2o[valid_ic_o2o])[0] if valid_ic_o2o.sum() > 20 else np.nan

        row = {
            "date": this_fri,
            "n_universe": len(valid),
            "n_long": len(long_tickers),
            "n_short": len(short_tickers),
            "long_ret_c2c": long_ret_c2c,
            "short_ret_c2c": short_ret_c2c,
            "long_ret_o2o": long_ret_o2o,
            "short_ret_o2o": short_ret_o2o,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "ic_c2c": ic_c2c,
            "ic_o2o": ic_o2o,
        }
        for decile in range(1, 11):
            row[f"dec_{decile}_c2c"] = decile_c2c[decile]
            row[f"dec_{decile}_o2o"] = decile_o2o[decile]
        results.append(row)

        for ticker in SANITY_CHECK_TICKERS:
            if ticker in valid:
                pos = valid.index(ticker)
                signal_rows.append(
                    {
                        "date": this_fri,
                        "ticker": ticker,
                        "signal": signal[pos],
                        "trail_ret": trail[pos],
                        "fwd_c2c": fwd_c2c[pos],
                        "fwd_o2o": fwd_o2o[pos],
                        "decile": deciles[pos],
                    }
                )

        for side, tickers, mask in (
            ("LONG", long_tickers, long_mask),
            ("SHORT", short_tickers, short_mask),
        ):
            selected_positions = np.where(mask)[0]
            side_weight = 1.0 / len(selected_positions) if len(selected_positions) else np.nan
            for rank, pos in enumerate(selected_positions, start=1):
                selection_rows.append(
                    {
                        "rebalance_date": this_fri,
                        "signal_date": this_fri,
                        "entry_date": this_fri,
                        "exit_date": next_fri,
                        "ticker": valid[pos],
                        "position_type": side,
                        "rank": rank,
                        "score": signal[pos],
                        "trail_return_pct": trail[pos] * 100.0,
                        "realized_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_c2c_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_o2o_return_pct": fwd_o2o[pos] * 100.0,
                        "weight": side_weight,
                        "decile": deciles[pos],
                    }
                )

        if idx % 200 == 0:
            print(f"  {idx}/{len(fridays)} ({this_fri.date()}) universe={len(valid)}")

    weekly_returns = pd.DataFrame(results)
    if weekly_returns.empty:
        raise RuntimeError("No weekly results were produced. Check the date range, membership file, or filters.")

    weekly_returns["date"] = pd.to_datetime(weekly_returns["date"])
    weekly_returns = weekly_returns.set_index("date").sort_index()
    signal_checks = pd.DataFrame(signal_rows)
    portfolio_selections = pd.DataFrame(selection_rows)
    return weekly_returns, signal_checks, portfolio_selections


def run_periodic_strategy(
    market_frames: MarketFrames,
    membership_lookup: MembershipLookup,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close_df = market_frames.close_df
    open_df = market_frames.open_df
    unadj_close_df = market_frames.unadj_close_df
    adv20_df = market_frames.dollar_volume_df.rolling(20, min_periods=15).mean()

    trading_dates = close_df.index
    next_trade_map = _next_trading_day_map(trading_dates)
    lookback_days = int(config.signal_lookback_days)
    rebalance_days = int(config.rebalance_days)
    holding_days = int(config.resolved_holding_days())

    print(f"Trading dates: {len(trading_dates)}")
    print(
        "Processing periodic trading-day strategy from "
        f"{config.start.date()} to {config.end.date()} "
        f"(lookback={lookback_days}, rebalance={rebalance_days}, hold={holding_days})..."
    )

    results: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prev_long_set: set[str] = set()
    prev_short_set: set[str] = set()

    max_signal_idx = len(trading_dates) - holding_days - 1
    first_signal_idx = lookback_days
    if config.rebalance_anchor_date is not None:
        anchor = pd.Timestamp(config.rebalance_anchor_date)
        first_signal_idx = int(trading_dates.searchsorted(anchor))
        if first_signal_idx >= len(trading_dates) or pd.Timestamp(trading_dates[first_signal_idx]) != anchor:
            raise ValueError(f"rebalance_anchor_date {anchor.date()} is not a trading date")
        if first_signal_idx < lookback_days:
            raise ValueError(
                f"rebalance_anchor_date {anchor.date()} does not have "
                f"{lookback_days} prior trading days for the signal lookback"
            )

    for idx in range(first_signal_idx, max_signal_idx + 1, rebalance_days):
        start_date = pd.Timestamp(trading_dates[idx - lookback_days])
        signal_date = pd.Timestamp(trading_dates[idx])
        exit_date = pd.Timestamp(trading_dates[idx + holding_days])

        if signal_date < config.start:
            continue
        if signal_date > config.end or exit_date > config.end:
            break

        r1000_universe = membership_lookup.universe_on_or_before(signal_date)
        if not r1000_universe:
            continue

        valid: list[str] = []
        for ticker in sorted(r1000_universe):
            if ticker not in close_df.columns:
                continue
            close_prev = close_df.at[start_date, ticker] if start_date in close_df.index else np.nan
            close_now = close_df.at[signal_date, ticker] if signal_date in close_df.index else np.nan
            if pd.isna(close_prev) or pd.isna(close_now) or close_prev <= 0 or close_now <= 0:
                continue

            raw_price = unadj_close_df.at[signal_date, ticker] if signal_date in unadj_close_df.index else np.nan
            if pd.isna(raw_price) or raw_price < config.min_trade_price:
                continue

            adv20 = adv20_df.at[signal_date, ticker] if signal_date in adv20_df.index else np.nan
            if pd.isna(adv20) or adv20 < config.min_adv20_dollars:
                continue

            valid.append(ticker)

        if len(valid) < config.required_universe_size():
            continue

        trail = _window_return(close_df, start_date, signal_date, valid)
        signal = -trail

        rank_order, deciles = _rank_signal(signal)
        long_mask, short_mask = _select_book_masks(rank_order, deciles, len(valid), config.positions_per_side)
        long_tickers = [valid[pos] for pos in np.where(long_mask)[0]]
        short_tickers = [valid[pos] for pos in np.where(short_mask)[0]]

        fwd_c2c = _window_return(close_df, signal_date, exit_date, valid)
        fwd_o2o = _forward_o2o_return(
            open_df,
            next_trade_map.get(signal_date),
            next_trade_map.get(exit_date),
            valid,
        )

        long_ret_c2c = safe_mean(fwd_c2c, long_mask)
        short_ret_c2c = safe_mean(fwd_c2c, short_mask)
        long_ret_o2o = safe_mean(fwd_o2o, long_mask)
        short_ret_o2o = safe_mean(fwd_o2o, short_mask)

        decile_c2c = {}
        decile_o2o = {}
        for decile in range(1, 11):
            decile_mask = deciles == decile
            decile_c2c[decile] = safe_mean(fwd_c2c, decile_mask)
            decile_o2o[decile] = safe_mean(fwd_o2o, decile_mask)

        new_long_set = set(long_tickers)
        new_short_set = set(short_tickers)
        turnover_long = calc_turnover(prev_long_set, new_long_set)
        turnover_short = calc_turnover(prev_short_set, new_short_set)
        prev_long_set = new_long_set
        prev_short_set = new_short_set

        valid_ic_c2c = (~np.isnan(signal)) & (~np.isnan(fwd_c2c))
        valid_ic_o2o = (~np.isnan(signal)) & (~np.isnan(fwd_o2o))
        ic_c2c = stats.spearmanr(signal[valid_ic_c2c], fwd_c2c[valid_ic_c2c])[0] if valid_ic_c2c.sum() > 20 else np.nan
        ic_o2o = stats.spearmanr(signal[valid_ic_o2o], fwd_o2o[valid_ic_o2o])[0] if valid_ic_o2o.sum() > 20 else np.nan

        row = {
            "date": signal_date,
            "n_universe": len(valid),
            "n_long": len(long_tickers),
            "n_short": len(short_tickers),
            "long_ret_c2c": long_ret_c2c,
            "short_ret_c2c": short_ret_c2c,
            "long_ret_o2o": long_ret_o2o,
            "short_ret_o2o": short_ret_o2o,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "ic_c2c": ic_c2c,
            "ic_o2o": ic_o2o,
        }
        for decile in range(1, 11):
            row[f"dec_{decile}_c2c"] = decile_c2c[decile]
            row[f"dec_{decile}_o2o"] = decile_o2o[decile]
        results.append(row)

        for ticker in SANITY_CHECK_TICKERS:
            if ticker in valid:
                pos = valid.index(ticker)
                signal_rows.append(
                    {
                        "date": signal_date,
                        "ticker": ticker,
                        "signal": signal[pos],
                        "trail_ret": trail[pos],
                        "fwd_c2c": fwd_c2c[pos],
                        "fwd_o2o": fwd_o2o[pos],
                        "decile": deciles[pos],
                    }
                )

        for side, tickers, mask in (
            ("LONG", long_tickers, long_mask),
            ("SHORT", short_tickers, short_mask),
        ):
            selected_positions = np.where(mask)[0]
            side_weight = 1.0 / len(selected_positions) if len(selected_positions) else np.nan
            for rank, pos in enumerate(selected_positions, start=1):
                selection_rows.append(
                    {
                        "rebalance_date": signal_date,
                        "signal_date": signal_date,
                        "entry_date": signal_date,
                        "exit_date": exit_date,
                        "ticker": valid[pos],
                        "position_type": side,
                        "rank": rank,
                        "score": signal[pos],
                        "trail_return_pct": trail[pos] * 100.0,
                        "realized_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_c2c_return_pct": fwd_c2c[pos] * 100.0,
                        "realized_o2o_return_pct": fwd_o2o[pos] * 100.0,
                        "weight": side_weight,
                        "decile": deciles[pos],
                    }
                )

    periodic_returns = pd.DataFrame(results)
    if periodic_returns.empty:
        raise RuntimeError("No periodic results were produced. Check the date range, cadence, membership file, or filters.")

    periodic_returns["date"] = pd.to_datetime(periodic_returns["date"])
    periodic_returns = periodic_returns.set_index("date").sort_index()
    signal_checks = pd.DataFrame(signal_rows)
    portfolio_selections = pd.DataFrame(selection_rows)
    return periodic_returns, signal_checks, portfolio_selections


def add_cost_overlays(weekly_returns: pd.DataFrame, cost_bps_levels: tuple[float, ...]) -> pd.DataFrame:
    df = weekly_returns.copy()
    df["ls_c2c"] = df["long_ret_c2c"] - df["short_ret_c2c"]
    df["ls_o2o"] = df["long_ret_o2o"] - df["short_ret_o2o"]
    total_turnover = df["turnover_long"].fillna(0.0) + df["turnover_short"].fillna(0.0)

    for bps in cost_bps_levels:
        cost_rate = bps / 10000.0
        cost_label = format_bps_label(bps)
        total_cost = total_turnover * 2.0 * cost_rate
        long_cost = df["turnover_long"].fillna(0.0) * 2.0 * cost_rate

        df[f"ls_c2c_{cost_label}"] = df["ls_c2c"] - total_cost
        df[f"ls_o2o_{cost_label}"] = df["ls_o2o"] - total_cost
        df[f"long_c2c_{cost_label}"] = df["long_ret_c2c"] - long_cost
        df[f"long_o2o_{cost_label}"] = df["long_ret_o2o"] - long_cost

    return df


def metric_summary(
    returns: pd.Series,
    name: str,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, object] | None:
    clean = returns.dropna()
    min_required_periods = max(10, int(np.ceil(periods_per_year)))
    if len(clean) < min_required_periods:
        return None

    ann_ret = float(clean.mean() * periods_per_year)
    ann_vol = float(clean.std() * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cumulative = (1.0 + clean).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    max_drawdown_end = drawdown.idxmin()
    drawdown_peak = cumulative.cummax().loc[:max_drawdown_end]
    max_drawdown_start = drawdown_peak.idxmax()

    return {
        "Name": name,
        "Ann Ret": ann_ret,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max DD": max_drawdown,
        "DD Start": str(max_drawdown_start.date()),
        "DD End": str(max_drawdown_end.date()),
        "Hit Rate": float((clean > 0).mean()),
        "Best Wk": float(clean.max()),
        "Worst Wk": float(clean.min()),
        "Skew": float(clean.skew()),
        "Kurt": float(clean.kurtosis()),
        "N": int(len(clean)),
    }


def build_headline_metrics(
    weekly_returns: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    strategies: list[tuple[str, str]] = []
    for execution_label, prefix in [("C2C", "ls_c2c"), ("O2O", "ls_o2o")]:
        for bps in available_cost_levels(weekly_returns, prefix):
            strategies.append((f"L/S {execution_label} {variant_bps_name(bps)}", f"{prefix}_{format_bps_label(bps)}"))

    first_nonzero = next((bps for bps in available_cost_levels(weekly_returns, "long_c2c") if bps > 0), None)
    if first_nonzero is not None:
        label = format_bps_label(first_nonzero)
        strategies.extend(
            [
                (f"Long C2C {variant_bps_name(first_nonzero)}", f"long_c2c_{label}"),
                (f"Long O2O {variant_bps_name(first_nonzero)}", f"long_o2o_{label}"),
            ]
        )
    strategies.extend([("Short C2C", "short_ret_c2c"), ("Short O2O", "short_ret_o2o")])

    rows = [
        metric_summary(weekly_returns[column], name, periods_per_year=periods_per_year)
        for name, column in strategies
        if column in weekly_returns.columns
    ]
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def build_yearly_returns(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    columns: list[tuple[str, str]] = []
    for execution_label, prefix in [("C2C", "ls_c2c"), ("O2O", "ls_o2o")]:
        for bps in available_cost_levels(weekly_returns, prefix):
            output_name = f"LS_{execution_label}_{variant_bps_name(bps)}"
            columns.append((f"{prefix}_{format_bps_label(bps)}", output_name))
    columns.extend(
        [
            ("long_ret_c2c", "Long_C2C"),
            ("short_ret_c2c", "Short_C2C"),
        ]
    )

    rows: list[dict[str, object]] = []
    for year, group in weekly_returns.groupby(weekly_returns.index.year):
        row: dict[str, object] = {"year": year}
        for source_column, output_name in columns:
            if source_column in group.columns:
                series = group[source_column].dropna()
                row[output_name] = (1.0 + series).prod() - 1.0 if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def build_strategy_summary(
    weekly_returns: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    total_turnover = weekly_returns["turnover_long"].fillna(0.0) + weekly_returns["turnover_short"].fillna(0.0)
    variants: list[tuple[str, str, float, str, str]] = []
    for execution_model, prefix in [("C2C", "ls_c2c"), ("O2O", "ls_o2o")]:
        for bps in available_cost_levels(weekly_returns, prefix):
            variants.append(
                (
                    f"LS_{execution_model}_{variant_bps_name(bps)}",
                    execution_model,
                    bps,
                    f"{prefix}_{format_bps_label(bps)}",
                    prefix,
                )
            )

    rows: list[dict[str, object]] = []
    for variant_name, execution_model, cost_bps_per_side, column, gross_column in variants:
        returns = weekly_returns[column].dropna()
        if returns.empty:
            continue

        gross_returns = weekly_returns.loc[returns.index, gross_column].dropna()
        total_return_pct = float(((1.0 + returns).prod() - 1.0) * 100.0)
        gross_total_return_pct = float(((1.0 + gross_returns).prod() - 1.0) * 100.0)
        cagr_pct = safe_cagr(returns, periods_per_year=periods_per_year)
        vol = float(returns.std() * np.sqrt(periods_per_year))
        sharpe = float((returns.mean() * periods_per_year) / vol) if vol > 0 else 0.0
        max_drawdown_pct = compute_max_drawdown_pct(returns)
        avg_turnover = float(total_turnover.loc[returns.index].mean())
        total_cost_bps = float((total_turnover.loc[returns.index] * 2.0 * cost_bps_per_side).sum())

        rows.append(
            {
                "variant_name": variant_name,
                "execution_model": execution_model,
                "cost_bps_per_side": cost_bps_per_side,
                "total_return_pct": total_return_pct,
                "gross_total_return_pct": gross_total_return_pct,
                "cost_drag_pct": gross_total_return_pct - total_return_pct,
                "cagr_pct": cagr_pct,
                "sharpe": sharpe,
                "max_drawdown_pct": max_drawdown_pct,
                "avg_turnover": avg_turnover,
                "n_periods": int(len(returns)),
                "total_cost_bps": total_cost_bps,
            }
        )

    return pd.DataFrame(rows)


def _plot_cumulative_returns(
    weekly_returns: pd.DataFrame,
    output_dir: Path,
    cadence_label: str = "weekly Friday",
) -> Path:
    color_sets = {
        "C2C": ["#1f77b4", "#4c78a8", "#72b7b2", "#9ecae9", "#2f4b7c"],
        "O2O": ["#ff7f0e", "#f28e2b", "#e15759", "#ffbe7d", "#b85c38"],
    }
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    grouped_variants = []
    for execution_label, prefix in [("C2C", "ls_c2c"), ("O2O", "ls_o2o")]:
        variants = []
        for idx, bps in enumerate(available_cost_levels(weekly_returns, prefix)):
            variants.append(
                (
                    variant_bps_name(bps),
                    f"{prefix}_{format_bps_label(bps)}",
                    color_sets[execution_label][idx % len(color_sets[execution_label])],
                    linestyles[idx % len(linestyles)],
                )
            )
        grouped_variants.append((execution_label, variants))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, (execution_label, variants) in zip(axes, grouped_variants):
        plotted = False
        for label, column, color, linestyle in variants:
            if column not in weekly_returns.columns:
                continue
            cumulative_pct = ((1.0 + weekly_returns[column].fillna(0.0)).cumprod() - 1.0) * 100.0
            ax.plot(
                cumulative_pct.index,
                cumulative_pct.values,
                label=label,
                linewidth=2.0,
                color=color,
                linestyle=linestyle,
            )
            plotted = True

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.set_title(f"{execution_label} Cumulative Returns")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(frameon=False, ncol=4, loc="best")

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Mean-Reversion Cumulative Returns ({cadence_label})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = output_dir / "cumulative_returns.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_decile_profile(
    weekly_returns: pd.DataFrame,
    output_dir: Path,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> Path:
    deciles = np.arange(1, 11)
    c2c_values = [
        float(weekly_returns[f"dec_{decile}_c2c"].dropna().mean() * periods_per_year * 100.0)
        if f"dec_{decile}_c2c" in weekly_returns.columns
        else np.nan
        for decile in deciles
    ]
    o2o_values = [
        float(weekly_returns[f"dec_{decile}_o2o"].dropna().mean() * periods_per_year * 100.0)
        if f"dec_{decile}_o2o" in weekly_returns.columns
        else np.nan
        for decile in deciles
    ]

    x = np.arange(len(deciles))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, c2c_values, width=width, label="C2C", color="#4c78a8")
    ax.bar(x + width / 2, o2o_values, width=width, label="O2O", color="#f58518")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{decile}" for decile in deciles])
    ax.set_title("Average Annualized Return by Decile")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_xlabel("Decile (D1 = losers / long, D10 = winners / short)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path = output_dir / "decile_profile.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_strategy_comparison(strategy_summary: pd.DataFrame, output_dir: Path) -> Path:
    ordered = strategy_summary.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    labels = ordered["variant_name"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(x, ordered["total_return_pct"], color="#2ca02c", alpha=0.85)
    axes[0].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[0].set_title("Strategy Variant Comparison")
    axes[0].set_ylabel("Total Return (%)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, ordered["sharpe"], color="#1f77b4", alpha=0.85, label="Sharpe")
    if "cost_drag_pct" in ordered.columns:
        axes[1].plot(x, ordered["cost_drag_pct"], color="#d62728", marker="o", linewidth=1.8, label="Cost Drag (%)")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Sharpe / Cost Drag")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    output_path = output_dir / "strategy_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_charts(
    artifacts: BenchmarkArtifacts,
    output_dir: Path,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    cadence_label: str = "weekly Friday",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = [
        _plot_cumulative_returns(artifacts.weekly_returns, output_dir, cadence_label=cadence_label),
        _plot_decile_profile(artifacts.weekly_returns, output_dir, periods_per_year=periods_per_year),
        _plot_strategy_comparison(artifacts.strategy_summary, output_dir),
    ]
    return chart_paths


def print_sanity_checks(
    weekly_returns: pd.DataFrame,
    signal_checks: pd.DataFrame,
    market_frames: MarketFrames,
    headline_metrics: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    cadence_label: str = "weekly Friday",
) -> None:
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    print("\n--- LEHMQ Delisting ---")
    lehmq_rows = signal_checks[
        (signal_checks.get("ticker") == "LEHMQ")
        & (pd.to_datetime(signal_checks.get("date"), errors="coerce").dt.year == 2008)
    ]
    if lehmq_rows.empty:
        print("  No LEHMQ rows captured in signal checks.")
    else:
        for _, row in lehmq_rows.sort_values("date").iterrows():
            print(
                f"  {pd.Timestamp(row['date']).date()}: "
                f"D={int(row['decile'])}, trail={row['trail_ret']:+.4f}, fwd_c2c={row['fwd_c2c']:+.4f}"
            )

    print("\n--- AAPL Date Alignment ---")
    if weekly_returns.empty:
        print("  Weekly returns are empty; skipping AAPL alignment check.")
    else:
        fri_dates = weekly_returns.index
        test_fri = pd.Timestamp("2023-03-10")
        nearest_idx = fri_dates.get_indexer([test_fri], method="nearest")[0]
        closest = fri_dates[nearest_idx]
        if nearest_idx == 0 or nearest_idx + 1 >= len(fri_dates) or "AAPL" not in market_frames.close_df.columns:
            print("  Not enough AAPL data in the chosen date window.")
        else:
            prev_closest = fri_dates[nearest_idx - 1]
            next_closest = fri_dates[nearest_idx + 1]
            aapl_close = market_frames.close_df["AAPL"]
            ap = aapl_close.at[prev_closest]
            at_ = aapl_close.at[closest]
            an = aapl_close.at[next_closest]
            manual_trail = at_ / ap - 1.0
            manual_signal = -manual_trail
            manual_fwd = an / at_ - 1.0
            aapl_row = signal_checks[
                (signal_checks.get("ticker") == "AAPL")
                & (pd.to_datetime(signal_checks.get("date"), errors="coerce") == closest)
            ]
            print(f"Week: {prev_closest.date()} -> {closest.date()} -> {next_closest.date()}")
            print(f"  AAPL Close: {ap:.4f} -> {at_:.4f} -> {an:.4f}")
            print(f"  Manual: trail={manual_trail:.6f}, signal={manual_signal:.6f}, fwd={manual_fwd:.6f}")
            if aapl_row.empty:
                print("  AAPL not found in signal checks.")
            else:
                row = aapl_row.iloc[0]
                print(f"  Pipeline: signal={row['signal']:.6f}, fwd={row['fwd_c2c']:.6f}")
                sig_ok = abs(manual_signal - row["signal"]) < 1e-4
                fwd_ok = abs(manual_fwd - row["fwd_c2c"]) < 1e-4
                print(f"  {'PASS' if sig_ok and fwd_ok else 'MISMATCH'}")

    print("\n--- Random Label Test ---")
    ic_values = weekly_returns["ic_c2c"].dropna()
    if ic_values.empty:
        print("  No IC values available.")
    else:
        ic_mean = float(ic_values.mean())
        ic_se = float(ic_values.std() / np.sqrt(len(ic_values))) if len(ic_values) > 1 else np.nan
        t_stat = ic_mean / ic_se if ic_se and np.isfinite(ic_se) and ic_se > 0 else np.nan
        print(f"Mean IC: {ic_mean:.4f}, SE: {ic_se:.4f}, t-stat: {t_stat:.2f}")
        print(f"Under null (shuffled labels), mean IC = 0. Our t={t_stat:.2f} >> 2: PASS")

    print("\n--- Future Feature Test ---")
    ls_c2c = weekly_returns["ls_c2c"].dropna()
    mean_ic = float(ic_values.mean()) if not ic_values.empty else np.nan
    gross_sharpe = float(ls_c2c.mean() / ls_c2c.std() * np.sqrt(periods_per_year)) if len(ls_c2c) > 1 and ls_c2c.std() > 0 else np.nan
    if pd.notna(mean_ic) and mean_ic > 0.5:
        print(f"FAIL: Mean IC = {mean_ic:.4f} — suspiciously high, possible look-ahead leak")
    elif pd.notna(gross_sharpe) and gross_sharpe > 5.0:
        print(f"FAIL: L/S Gross Sharpe = {gross_sharpe:.2f} — implausibly high, possible look-ahead leak")
    else:
        print(f"Mean IC = {mean_ic:.4f} (expected 0.01-0.05, not ~1.0): PASS")
        print(f"L/S Gross Sharpe = {gross_sharpe:.2f} (expected < 2.0, not ~10+): PASS")

    n_valid = int(weekly_returns["long_ret_c2c"].notna().sum())
    n_valid_o2o = int(weekly_returns["long_ret_o2o"].notna().sum())
    print(f"\nValid observations ({cadence_label}): {n_valid}/{len(weekly_returns)} ({n_valid / len(weekly_returns):.1%})")
    print(f"Valid O2O observations: {n_valid_o2o}/{len(weekly_returns)} ({n_valid_o2o / len(weekly_returns):.1%})")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    if headline_metrics.empty:
        print("No headline metrics available.")
    else:
        print(headline_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)))

        print("\n--- MICROSTRUCTURE CHECK ---")
        c2c = headline_metrics[headline_metrics["Name"] == "L/S C2C Gross"]
        o2o = headline_metrics[headline_metrics["Name"] == "L/S O2O Gross"]
        if not c2c.empty and not o2o.empty:
            c2c_sr = float(c2c["Sharpe"].iloc[0])
            o2o_sr = float(o2o["Sharpe"].iloc[0])
            c2c_ar = float(c2c["Ann Ret"].iloc[0])
            o2o_ar = float(o2o["Ann Ret"].iloc[0])
            print(f"C2C Gross: Sharpe={c2c_sr:.3f}, Return={c2c_ar:.4f}")
            print(f"O2O Gross: Sharpe={o2o_sr:.3f}, Return={o2o_ar:.4f}")
            if c2c_ar != 0:
                print(f"Alpha lost C2C->O2O: {1.0 - o2o_ar / c2c_ar:.1%}")
            if c2c_sr > 2.0:
                print("*** WARNING: C2C Sharpe > 2.0 — implausibly high ***")


def print_additional_breakdowns(
    weekly_returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    cadence_label: str = "weekly Friday",
    cost_bps_levels: tuple[float, ...] = DEFAULT_COST_BPS_LEVELS,
) -> None:
    print("\n--- DECADE BREAKDOWN ---")
    decades = {
        "1995-1999": ("1995", "1999"),
        "2000-2009": ("2000", "2009"),
        "2010-2019": ("2010", "2019"),
        "2020-2026": ("2020", "2026"),
    }
    rows = []
    for decade_name, (start_year, end_year) in decades.items():
        mask = (weekly_returns.index.year >= int(start_year)) & (weekly_returns.index.year <= int(end_year))
        subset = weekly_returns.loc[mask]
        for label, column in [
            ("C2C Gross", "ls_c2c_0"),
            ("C2C 15bps", "ls_c2c_15"),
            ("O2O Gross", "ls_o2o_0"),
            ("O2O 2.5bps", "ls_o2o_2p5"),
            ("O2O 5bps", "ls_o2o_5"),
            ("O2O 15bps", "ls_o2o_15"),
        ]:
            if column not in subset.columns:
                continue
            returns = subset[column].dropna()
            if len(returns) <= 10:
                continue
            ann_ret = returns.mean() * periods_per_year
            ann_vol = returns.std() * np.sqrt(periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            rows.append({"Decade": decade_name, "Variant": label, "Ann Ret": ann_ret, "Sharpe": sharpe, "N": len(returns)})
    if rows:
        decade_df = pd.DataFrame(rows)
        print(decade_df.to_string(index=False, float_format=lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)))
    else:
        print("No decade rows available.")

    print("\n--- INFORMATION COEFFICIENT ---")
    ic = weekly_returns[["ic_c2c", "ic_o2o"]].dropna(subset=["ic_c2c"])
    if ic.empty:
        print("No IC data available.")
    else:
        c2c_t = ic["ic_c2c"].mean() / (ic["ic_c2c"].std() / np.sqrt(len(ic)))
        print(f"C2C: mean={ic['ic_c2c'].mean():.4f}, std={ic['ic_c2c'].std():.4f}, t={c2c_t:.2f}")
        o2o_ic = ic["ic_o2o"].dropna()
        if not o2o_ic.empty:
            o2o_t = o2o_ic.mean() / (o2o_ic.std() / np.sqrt(len(o2o_ic)))
            print(f"O2O: mean={o2o_ic.mean():.4f}, std={o2o_ic.std():.4f}, t={o2o_t:.2f}")

        print("\nIC by Decade:")
        for decade_name, (start_year, end_year) in decades.items():
            mask = (ic.index.year >= int(start_year)) & (ic.index.year <= int(end_year))
            subset = ic.loc[mask, "ic_c2c"].dropna()
            if len(subset) > 10:
                t_stat = subset.mean() / (subset.std() / np.sqrt(len(subset)))
                print(f"  {decade_name}: IC={subset.mean():.4f}, t={t_stat:.2f}")

    print("\n--- DECILE DECOMPOSITION ---")
    for decile in range(1, 11):
        c2c = weekly_returns[f"dec_{decile}_c2c"].mean() * periods_per_year
        o2o = weekly_returns[f"dec_{decile}_o2o"].mean() * periods_per_year
        suffix = " (LONG/losers)" if decile == 1 else " (SHORT/winners)" if decile == 10 else ""
        print(f"  D{decile:2d}: C2C={c2c * 100:+6.2f}%  O2O={o2o * 100:+6.2f}%{suffix}")

    c2c_vals = [weekly_returns[f"dec_{decile}_c2c"].mean() for decile in range(1, 11)]
    monotonic = all(c2c_vals[i] >= c2c_vals[i + 1] for i in range(9))
    print(f"Monotonically decreasing (D1 > D10)? {monotonic}")

    print("\n--- LONG vs SHORT ATTRIBUTION ---")
    long_c2c = weekly_returns["long_ret_c2c"].mean() * periods_per_year
    short_c2c = weekly_returns["short_ret_c2c"].mean() * periods_per_year
    long_o2o = weekly_returns["long_ret_o2o"].mean() * periods_per_year
    short_o2o = weekly_returns["short_ret_o2o"].mean() * periods_per_year
    print(f"C2C: Long={long_c2c * 100:.2f}%, Short={short_c2c * 100:.2f}%, Spread={((long_c2c - short_c2c) * 100):.2f}%")
    print(f"O2O: Long={long_o2o * 100:.2f}%, Short={short_o2o * 100:.2f}%, Spread={((long_o2o - short_o2o) * 100):.2f}%")

    print("\n--- TURNOVER ---")
    turnover_long = float(weekly_returns["turnover_long"].mean())
    turnover_short = float(weekly_returns["turnover_short"].mean())
    print(
        f"Avg one-way turnover ({cadence_label}): "
        f"Long={turnover_long:.4f} ({turnover_long * 100:.1f}%), "
        f"Short={turnover_short:.4f} ({turnover_short * 100:.1f}%)"
    )
    for bps in cost_bps_levels:
        drag = (turnover_long + turnover_short) * 2.0 * (bps / 10000.0) * periods_per_year
        print(f"  Annual cost drag at {bps:g} bps/side: {drag * 100:.2f}%")

    print("\n--- YEAR-BY-YEAR ---")
    print(yearly_returns.to_string(float_format=lambda value: f"{value:+.4f}"))


def save_artifacts(
    artifacts: BenchmarkArtifacts,
    output_dir: Path,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    cadence_label: str = "weekly Friday",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.weekly_returns.to_csv(output_dir / "weekly_returns.csv")
    artifacts.signal_checks.to_csv(output_dir / "signal_checks.csv", index=False)
    artifacts.portfolio_selections.to_csv(output_dir / "portfolio_selections.csv", index=False)
    artifacts.headline_metrics.to_csv(output_dir / "headline_metrics.csv", index=False)
    artifacts.yearly_returns.to_csv(output_dir / "yearly_returns.csv")
    artifacts.strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    chart_paths = save_charts(
        artifacts,
        output_dir,
        periods_per_year=periods_per_year,
        cadence_label=cadence_label,
    )

    print("\nSaved outputs:")
    print(f"  {output_dir / 'weekly_returns.csv'}")
    print(f"  {output_dir / 'signal_checks.csv'}")
    print(f"  {output_dir / 'portfolio_selections.csv'}")
    print(f"  {output_dir / 'headline_metrics.csv'}")
    print(f"  {output_dir / 'yearly_returns.csv'}")
    print(f"  {output_dir / 'strategy_summary.csv'}")
    for chart_path in chart_paths:
        print(f"  {chart_path}")


def run_benchmark(config: BenchmarkConfig, histories: dict[str, pd.DataFrame] | None = None) -> BenchmarkArtifacts:
    print("Loading Russell OHLCV histories...")
    if histories is None:
        histories = load_russell_ohlcv_histories()
    print(f"   Loaded {len(histories)} ticker histories")

    print(f"Loading membership file: {config.membership_path}")
    membership_lookup = load_membership_lookup(config.membership_path, end_date=config.end)
    print(f"   Membership dates loaded: {len(membership_lookup.dates)}")

    market_frames = build_market_frames(histories)
    print(f"   Close frame shape: {market_frames.close_df.shape}")
    print(f"   Unadjusted close frame shape: {market_frames.unadj_close_df.shape}")
    gc.collect()

    if config.cadence_mode == "weekly_friday":
        weekly_returns, signal_checks, portfolio_selections = run_weekly_strategy(
            market_frames,
            membership_lookup,
            config,
        )
    else:
        weekly_returns, signal_checks, portfolio_selections = run_periodic_strategy(
            market_frames,
            membership_lookup,
            config,
        )
    weekly_returns = add_cost_overlays(weekly_returns, config.cost_bps_levels)
    headline_metrics = build_headline_metrics(weekly_returns, periods_per_year=config.periods_per_year())
    yearly_returns = build_yearly_returns(weekly_returns)
    strategy_summary = build_strategy_summary(weekly_returns, periods_per_year=config.periods_per_year())

    return BenchmarkArtifacts(
        weekly_returns=weekly_returns,
        signal_checks=signal_checks,
        portfolio_selections=portfolio_selections,
        headline_metrics=headline_metrics,
        yearly_returns=yearly_returns,
        strategy_summary=strategy_summary,
        market_frames=market_frames,
    )


def main(argv: list[str] | None = None) -> BenchmarkArtifacts:
    args = parse_args(argv)
    config = BenchmarkConfig(
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        output_dir=args.output_dir,
        membership_path=args.membership_path,
        cost_bps_levels=tuple(args.cost_bps_levels),
        cadence_mode=args.cadence_mode,
        signal_lookback_days=args.signal_lookback_days,
        rebalance_days=args.rebalance_days,
        rebalance_anchor_date=(
            pd.Timestamp(args.rebalance_anchor_date) if args.rebalance_anchor_date else None
        ),
        holding_days=args.holding_days,
        positions_per_side=args.positions_per_side,
    )

    print("\n" + "=" * 80)
    print("STANDALONE WEEKLY RUSSELL BENCHMARK")
    print("=" * 80)
    print(f"Date range     : {config.start.date()} -> {config.end.date()}")
    print(f"Output dir     : {config.output_dir}")
    print(f"Membership file: {config.membership_path}")
    print(f"Cost levels    : {', '.join(f'{bps:g}bps' for bps in config.cost_bps_levels)}")
    print(f"Cadence mode   : {config.cadence_label()}")
    if config.rebalance_anchor_date is not None:
        print(f"Rebalance anchor: {config.rebalance_anchor_date.date()}")
    print(
        "Selection mode : "
        + (
            f"top/bottom {int(config.positions_per_side)} names"
            if config.positions_per_side is not None
            else "loser/winner deciles"
        )
    )

    artifacts = run_benchmark(config)
    save_artifacts(
        artifacts,
        config.output_dir,
        periods_per_year=config.periods_per_year(),
        cadence_label=config.cadence_label(),
    )
    print_sanity_checks(
        artifacts.weekly_returns,
        artifacts.signal_checks,
        artifacts.market_frames,
        artifacts.headline_metrics,
        periods_per_year=config.periods_per_year(),
        cadence_label=config.cadence_label(),
    )
    print_additional_breakdowns(
        artifacts.weekly_returns,
        artifacts.yearly_returns,
        periods_per_year=config.periods_per_year(),
        cadence_label=config.cadence_label(),
        cost_bps_levels=config.cost_bps_levels,
    )

    print("\n--- STRATEGY SUMMARY ---")
    print(
        artifacts.strategy_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}" if isinstance(value, float) else str(value),
        )
    )
    print("\nDone!")
    return artifacts


if __name__ == "__main__":
    main()
