"""Policy helpers for deterministic portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


DEFAULT_REGIME_REVIEW_INTERVALS = {
    "normal": 20,
    "grind_up": 20,
    "slow_bleed": 20,
    "squeeze": 5,
    "crisis": 5,
    "unknown": 20,
}


@dataclass(frozen=True)
class PositionState:
    ticker: str
    side: str
    weight: float
    entry_date: pd.Timestamp
    entry_signal_date: pd.Timestamp | None = None
    entry_price: float | None = None
    entry_score: float | None = None
    last_price: float | None = None


@dataclass(frozen=True)
class ReviewDecision:
    due: bool
    reason: str
    review_date: pd.Timestamp | None
    signal_date: pd.Timestamp | None
    regime: str
    candidate_longs: Tuple[str, ...] = ()
    candidate_shorts: Tuple[str, ...] = ()
    target_longs: Tuple[str, ...] = ()
    target_shorts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TradeDecision:
    executed: bool
    target_weights: Dict[str, float]
    turnover_total: float
    turnover_long: float
    turnover_short: float
    cost_bps: float
    long_entries: int
    long_exits: int
    short_entries: int
    short_exits: int
    replacements: int


@dataclass(frozen=True)
class ReviewPolicy:
    mode: str = "static"
    static_review_days: int = 10
    regime_intervals: Mapping[str, int] = field(default_factory=lambda: DEFAULT_REGIME_REVIEW_INTERVALS.copy())
    early_review_regimes: Tuple[str, ...] = ("squeeze", "crisis")
    scheduled_signal_dates: frozenset[pd.Timestamp] = frozenset()

    def interval_for_regime(self, regime: str) -> int:
        return int(self.regime_intervals.get(regime, self.regime_intervals.get("unknown", self.static_review_days)))

    def should_review(
        self,
        signal_date: pd.Timestamp | None,
        current_regime: str,
        previous_regime: str | None,
        days_since_last_review: int | None,
    ) -> Tuple[bool, str]:
        if signal_date is None:
            return False, ""

        signal_ts = pd.Timestamp(signal_date)
        if self.mode == "static":
            if signal_ts in self.scheduled_signal_dates:
                return True, "scheduled_interval"
            return False, ""

        if days_since_last_review is None:
            return True, "initial_review"

        if (
            current_regime in self.early_review_regimes
            and previous_regime != current_regime
            and previous_regime not in self.early_review_regimes
        ):
            return True, "regime_deterioration"

        if days_since_last_review >= self.interval_for_regime(current_regime):
            return True, "scheduled_interval"
        return False, ""


@dataclass(frozen=True)
class HysteresisPolicy:
    mode: str = "none"
    top_n: int = 30
    bottom_n: int = 30
    long_retain_buffer: int = 15
    short_retain_buffer: int = 10
    replacement_gap: int = 5
    min_weight_delta: float | None = None

    def target_book(
        self,
        current_longs: Iterable[str],
        current_shorts: Iterable[str],
        candidate_longs: Sequence[str],
        candidate_shorts: Sequence[str],
        long_rank_map: Mapping[str, int],
        short_rank_map: Mapping[str, int],
        target_long_n: int | None = None,
        target_short_n: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        long_n = self.top_n if target_long_n is None else int(target_long_n)
        short_n = self.bottom_n if target_short_n is None else int(target_short_n)
        mode = self.mode
        if mode == "none":
            return list(candidate_longs[:long_n]), list(candidate_shorts[:short_n])

        use_long_band = mode in {"rank_band_symmetric", "rank_band_asymmetric", "combined"}
        use_short_band = mode in {"rank_band_symmetric", "rank_band_asymmetric", "combined"}
        use_replacement_gap = mode in {"replacement_gap", "combined"}

        long_buffer = self.long_retain_buffer if mode in {"rank_band_asymmetric", "combined"} else max(self.long_retain_buffer, self.short_retain_buffer)
        short_buffer = self.short_retain_buffer if mode in {"rank_band_asymmetric", "combined"} else max(self.long_retain_buffer, self.short_retain_buffer)

        target_longs = _target_side(
            current_names=current_longs,
            candidate_names=candidate_longs,
            rank_map=long_rank_map,
            target_n=long_n,
            retain_buffer=long_buffer,
            replacement_gap=self.replacement_gap,
            use_rank_band=use_long_band,
            use_replacement_gap=use_replacement_gap,
        )
        target_shorts = _target_side(
            current_names=current_shorts,
            candidate_names=candidate_shorts,
            rank_map=short_rank_map,
            target_n=short_n,
            retain_buffer=short_buffer,
            replacement_gap=self.replacement_gap,
            use_rank_band=use_short_band,
            use_replacement_gap=use_replacement_gap,
        )
        return target_longs, target_shorts


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    review_policy: ReviewPolicy
    hysteresis_policy: HysteresisPolicy
    top_n: int = 30
    bottom_n: int = 30
    weighting_mode: str = "equal"
    book_sizing_mode: str = "fixed"

    def target_book_sizes(self, candidate_universe_size: int) -> Tuple[int, int]:
        universe_size = max(0, int(candidate_universe_size))
        if self.book_sizing_mode == "fixed":
            return max(0, int(self.top_n)), max(0, int(self.bottom_n))
        if self.book_sizing_mode == "decile":
            if universe_size <= 0:
                return 0, 0
            # Match the weekly mean-reversion decile bucketing:
            # long decile gets the highest-ranked ceil(N/10) names,
            # short decile gets the lowest-ranked floor(N/10) names.
            long_n = (universe_size + 9) // 10
            short_n = universe_size - ((9 * universe_size + 9) // 10)
            return long_n, short_n
        raise ValueError(f"Unsupported book sizing mode: {self.book_sizing_mode}")

    def describe_book_sizing(self) -> str:
        if self.book_sizing_mode == "fixed":
            return f"fixed counts ({self.top_n} long / {self.bottom_n} short)"
        if self.book_sizing_mode == "decile":
            return "dynamic deciles (top 10% long / bottom 10% short)"
        return f"unknown ({self.book_sizing_mode})"

    def variant_label(self) -> str:
        label = (
            f"review_{self.review_policy.mode}_{self.review_policy.static_review_days}d"
            f"__hyst_{self.hysteresis_policy.mode}"
        )
        if self.weighting_mode != "equal":
            label += f"__weights_{self.weighting_mode}"
        if self.book_sizing_mode != "fixed":
            label += f"__book_{self.book_sizing_mode}"
        return label


def build_equal_notional_weights(longs: Sequence[str], shorts: Sequence[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if longs:
        long_weight = 1.0 / len(longs)
        for ticker in longs:
            weights[ticker] = long_weight
    if shorts:
        short_weight = 1.0 / len(shorts)
        for ticker in shorts:
            weights[ticker] = -short_weight
    return weights


def build_rank_tilted_weights(longs: Sequence[str], shorts: Sequence[str]) -> Dict[str, float]:
    """Weight stronger-ranked names more heavily while staying dollar-neutral by side."""
    weights: Dict[str, float] = {}

    if longs:
        long_strengths = list(range(len(longs), 0, -1))
        long_total = float(sum(long_strengths))
        for ticker, strength in zip(longs, long_strengths):
            weights[ticker] = strength / long_total

    if shorts:
        short_strengths = list(range(len(shorts), 0, -1))
        short_total = float(sum(short_strengths))
        for ticker, strength in zip(shorts, short_strengths):
            weights[ticker] = -(strength / short_total)

    return weights


def build_target_weights(
    longs: Sequence[str],
    shorts: Sequence[str],
    weighting_mode: str = "equal",
) -> Dict[str, float]:
    if weighting_mode == "equal":
        return build_equal_notional_weights(longs, shorts)
    if weighting_mode == "rank":
        return build_rank_tilted_weights(longs, shorts)
    raise ValueError(f"Unsupported weighting mode: {weighting_mode}")


def apply_min_weight_delta_filter(
    old_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
    min_weight_delta: float | None,
) -> Dict[str, float]:
    if min_weight_delta is None or min_weight_delta <= 0:
        return {ticker: float(weight) for ticker, weight in new_weights.items() if abs(weight) > 1e-12}

    filtered: Dict[str, float] = {}
    tickers = set(old_weights.keys()) | set(new_weights.keys())
    for ticker in tickers:
        old_weight = float(old_weights.get(ticker, 0.0))
        new_weight = float(new_weights.get(ticker, 0.0))

        # Never let the small-resize filter block a true exit or side flip.
        # Otherwise a tiny retained position can outlive membership/tradability
        # and poison later PnL if the symbol becomes stale or delisted.
        if abs(new_weight) <= 1e-12:
            continue
        if old_weight * new_weight < 0:
            filtered[ticker] = new_weight
            continue

        if abs(new_weight - old_weight) < min_weight_delta:
            if abs(old_weight) > 1e-12:
                filtered[ticker] = old_weight
        elif abs(new_weight) > 1e-12:
            filtered[ticker] = new_weight
    return filtered


def compute_trade_decision(
    old_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    cost_rate: float,
    min_weight_delta: float | None = None,
) -> TradeDecision:
    filtered_weights = apply_min_weight_delta_filter(old_weights, target_weights, min_weight_delta)

    turnover_total = 0.0
    turnover_long = 0.0
    turnover_short = 0.0
    tickers = set(old_weights.keys()) | set(filtered_weights.keys())
    for ticker in tickers:
        old_weight = float(old_weights.get(ticker, 0.0))
        new_weight = float(filtered_weights.get(ticker, 0.0))
        delta = abs(new_weight - old_weight)
        turnover_total += delta
        if new_weight >= 0 and old_weight >= 0:
            turnover_long += delta
        elif new_weight <= 0 and old_weight <= 0:
            turnover_short += delta
        elif old_weight > 0 and new_weight < 0:
            turnover_long += old_weight
            turnover_short += abs(new_weight)
        else:
            turnover_short += abs(old_weight)
            turnover_long += new_weight

    old_longs = {ticker for ticker, weight in old_weights.items() if weight > 0}
    old_shorts = {ticker for ticker, weight in old_weights.items() if weight < 0}
    new_longs = {ticker for ticker, weight in filtered_weights.items() if weight > 0}
    new_shorts = {ticker for ticker, weight in filtered_weights.items() if weight < 0}

    long_entries = len(new_longs - old_longs)
    long_exits = len(old_longs - new_longs)
    short_entries = len(new_shorts - old_shorts)
    short_exits = len(old_shorts - new_shorts)
    replacements = min(long_entries, long_exits) + min(short_entries, short_exits)

    executed = turnover_total > 1e-12
    cost_bps = cost_rate * turnover_total * 10000 if executed else 0.0
    return TradeDecision(
        executed=executed,
        target_weights=dict(filtered_weights),
        turnover_total=turnover_total,
        turnover_long=turnover_long,
        turnover_short=turnover_short,
        cost_bps=cost_bps,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        replacements=replacements,
    )


def _ordered_unique(names: Iterable[str], rank_map: Mapping[str, int]) -> List[str]:
    seen = set()
    ordered = []
    for name in sorted(list(names), key=lambda item: rank_map.get(item, 10**9)):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _target_side(
    current_names: Iterable[str],
    candidate_names: Sequence[str],
    rank_map: Mapping[str, int],
    target_n: int,
    retain_buffer: int,
    replacement_gap: int,
    use_rank_band: bool,
    use_replacement_gap: bool,
) -> List[str]:
    candidates = list(candidate_names[:target_n])
    if target_n <= 0:
        return []
    if not candidates:
        return []

    current = _ordered_unique(current_names, rank_map)
    if not current:
        return candidates

    if use_rank_band:
        retain_cutoff = target_n + retain_buffer
        target = [
            ticker for ticker in current
            if rank_map.get(ticker, 10**9) <= retain_cutoff
        ]
    else:
        target = current[:]

    target = _ordered_unique(target, rank_map)
    for candidate in candidates:
        if candidate not in target and len(target) < target_n:
            target.append(candidate)

    if len(target) > target_n:
        target = _ordered_unique(target, rank_map)[:target_n]

    if use_replacement_gap and target:
        challengers = [candidate for candidate in candidates if candidate not in target]
        for challenger in challengers:
            challenger_rank = rank_map.get(challenger, 10**9)
            worst = max(target, key=lambda ticker: rank_map.get(ticker, 10**9))
            worst_rank = rank_map.get(worst, 10**9)
            required_gap = replacement_gap
            if use_rank_band and worst_rank > target_n + retain_buffer:
                required_gap = 0
            if worst_rank - challenger_rank >= required_gap:
                target.remove(worst)
                target.append(challenger)
                target = _ordered_unique(target, rank_map)
            else:
                break

    target = _ordered_unique(target, rank_map)
    for candidate in candidates:
        if candidate not in target and len(target) < target_n:
            target.append(candidate)
    return target[:target_n]
