from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from indicator_feature_schema import (
    CANONICAL_INDICATOR_FEATURES,
    extract_canonical_indicator_features,
)


LOSS_MEMORY_FEATURE_COLUMNS = [
    "direction_score",
    "magnitude_score",
] + list(CANONICAL_INDICATOR_FEATURES)


@dataclass
class MemoryCase:
    ticker: str
    side: str
    entry_date: str
    exit_date: str
    regime: str
    position_return: float
    weighted_contribution: float
    direction_score: float
    magnitude_score: float
    rationale: str
    features: Dict[str, float]


class LossMemoryStore:
    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        top_k: int = 3,
        max_targets_per_side: int = 4,
    ):
        self.feature_columns = feature_columns or list(LOSS_MEMORY_FEATURE_COLUMNS)
        self.top_k = max(1, int(top_k))
        self.max_targets_per_side = max(1, int(max_targets_per_side))
        self._cases_by_side: Dict[str, List[MemoryCase]] = {"LONG": [], "SHORT": []}
        self._finalized_entry_dates: set[str] = set()

    @property
    def total_cases(self) -> int:
        return sum(len(cases) for cases in self._cases_by_side.values())

    def finalize_portfolio(
        self,
        entry_portfolio: Optional[Dict[str, Any]],
        exit_date: pd.Timestamp,
        composite_dfs: Dict[str, pd.DataFrame],
    ) -> int:
        if not entry_portfolio:
            return 0

        entry_date_str = str(entry_portfolio.get("rebalance_date", ""))
        if not entry_date_str or entry_date_str in self._finalized_entry_dates:
            return 0

        entry_date = pd.Timestamp(entry_date_str)
        exit_date = pd.Timestamp(exit_date)
        if exit_date <= entry_date:
            return 0

        added = 0
        for side_key, side in (("long_basket", "LONG"), ("short_basket", "SHORT")):
            for position in entry_portfolio.get(side_key, []):
                memory_case = self._build_case(
                    position=position,
                    side=side,
                    regime=str(entry_portfolio.get("regime", "unknown")),
                    entry_date=entry_date,
                    exit_date=exit_date,
                    composite_dfs=composite_dfs,
                )
                if memory_case is None:
                    continue
                self._cases_by_side[side].append(memory_case)
                added += 1

        self._finalized_entry_dates.add(entry_date_str)
        return added

    def build_prompt_context(
        self,
        universe: Dict[str, Dict[str, Any]],
        previous_portfolio: Optional[Dict[str, Any]],
        regime: str,
        as_of_date: pd.Timestamp,
        top_n_magnitude: int = 60,
    ) -> Dict[str, Any]:
        targets = self._select_targets(
            universe=universe,
            previous_portfolio=previous_portfolio,
            top_n_magnitude=top_n_magnitude,
        )
        if not targets:
            return {"text": "", "target_count": 0, "case_count": 0, "hits": []}

        grouped: Dict[str, List[str]] = {"LONG": [], "SHORT": []}
        hits: List[Dict[str, Any]] = []
        total_cases = 0

        for target in targets:
            cases = self._top_loss_cases_for_target(
                signal=target["signal"],
                side=target["side"],
                regime=regime,
                as_of_date=pd.Timestamp(as_of_date),
            )
            if not cases:
                continue

            total_cases += len(cases)
            avg_loss = float(np.mean([case.position_return for case in cases]))
            examples = "; ".join(
                f"{case.ticker} {case.entry_date}→{case.exit_date} ({case.position_return:.1%}, {case.regime})"
                for case in cases[: min(2, len(cases))]
            )
            grouped[target["side"]].append(
                f"- {target['ticker']} ({target['source']}): {len(cases)} similar prior losses, "
                f"avg {avg_loss:.1%}; examples: {examples}"
            )
            hits.append(
                {
                    "ticker": target["ticker"],
                    "side": target["side"],
                    "source": target["source"],
                    "n_cases": len(cases),
                    "avg_loss": avg_loss,
                }
            )

        if not hits:
            return {"text": "", "target_count": 0, "case_count": 0, "hits": []}

        lines: List[str] = [
            "Historical analogs are shown below as cautionary evidence, not hard exclusions."
        ]
        if grouped["LONG"]:
            lines.append("LONG analog warnings:")
            lines.extend(grouped["LONG"])
        if grouped["SHORT"]:
            lines.append("SHORT analog warnings:")
            lines.extend(grouped["SHORT"])

        return {
            "text": "\n".join(lines),
            "target_count": len(hits),
            "case_count": total_cases,
            "hits": hits,
        }

    def _build_case(
        self,
        position: Dict[str, Any],
        side: str,
        regime: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        composite_dfs: Dict[str, pd.DataFrame],
    ) -> Optional[MemoryCase]:
        ticker = str(position.get("ticker", "")).upper()
        if not ticker:
            return None

        df = composite_dfs.get(ticker)
        if df is None or df.empty:
            return None

        entry_row = self._row_for_date(df, entry_date)
        exit_row = self._row_for_date(df, exit_date)
        if entry_row is None or exit_row is None:
            return None

        entry_price = self._safe_float(entry_row.get("adj_close"))
        exit_price = self._safe_float(exit_row.get("adj_close"))
        if entry_price is None or exit_price is None or entry_price <= 0:
            return None

        raw_return = (exit_price / entry_price) - 1.0
        position_return = raw_return if side == "LONG" else -raw_return
        weight = self._safe_float(position.get("weight_exec"))
        if weight is None:
            weight = self._safe_float(position.get("weight"))
        if weight is None:
            weight = 0.0

        return MemoryCase(
            ticker=ticker,
            side=side,
            entry_date=entry_date.strftime("%Y-%m-%d"),
            exit_date=exit_date.strftime("%Y-%m-%d"),
            regime=regime,
            position_return=position_return,
            weighted_contribution=weight * position_return,
            direction_score=self._safe_float(entry_row.get("direction_score")) or 0.0,
            magnitude_score=self._safe_float(entry_row.get("magnitude_score")) or 0.0,
            rationale=str(position.get("rationale", "")),
            features=self._features_from_row(entry_row),
        )

    def _top_loss_cases_for_target(
        self,
        signal: Dict[str, Any],
        side: str,
        regime: str,
        as_of_date: pd.Timestamp,
    ) -> List[MemoryCase]:
        eligible_cases = [
            case
            for case in self._cases_by_side.get(side, [])
            if case.position_return < 0 and pd.Timestamp(case.exit_date) <= as_of_date
        ]
        if not eligible_cases:
            return []

        query_vector = self._features_from_signal(signal)
        matrix = np.array(
            [
                [case.features.get(column, np.nan) for column in self.feature_columns]
                for case in eligible_cases
            ],
            dtype=float,
        )
        query = np.array(
            [query_vector.get(column, np.nan) for column in self.feature_columns],
            dtype=float,
        )

        col_means = np.zeros(matrix.shape[1], dtype=float)
        col_stds = np.ones(matrix.shape[1], dtype=float)
        for idx in range(matrix.shape[1]):
            valid = matrix[:, idx]
            valid = valid[~np.isnan(valid)]
            if valid.size == 0:
                continue
            col_means[idx] = float(valid.mean())
            std = float(valid.std())
            if std >= 1e-8:
                col_stds[idx] = std

        matrix = np.where(np.isnan(matrix), col_means, matrix)
        query = np.where(np.isnan(query), col_means, query)

        matrix = (matrix - col_means) / col_stds
        query = (query - col_means) / col_stds

        denom = np.linalg.norm(matrix, axis=1) * max(np.linalg.norm(query), 1e-8)
        similarity = np.divide(matrix @ query, np.where(denom < 1e-8, 1.0, denom))
        regime_bonus = np.array(
            [0.05 if case.regime == regime else 0.0 for case in eligible_cases],
            dtype=float,
        )
        scores = similarity + regime_bonus
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [eligible_cases[idx] for idx in top_idx]

    def _select_targets(
        self,
        universe: Dict[str, Dict[str, Any]],
        previous_portfolio: Optional[Dict[str, Any]],
        top_n_magnitude: int,
    ) -> List[Dict[str, Any]]:
        direction_threshold = 5.0
        selected: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        def add_target(ticker: str, side: str, source: str):
            if ticker not in universe:
                return
            key = (ticker, side)
            if key in seen_keys:
                return
            seen_keys.add(key)
            selected.append(
                {
                    "ticker": ticker,
                    "side": side,
                    "source": source,
                    "signal": universe[ticker],
                }
            )

        if previous_portfolio:
            for side_key, side in (("long_basket", "LONG"), ("short_basket", "SHORT")):
                sorted_positions = sorted(
                    previous_portfolio.get(side_key, []),
                    key=lambda position: float(
                        position.get("weight_exec", position.get("weight", 0.0))
                    ),
                    reverse=True,
                )
                for position in sorted_positions[: self.max_targets_per_side]:
                    add_target(str(position.get("ticker", "")).upper(), side, "kept_or_reviewed")

        pools = {"LONG": [], "SHORT": []}
        for ticker, signal in universe.items():
            direction_score = self._safe_float(signal.get("direction_score")) or 0.0
            if direction_score > direction_threshold:
                pools["LONG"].append((ticker, signal))
            elif direction_score < -direction_threshold:
                pools["SHORT"].append((ticker, signal))

        for side in ("LONG", "SHORT"):
            ranked = sorted(
                pools[side],
                key=lambda item: float(item[1].get("magnitude_score", 0.0)),
                reverse=True,
            )[:top_n_magnitude]
            current_count = sum(1 for item in selected if item["side"] == side)
            needed = max(self.max_targets_per_side - current_count, 0)
            for ticker, _ in ranked[:needed]:
                add_target(ticker, side, "top_candidate")

        return selected

    def _features_from_row(self, row: pd.Series) -> Dict[str, float]:
        features: Dict[str, float] = extract_canonical_indicator_features(row, CANONICAL_INDICATOR_FEATURES)
        for column in ("direction_score", "magnitude_score"):
            value = self._safe_float(row.get(column))
            if value is not None:
                features[column] = value
        return features

    def _features_from_signal(self, signal: Dict[str, Any]) -> Dict[str, float]:
        indicators = signal.get("indicators", {})
        features: Dict[str, float] = {}
        for column in self.feature_columns:
            source = signal if column in {"direction_score", "magnitude_score"} else indicators
            value = self._safe_float(source.get(column))
            if value is not None:
                features[column] = value
        return features

    @staticmethod
    def _row_for_date(df: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
        row = df[df["date"] == pd.Timestamp(date)]
        if row.empty:
            return None
        return row.iloc[0]

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
