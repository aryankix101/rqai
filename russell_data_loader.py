"""
Russell dataset loaders.

Supports both the older close-only wide export and the newer wide OHLCV export.
The primary source is parquet when parquet support is available; the legacy
close-only path still falls back to streaming workbook XML directly from Excel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile
import re

import numpy as np
import pandas as pd

from data_source_config import RUSSELL_EXCEL_PATH, RUSSELL_OHLCV_PATH, RUSSELL_PARQUET_PATH


SUFFIX_RE = re.compile(r"^(?P<ticker>[A-Z]+)(?:-(?P<suffix>\d{6}))?$")
WIDE_OHLCV_FIELD_NAMES = frozenset(
    {"Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend"}
)
OHLCV_HISTORY_COLUMNS = [
    "date",
    "ticker",
    "asset_id",
    "source_column",
    "is_historical_suffix",
    "collision_count",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
]


@dataclass(frozen=True)
class RussellAssetMetadata:
    ticker: str
    asset_id: str
    source_column: str
    is_historical_suffix: bool
    suffix_yyyymm: Optional[int]


def parse_source_column(source_column: str) -> RussellAssetMetadata:
    raw = str(source_column).strip().upper()
    match = SUFFIX_RE.match(raw)
    if not match:
        ticker = raw.split("-", 1)[0]
        return RussellAssetMetadata(
            ticker=ticker,
            asset_id=raw,
            source_column=raw,
            is_historical_suffix="-" in raw,
            suffix_yyyymm=None,
        )

    suffix = match.group("suffix")
    return RussellAssetMetadata(
        ticker=match.group("ticker"),
        asset_id=raw,
        source_column=raw,
        is_historical_suffix=suffix is not None,
        suffix_yyyymm=int(suffix) if suffix is not None else None,
    )


def parquet_support_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def _excel_serial_to_timestamp(value: object) -> pd.Timestamp:
    serial = float(value)
    return pd.Timestamp(datetime(1899, 12, 30) + timedelta(days=serial))


def _split_cell_ref(cell_ref: str) -> tuple[str, str]:
    letters = []
    digits = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch)
        elif ch.isdigit():
            digits.append(ch)
    return "".join(letters), "".join(digits)


def _column_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def _load_excel_wide_frame(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Russell Excel file not found: {excel_path}")

    with ZipFile(excel_path) as workbook, workbook.open("xl/worksheets/sheet2.xml") as fh:
        headers: Optional[list[str]] = None
        price_rows: list[np.ndarray] = []
        dates: list[pd.Timestamp] = []
        current_cell_ref: Optional[str] = None
        current_row_cells: Dict[int, str] = {}
        current_row_number: Optional[int] = None

        for event, elem in iterparse(fh, events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1]

            if event == "start":
                if tag == "row":
                    current_row_cells = {}
                    current_row_number = int(elem.attrib.get("r", "0"))
                elif tag == "c":
                    current_cell_ref = elem.attrib.get("r")
                continue

            if tag in {"v", "t"} and current_cell_ref is not None:
                letters, _ = _split_cell_ref(current_cell_ref)
                current_row_cells[_column_letters_to_index(letters)] = elem.text or ""
            elif tag == "c":
                current_cell_ref = None
            elif tag == "row":
                if current_row_number == 1:
                    max_col = max(current_row_cells.keys()) if current_row_cells else 0
                    headers = [current_row_cells.get(idx, "") for idx in range(max_col + 1)]
                elif headers is not None:
                    row = np.full(len(headers) - 1, np.nan, dtype=np.float32)
                    raw_date = current_row_cells.get(0)
                    if raw_date is None or raw_date == "":
                        elem.clear()
                        continue

                    dates.append(_excel_serial_to_timestamp(raw_date))
                    for col_idx, value in current_row_cells.items():
                        if col_idx == 0 or col_idx >= len(headers):
                            continue
                        if value in ("", None):
                            continue
                        try:
                            row[col_idx - 1] = float(value)
                        except ValueError:
                            row[col_idx - 1] = np.nan
                    price_rows.append(row)

                elem.clear()

    if headers is None:
        raise ValueError("Failed to read Russell workbook headers")

    wide = pd.DataFrame(np.vstack(price_rows), columns=headers[1:])
    wide.insert(0, "Date", pd.to_datetime(dates))
    return wide


def _extract_date_series(df: pd.DataFrame) -> pd.Series:
    if "Date" in df.columns:
        raw_dates = df["Date"]
    elif "date" in df.columns:
        raw_dates = df["date"]
    else:
        raw_dates = pd.Series(df.index, index=df.index)

    raw_arr = np.asarray(raw_dates)
    if np.issubdtype(raw_arr.dtype, np.number):
        return raw_dates.map(_excel_serial_to_timestamp).reset_index(drop=True)
    return pd.to_datetime(raw_dates).reset_index(drop=True)


def _read_russell_ohlcv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Russell OHLCV file not found at {path}. "
            "Drop the OHLCV dataset there or update RUSSELL_OHLCV_PATH."
        )

    if path.suffix == ".parquet":
        if not parquet_support_available():
            raise ImportError("Parquet support unavailable; install pyarrow or fastparquet")
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported Russell OHLCV file type: {path.suffix}")


def _looks_like_wide_ohlcv_frame(df: pd.DataFrame) -> bool:
    if not len(df.columns):
        return False

    match_count = 0
    for column in df.columns:
        name = str(column)
        if "_" not in name:
            continue
        _, field = name.rsplit("_", 1)
        if field in WIDE_OHLCV_FIELD_NAMES:
            match_count += 1
            if match_count >= 5:
                return True
    return False


def _wide_ohlcv_fields_by_asset(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    fields_by_asset: dict[str, dict[str, str]] = {}
    for column in df.columns:
        name = str(column)
        if "_" not in name:
            continue
        asset_id, field = name.rsplit("_", 1)
        if field not in WIDE_OHLCV_FIELD_NAMES:
            continue
        fields_by_asset.setdefault(asset_id, {})[field] = name
    return fields_by_asset


def _wide_value_matrix(df: pd.DataFrame, columns: list[Optional[str]], row_count: int) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for column in columns:
        if column is None:
            arrays.append(np.full(row_count, np.nan, dtype=np.float64))
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        arrays.append(series.to_numpy(dtype=np.float64, copy=False))
    if not arrays:
        return np.empty((row_count, 0), dtype=np.float64)
    return np.column_stack(arrays)


def _collapse_wide_ohlcv_histories(
    df_wide: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
) -> dict[str, pd.DataFrame]:
    fields_by_asset = _wide_ohlcv_fields_by_asset(df_wide)
    if not fields_by_asset:
        return {}

    ticker_filter = {ticker.upper().strip() for ticker in tickers} if tickers else None
    ticker_groups: Dict[str, list[tuple[RussellAssetMetadata, dict[str, str]]]] = {}
    for asset_id, field_map in fields_by_asset.items():
        if "Close" not in field_map:
            continue
        meta = parse_source_column(asset_id)
        if ticker_filter is not None and meta.ticker not in ticker_filter:
            continue
        ticker_groups.setdefault(meta.ticker, []).append((meta, field_map))

    if not ticker_groups:
        return {}

    date_values = _extract_date_series(df_wide)
    row_count = len(date_values)
    row_idx = np.arange(row_count)
    collapsed: dict[str, pd.DataFrame] = {}

    for ticker, group in ticker_groups.items():
        ordered = sorted(
            group,
            key=lambda item: (
                0 if not item[0].is_historical_suffix else 1,
                -(item[0].suffix_yyyymm or 0),
                item[0].source_column,
            ),
        )
        metas = [meta for meta, _ in ordered]
        field_maps = [field_map for _, field_map in ordered]

        adj_close_matrix = _wide_value_matrix(
            df_wide,
            [field_map.get("Close") for field_map in field_maps],
            row_count,
        )
        valid = np.isfinite(adj_close_matrix)
        has_any = valid.any(axis=1)
        if not has_any.any():
            continue

        first_valid_idx = valid.argmax(axis=1)

        def choose_from_matrix(matrix: np.ndarray) -> np.ndarray:
            chosen = np.full(row_count, np.nan, dtype=np.float64)
            chosen[has_any] = matrix[row_idx[has_any], first_valid_idx[has_any]]
            return chosen

        adj_open = choose_from_matrix(
            _wide_value_matrix(df_wide, [field_map.get("Open") for field_map in field_maps], row_count)
        )
        adj_high = choose_from_matrix(
            _wide_value_matrix(df_wide, [field_map.get("High") for field_map in field_maps], row_count)
        )
        adj_low = choose_from_matrix(
            _wide_value_matrix(df_wide, [field_map.get("Low") for field_map in field_maps], row_count)
        )
        adj_close = choose_from_matrix(adj_close_matrix)
        volume = choose_from_matrix(
            _wide_value_matrix(df_wide, [field_map.get("Volume") for field_map in field_maps], row_count)
        )

        raw_close_matrix = _wide_value_matrix(
            df_wide,
            [field_map.get("Unadjusted Close") for field_map in field_maps],
            row_count,
        )
        raw_close_matrix = np.where(np.isfinite(raw_close_matrix), raw_close_matrix, adj_close_matrix)
        close = choose_from_matrix(raw_close_matrix)

        ordered_source_arr = np.array([meta.source_column for meta in metas], dtype=object)
        ordered_hist_arr = np.array([meta.is_historical_suffix for meta in metas], dtype=bool)
        chosen_sources = np.full(row_count, "", dtype=object)
        chosen_hist_flags = np.full(row_count, False, dtype=bool)
        chosen_collision_counts = valid.sum(axis=1).astype(np.int32)
        chosen_sources[has_any] = ordered_source_arr[first_valid_idx[has_any]]
        chosen_hist_flags[has_any] = ordered_hist_arr[first_valid_idx[has_any]]

        history = pd.DataFrame(
            {
                "date": date_values,
                "ticker": ticker,
                "asset_id": chosen_sources,
                "source_column": chosen_sources,
                "is_historical_suffix": chosen_hist_flags,
                "collision_count": chosen_collision_counts,
                "open": adj_open,
                "high": adj_high,
                "low": adj_low,
                "close": close,
                "volume": volume,
                "adj_open": adj_open,
                "adj_high": adj_high,
                "adj_low": adj_low,
                "adj_close": adj_close,
            }
        )
        history = history[history["asset_id"] != ""].copy()
        history = history[history["adj_close"].notna()].reset_index(drop=True)
        collapsed[ticker] = history

    return collapsed


def load_russell_wide_frame() -> pd.DataFrame:
    if RUSSELL_PARQUET_PATH.exists() and parquet_support_available():
        wide = pd.read_parquet(RUSSELL_PARQUET_PATH)
    else:
        wide = _load_excel_wide_frame(RUSSELL_EXCEL_PATH)

    date_values = _extract_date_series(wide)
    date_cols = [col for col in ["Date", "date"] if col in wide.columns]
    if date_cols:
        wide = wide.drop(columns=date_cols).copy()
    else:
        wide = wide.copy()
    wide.insert(0, "Date", date_values)

    price_cols = [col for col in wide.columns if col != "Date"]
    wide[price_cols] = wide[price_cols].apply(pd.to_numeric, errors="coerce")
    wide = wide.sort_values("Date").reset_index(drop=True)
    return wide


def build_asset_metadata(source_columns: Iterable[str]) -> pd.DataFrame:
    records = [parse_source_column(col).__dict__ for col in source_columns]
    return pd.DataFrame.from_records(records).sort_values(["ticker", "is_historical_suffix", "suffix_yyyymm"])


def collapse_to_base_ticker_histories(df_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    source_columns = [col for col in df_wide.columns if col != "Date"]
    metadata = [parse_source_column(col) for col in source_columns]

    ticker_groups: Dict[str, list[RussellAssetMetadata]] = {}
    for meta in metadata:
        ticker_groups.setdefault(meta.ticker, []).append(meta)

    collapsed: dict[str, pd.DataFrame] = {}
    date_values = pd.to_datetime(df_wide["Date"]).reset_index(drop=True)

    for ticker, group in ticker_groups.items():
        ordered = sorted(
            group,
            key=lambda item: (
                0 if not item.is_historical_suffix else 1,
                -(item.suffix_yyyymm or 0),
                item.source_column,
            ),
        )
        ordered_cols = [item.source_column for item in ordered]
        values = df_wide[ordered_cols].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        has_any = valid.any(axis=1)
        first_valid_idx = valid.argmax(axis=1)

        chosen_prices = np.full(len(df_wide), np.nan, dtype=np.float64)
        chosen_sources = np.full(len(df_wide), "", dtype=object)
        chosen_hist_flags = np.full(len(df_wide), False, dtype=bool)
        chosen_collision_counts = valid.sum(axis=1).astype(np.int32)

        ordered_source_arr = np.array(ordered_cols, dtype=object)
        ordered_hist_arr = np.array([item.is_historical_suffix for item in ordered], dtype=bool)

        row_idx = np.arange(len(df_wide))
        chosen_prices[has_any] = values[row_idx[has_any], first_valid_idx[has_any]]
        chosen_sources[has_any] = ordered_source_arr[first_valid_idx[has_any]]
        chosen_hist_flags[has_any] = ordered_hist_arr[first_valid_idx[has_any]]

        history = pd.DataFrame(
            {
                "date": date_values,
                "ticker": ticker,
                "asset_id": chosen_sources,
                "source_column": chosen_sources,
                "is_historical_suffix": chosen_hist_flags,
                "adj_close": chosen_prices,
                "collision_count": chosen_collision_counts,
            }
        )
        history = history[history["asset_id"] != ""].copy()
        history = history[history["adj_close"].notna()].reset_index(drop=True)
        collapsed[ticker] = history

    return collapsed


def load_russell_long_panel() -> pd.DataFrame:
    wide = load_russell_wide_frame()
    histories = collapse_to_base_ticker_histories(wide)
    if not histories:
        return pd.DataFrame(
            columns=["date", "ticker", "asset_id", "source_column", "is_historical_suffix", "adj_close", "collision_count"]
        )
    return pd.concat(histories.values(), ignore_index=True)


def _normalize_ohlcv_panel(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "date": "date",
        "Ticker": "ticker",
        "ticker": "ticker",
        "asset_id": "asset_id",
        "Asset_ID": "asset_id",
        "source_column": "source_column",
        "Source_Column": "source_column",
        "Open": "open",
        "open": "open",
        "High": "high",
        "high": "high",
        "Low": "low",
        "low": "low",
        "Close": "close",
        "close": "close",
        "Volume": "volume",
        "volume": "volume",
        "Adj_Open": "adj_open",
        "adj_open": "adj_open",
        "Adj_High": "adj_high",
        "adj_high": "adj_high",
        "Adj_Low": "adj_low",
        "adj_low": "adj_low",
        "Adj_Close": "adj_close",
        "adj_close": "adj_close",
        "Adj_Volume": "adj_volume",
        "adj_volume": "adj_volume",
    }
    df = df.rename(columns={col: rename_map[col] for col in df.columns if col in rename_map}).copy()

    if "adj_open" not in df.columns and "open" in df.columns:
        df["adj_open"] = df["open"]
    if "adj_high" not in df.columns and "high" in df.columns:
        df["adj_high"] = df["high"]
    if "adj_low" not in df.columns and "low" in df.columns:
        df["adj_low"] = df["low"]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "volume" not in df.columns and "adj_volume" in df.columns:
        df["volume"] = df["adj_volume"]
    if "open" not in df.columns and "adj_open" in df.columns:
        df["open"] = df["adj_open"]
    if "high" not in df.columns and "adj_high" in df.columns:
        df["high"] = df["adj_high"]
    if "low" not in df.columns and "adj_low" in df.columns:
        df["low"] = df["adj_low"]
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required_cols = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Russell OHLCV dataset missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "asset_id" not in df.columns:
        df["asset_id"] = df["ticker"]
    if "source_column" not in df.columns:
        df["source_column"] = df["asset_id"]

    metadata = df["source_column"].map(parse_source_column)
    df["base_ticker"] = metadata.map(lambda item: item.ticker)
    df["is_historical_suffix"] = metadata.map(lambda item: item.is_historical_suffix)
    df["suffix_yyyymm"] = metadata.map(lambda item: item.suffix_yyyymm or 0)
    return df


def load_russell_ohlcv_panel(path: Path = RUSSELL_OHLCV_PATH) -> pd.DataFrame:
    raw = _read_russell_ohlcv_frame(path)
    if _looks_like_wide_ohlcv_frame(raw):
        histories = _collapse_wide_ohlcv_histories(raw)
        if not histories:
            return pd.DataFrame(columns=OHLCV_HISTORY_COLUMNS)
        return pd.concat(histories.values(), ignore_index=True)
    return _normalize_ohlcv_panel(raw)


def load_russell_ohlcv_histories(
    path: Path = RUSSELL_OHLCV_PATH,
    tickers: Optional[Iterable[str]] = None,
) -> dict[str, pd.DataFrame]:
    raw = _read_russell_ohlcv_frame(path)
    ticker_filter = {ticker.upper().strip() for ticker in tickers} if tickers else None

    if _looks_like_wide_ohlcv_frame(raw):
        return _collapse_wide_ohlcv_histories(raw, tickers=ticker_filter)

    panel = _normalize_ohlcv_panel(raw)
    if ticker_filter is not None:
        panel = panel[panel["base_ticker"].isin(ticker_filter)].copy()

    ordered = panel.sort_values(
        ["date", "base_ticker", "is_historical_suffix", "suffix_yyyymm", "source_column"],
        ascending=[True, True, True, False, True],
    )

    valid_close = ordered["adj_close"].notna()
    collision_counts = (
        ordered[valid_close]
        .groupby(["date", "base_ticker"])["source_column"]
        .transform("size")
        .reindex(ordered.index, fill_value=0)
    )
    ordered = ordered[valid_close].copy()
    ordered["collision_count"] = collision_counts.loc[ordered.index].astype(np.int32)
    chosen = ordered.drop_duplicates(subset=["date", "base_ticker"], keep="first")

    histories: dict[str, pd.DataFrame] = {}
    for ticker, hist in chosen.groupby("base_ticker", sort=True):
        frame = hist[
            [
                "date",
                "base_ticker",
                "asset_id",
                "source_column",
                "is_historical_suffix",
                "collision_count",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
            ]
        ].rename(columns={"base_ticker": "ticker"})
        histories[ticker] = frame.sort_values("date").reset_index(drop=True)

    return histories
