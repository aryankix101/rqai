"""
Russell OHLCV signal dump builder.

This is the Russell-backed equivalent of the legacy Tiingo `signal_data_dump`
path and is expected to emit the full indicator contract consumed by
`deterministic.composite_score_calculator`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from data_source_config import DATASET_LABEL, DATASET_VERSION, RUSSELL_SIGNAL_DATA_DIR
from deterministic.signal_generation import MIN_SIGNAL_HISTORY_ROWS, SIGNAL_OUTPUT_COLUMNS, build_signal_frame_from_history
from russell_data_loader import load_russell_ohlcv_histories

METADATA_COLUMNS = [
    "date",
    "ticker",
    "asset_id",
    "source_column",
    "is_historical_suffix",
    "collision_count",
]
MANIFEST_COLUMNS = [
    "ticker",
    "status",
    "history_rows",
    "signal_rows",
    "range_start",
    "range_end",
    "output_file",
    "reason",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _empty_russell_signal_frame() -> pd.DataFrame:
    output_columns = SIGNAL_OUTPUT_COLUMNS + [col for col in METADATA_COLUMNS if col != "date"] + [
        "dataset_label",
        "dataset_version",
    ]
    return pd.DataFrame(columns=output_columns)


def _promote_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir = output_dir.with_name(f"{output_dir.name}.__previous__")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if output_dir.exists():
        output_dir.rename(backup_dir)

    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def build_russell_signal_frame(history: pd.DataFrame) -> pd.DataFrame:
    missing_metadata = [col for col in METADATA_COLUMNS if col not in history.columns]
    if missing_metadata:
        raise ValueError(f"history missing metadata columns: {missing_metadata}")

    signal_df = build_signal_frame_from_history(history)
    if signal_df.empty:
        return _empty_russell_signal_frame()

    metadata = history[METADATA_COLUMNS].copy()
    metadata["date"] = pd.to_datetime(metadata["date"])
    metadata = metadata.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    merged = signal_df.merge(metadata, on="date", how="left")
    merged["dataset_label"] = DATASET_LABEL
    merged["dataset_version"] = DATASET_VERSION
    return merged


def write_russell_signal_dumps(
    start_date: str | None = None,
    end_date: str | None = None,
    tickers: list[str] | None = None,
    output_dir: Path = RUSSELL_SIGNAL_DATA_DIR,
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir.parent)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=str(output_dir.parent)))
    ticker_filter = {ticker.upper() for ticker in tickers} if tickers else None
    histories = load_russell_ohlcv_histories(tickers=ticker_filter)

    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    print(f"\n{'=' * 80}")
    print("RUSSELL SIGNAL DUMP")
    print(f"Output directory: {output_dir}")
    print(f"Universe size (pre-filter): {len(histories)}")
    print(f"{'=' * 80}\n")

    written = 0
    skipped = 0
    failed = 0
    manifest_rows: list[dict[str, object]] = []
    for ticker in sorted(histories.keys()):
        if ticker_filter and ticker not in ticker_filter:
            continue

        history = histories[ticker]
        history_rows = len(history)

        if history_rows == 0:
            skipped += 1
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "status": "skipped_empty_history",
                    "history_rows": 0,
                    "signal_rows": 0,
                    "range_start": "",
                    "range_end": "",
                    "output_file": "",
                    "reason": "history is empty",
                }
            )
            continue

        if history_rows < MIN_SIGNAL_HISTORY_ROWS:
            skipped += 1
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "status": "skipped_short_history",
                    "history_rows": history_rows,
                    "signal_rows": 0,
                    "range_start": "",
                    "range_end": "",
                    "output_file": "",
                    "reason": f"needs at least {MIN_SIGNAL_HISTORY_ROWS} rows for signal warmup",
                }
            )
            continue

        try:
            signal_df = build_russell_signal_frame(history)
        except Exception as exc:
            failed += 1
            print(f"   ⚠ Skipping {ticker}: {type(exc).__name__}: {exc}")
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "status": "failed_build",
                    "history_rows": history_rows,
                    "signal_rows": 0,
                    "range_start": "",
                    "range_end": "",
                    "output_file": "",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if start_ts is not None:
            signal_df = signal_df[signal_df["date"] >= start_ts]
        if end_ts is not None:
            signal_df = signal_df[signal_df["date"] <= end_ts]
        signal_df = signal_df.reset_index(drop=True)

        if signal_df.empty:
            skipped += 1
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "status": "skipped_empty_window",
                    "history_rows": history_rows,
                    "signal_rows": 0,
                    "range_start": "",
                    "range_end": "",
                    "output_file": "",
                    "reason": "no signal rows remained after warmup/date filters",
                }
            )
            continue

        range_start = signal_df["date"].min().strftime("%Y-%m-%d")
        range_end = signal_df["date"].max().strftime("%Y-%m-%d")
        out_path = staging_dir / f"russell_signal_dump_{ticker}_{range_start}_{range_end}.csv"
        signal_df.to_csv(out_path, index=False)
        written += 1
        manifest_rows.append(
            {
                "ticker": ticker,
                "status": "written",
                "history_rows": history_rows,
                "signal_rows": len(signal_df),
                "range_start": range_start,
                "range_end": range_end,
                "output_file": out_path.name,
                "reason": "",
            }
        )

        if written % 100 == 0:
            print(f"   ✓ Written {written} tickers...")

    manifest_df = pd.DataFrame.from_records(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_path = staging_dir / "_russell_signal_dump_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "output_dir": str(output_dir),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "start_date": start_date,
        "end_date": end_date,
        "min_signal_history_rows": MIN_SIGNAL_HISTORY_ROWS,
        "manifest_path": str(manifest_path),
    }
    summary_path = staging_dir / "_russell_signal_dump_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if written == 0:
        raise RuntimeError(
            f"No Russell signal dump CSVs were written. Inspect {manifest_path} for skip/failure details."
        )

    _promote_staged_directory(staging_dir, output_dir)

    print(f"\n   ✓ Completed: {written}")
    print(f"   ⏩ Skipped: {skipped}\n")
    if failed:
        print(f"   ⚠ Failed: {failed}")
    print(f"   📄 Manifest: {output_dir / manifest_path.name}")
    print(f"   📄 Summary:  {output_dir / summary_path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OHLCV Russell signal dump CSVs.")
    parser.add_argument("--start", help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", help="Optional end date YYYY-MM-DD")
    parser.add_argument("--tickers", help="Comma-separated ticker subset")
    parser.add_argument("--output-dir", default=str(RUSSELL_SIGNAL_DATA_DIR), help="Output directory")
    args = parser.parse_args()

    tickers = [item.strip().upper() for item in args.tickers.split(",")] if args.tickers else None
    write_russell_signal_dumps(
        start_date=args.start,
        end_date=args.end,
        tickers=tickers,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
