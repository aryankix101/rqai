from pathlib import Path


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DATASET_VERSION = "russell_ohlcv_v1"
DATASET_LABEL = "russell_ohlcv"

LEGACY_SIGNAL_DATA_DIR = Path("backtest_results/signal_data_dump")
RUSSELL_SIGNAL_DATA_DIR = Path("backtest_results/russell_signal_dump")

LEGACY_COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores")
RUSSELL_COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores_russell")
RUSSELL_LLM_COMPOSITE_SCORES_DIR = Path("backtest_results/composite_scores_russell_llm")

DEFAULT_SIGNAL_DATA_DIR = RUSSELL_SIGNAL_DATA_DIR
DEFAULT_COMPOSITE_SCORES_DIR = RUSSELL_COMPOSITE_SCORES_DIR
DEFAULT_LLM_COMPOSITE_SCORES_DIR = RUSSELL_LLM_COMPOSITE_SCORES_DIR

SPY_SIGNAL_DATA_PATH = LEGACY_SIGNAL_DATA_DIR / "signal_data_dump_SPY_1993-01-01_2025-12-31.csv"

RUSSELL_DATA_DIR = Path("RusselData")
RUSSELL_PARQUET_PATH = RUSSELL_DATA_DIR / "compressed" / "Russell_1000_Current_&_Past_Data.parquet"
RUSSELL_EXCEL_PATH = RUSSELL_DATA_DIR / "Russell 1000 Current & Past.xlsx"
RUSSELL_MEMBERSHIP_PATH = RUSSELL_DATA_DIR / "r1000_membership.parquet"
RUSSELL_OHLCV_PATH = _first_existing_path(
    RUSSELL_DATA_DIR / "compressed" / "Russell_1000_OHLCV.parquet",
    RUSSELL_DATA_DIR / "russell_data_loader.parquet",
)
