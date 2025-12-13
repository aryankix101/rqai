import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from technical_json_pipeline import fetch_tiingo_ohlcv

from technical_indicators_calculator import TechnicalIndicatorCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 800
OUTPUT_DIR = "./output"
COMPANY_METADATA = {
    "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"}
}

def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Fetching {symbol} from Tiingo: {start_date} to {end_date}")
        df = fetch_tiingo_ohlcv(
            symbol=symbol,
            start=start_date,
            end=end_date
        )
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        logger.info(f"✓ Fetched {len(df)} days of data for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"✗ Failed to fetch {symbol}: {e}")
        return None


def calculate_indicators_for_symbol(
    symbol: str,
    hist: pd.DataFrame,
    metadata: Dict[str, str],
    benchmark_prices: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Calculate all technical indicators for a single symbol.
    
    Args:
        symbol: Ticker symbol
        hist: DataFrame with adjusted OHLCV data
        metadata: Dictionary with 'sector' and 'industry' keys
        benchmark_prices: S&P 500 price array (aligned with hist dates)
        benchmark_returns: S&P 500 returns array
    
    Returns:
        Complete technical indicators dictionary, or None if error
    """
    try:
        logger.info(f"Calculating indicators for {symbol}...")
        
        # Initialize calculator
        calculator = TechnicalIndicatorCalculator(hist)
        
        # Calculate all indicators
        indicators = calculator.calculate_all_indicators(
            symbol=symbol,
            sector=metadata.get("sector", "Unknown"),
            industry=metadata.get("industry", "Unknown"),
            benchmark_prices=benchmark_prices,
            benchmark_returns=benchmark_returns
        )
        
        logger.info(f"✓ Completed {symbol}")
        return indicators
    
    except Exception as e:
        logger.error(f"✗ Failed to calculate indicators for {symbol}: {e}")
        return None


def save_indicators_to_json(
    symbol: str,
    indicators: Dict[str, Any],
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Save indicators to JSON file.
    
    Args:
        symbol: Ticker symbol
        indicators: Technical indicators dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    filename = output_path / f"{symbol}_technical_indicators.json"
    
    with open(filename, 'w') as f:
        json.dump(indicators, f, indent=2)
    
    logger.info(f"Saved {symbol} indicators to {filename}")


def process_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    spy_prices: Optional[np.ndarray] = None,
    spy_returns: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Complete pipeline for a single symbol: fetch data, calculate indicators, save.
    
    Args:
        symbol: Ticker symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        spy_prices: Pre-fetched SPY prices (optional)
        spy_returns: Pre-fetched SPY returns (optional)
    
    Returns:
        Technical indicators dictionary, or None if error
    """
    # Fetch stock data
    hist = fetch_stock_data(symbol, start_date, end_date)
    if hist is None:
        return None
    
    # Get metadata
    metadata = COMPANY_METADATA.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
    
    # Calculate indicators
    indicators = calculate_indicators_for_symbol(
        symbol=symbol,
        hist=hist,
        metadata=metadata,
        benchmark_prices=spy_prices,
        benchmark_returns=spy_returns
    )
    
    if indicators is None:
        return None
    
    # Save to JSON
    save_indicators_to_json(symbol, indicators)
    
    return indicators


def process_multiple_symbols(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple symbols in batch.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date 'YYYY-MM-DD' (defaults to 800 days ago)
        end_date: End date 'YYYY-MM-DD' (defaults to today)
    
    Returns:
        Dictionary mapping symbol to indicators dictionary
    """
    # Set default dates
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    
    logger.info(f"Processing {len(symbols)} symbols from {start_date} to {end_date}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    # Fetch SPY benchmark data 
    logger.info("\n" + "="*80)
    logger.info("Fetching S&P 500 (SPY) benchmark data...")
    logger.info("="*80)
    
    spy_hist = fetch_stock_data("SPY", start_date, end_date)
    spy_prices = None
    spy_returns = None
    
    if spy_hist is not None and not spy_hist.empty:
        spy_prices = spy_hist['Close'].values
        spy_returns = np.diff(np.log(spy_prices))
        logger.info(f"✓ SPY benchmark loaded ({len(spy_hist)} days)")
    else:
        logger.warning("⚠ Could not load SPY benchmark. Relative metrics will be None.")
    
    # Process each symbol
    results = {}
    
    logger.info("\n" + "="*80)
    logger.info("Processing individual stocks...")
    logger.info("="*80 + "\n")
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
        logger.info("-" * 60)
        
        try:
            # Fetch stock data
            hist = fetch_stock_data(symbol, start_date, end_date)
            if hist is None:
                logger.warning(f"Skipping {symbol} due to data fetch error\n")
                continue
            
            # Align SPY data with this stock's dates
            aligned_spy_prices = None
            aligned_spy_returns = None
            
            if spy_hist is not None:
                common_dates = hist.index.intersection(spy_hist.index)
                if len(common_dates) > 0:
                    spy_aligned = spy_hist.loc[common_dates]
                    aligned_spy_prices = spy_aligned['Close'].values
                    aligned_spy_returns = np.diff(np.log(aligned_spy_prices))
                    
                    # Align the stock history
                    hist = hist.loc[common_dates]
                    logger.info(f"Aligned {len(common_dates)} common dates with SPY")
            
            # Get metadata
            metadata = COMPANY_METADATA.get(symbol, {"sector": "Unknown", "industry": "Unknown"})
            
            # Calculate indicators
            indicators = calculate_indicators_for_symbol(
                symbol=symbol,
                hist=hist,
                metadata=metadata,
                benchmark_prices=aligned_spy_prices,
                benchmark_returns=aligned_spy_returns
            )
            
            if indicators is not None:
                save_indicators_to_json(symbol, indicators)
                results[symbol] = indicators
            else:
                logger.warning(f"Skipping {symbol} due to calculation error")
        
        except Exception as e:
            logger.error(f"✗ Error processing {symbol}: {e}")
        
        logger.info("") 
    
    # Summary
    logger.info("="*80)
    logger.info(f"COMPLETED: {len(results)}/{len(symbols)} symbols processed successfully")
    logger.info("="*80)
    
    if results:
        logger.info(f"\nSuccessful: {', '.join(results.keys())}")
    
    failed = set(symbols) - set(results.keys())
    if failed:
        logger.info(f"\nFailed: {', '.join(failed)}")
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate technical indicators for stocks using Tiingo API"
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock ticker symbols (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date YYYY-MM-DD (default: 800 days ago)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    args = parser.parse_args()
    
    # Use the output directory from args
    output_dir = args.output_dir
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Process symbols
    results = process_multiple_symbols(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Print summary
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        logger.info("No arguments provided")        
        
        default_symbols = ["AAPL"]
        results = process_multiple_symbols(default_symbols)
    else:
        main()
 