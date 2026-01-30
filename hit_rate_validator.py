"""
Hit Rate Validator for Technical Indicators
============================================

Validates that technical indicators have >52% hit rate before deploying LLM layer.

TRADING LOGIC:
- Signal computed DAILY at close (using all data up to EOD)
- Entry = Next trading day OPEN
- Exit = SAME day CLOSE (intraday only)
- Return = (Close / Open) - 1.0 on entry day

Example: Monday signal → Tuesday open entry → Tuesday close exit

This measures:
- Pure INTRADAY follow-through (no overnight exposure)
- Tests if Monday's signal predicts Tuesday's intraday direction
- Cleanest validation metric (no gap risk)

Key Concepts:
- Hit Rate = (Signals with Positive Forward Return) / (Total Signals Generated)
- Tests on mini-universe of stocks
- Avoids look-ahead bias: Uses Thursday data to predict Friday→Monday return

Process:
1. Generate binary signals from technical indicators (using Thursday data)
2. Calculate forward returns (Friday open → Monday open)
3. Calculate hit rate for each configuration
"""

import json
import asyncio
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

from technical_indicators_calculator import TechnicalIndicatorCalculator
from technical_json_pipeline import fetch_tiingo_ohlcv
from composite_scores import (
    calculate_momentum_composite,
    calculate_trend_composite,
    calculate_volume_confirmation_composite,
    calculate_risk_composite,
    calculate_relative_strength_composite,
    calculate_next_day_alpha_composite
)
from deterministic_scoring import calculate_tier1_master_score
from regime_classifier import classify_market_regime


# Stock validation
MINI_UNIVERSE = ['T']


@dataclass
class SignalResult:
    """Container for signal and its forward returns."""
    date: pd.Timestamp
    symbol: str  # Added symbol to track which stock
    signal: int  # 1 = Buy, 0 = Neutral, -1 = Sell (from tier1_score)
    signal_strength: float  # Underlying tier1 score
    # New: Next-Day Alpha signal
    alpha_signal: int  # 1 = LONG, 0 = HOLD, -1 = SHORT (from next_day_alpha)
    alpha_score: float  # next_day_alpha raw score
    alpha_score_scaled: float  # temperature-scaled alpha score
    alpha_regime: str  # mean_reversion, continuation, or mixed
    # Forward returns
    forward_return_1d: Optional[float]
    forward_return_3d: Optional[float]
    forward_return_5d: Optional[float]
    indicators: Dict[str, Any]


# =============================================================================
# Shared Utilities
# =============================================================================

def calculate_forward_return_label(
    hist: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: int = 1,
    execution_model: str = "next_open_to_close"
) -> Optional[float]:
    """
    Calculate forward return label for a given signal date and horizon.
    
    This is the centralized function used by all diagnostics to ensure
    consistent return calculation between hit rate validation and 
    factor analysis.
    
    Args:
        hist: Price history DataFrame
        signal_date: Date signal was generated
        horizon: Trading days forward
        execution_model: "close_to_close", "next_open_to_close", "next_open_to_next_open"
    
    Returns:
        Forward return as decimal (0.01 = 1%), or None if not available
    """
    forward_rets = calculate_forward_returns(hist, signal_date, [horizon], execution_model)
    return forward_rets.get(horizon)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_binary_signal(
    tier1_score: float,
    threshold_buy: float = 0.1,
    threshold_sell: float = -0.1
) -> int:
    """Convert continuous score [-1, 1] to binary signal {-1, 0, 1}."""
    if tier1_score >= threshold_buy:
        return 1
    elif tier1_score <= threshold_sell:
        return -1
    else:
        return 0


# =============================================================================
# Forward Return Calculation
# =============================================================================

def calculate_forward_returns(
    hist: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizons: List[int] = [1, 3, 5],
    execution_model: str = "next_open_to_close"
) -> Dict[int, Optional[float]]:
    """
    Calculate forward returns from signal date (avoiding look-ahead bias).
    
    EXECUTION MODELS:
    - "close_to_close": Entry at signal_date close, exit at future close
      Return = close(t+h) / close(t) - 1
    - "next_open_to_close": Entry at next open, exit at same day close (intraday)
      Return = close(t+h) / open(t+h) - 1  
    - "next_open_to_next_open": Entry at next open, exit at next next open
      Return = open(t+h+1) / open(t+h) - 1
    
    Args:
        hist: Price history DataFrame
        signal_date: Date signal was generated (using data up to this date)
        horizons: Trading days forward to measure
        execution_model: How to execute the trade
    
    Returns:
        Dict of horizon -> return (decimal)
    """
    if signal_date not in hist.index:
        return {h: None for h in horizons}
    
    returns = {}
    
    for horizon in horizons:
        try:
            if execution_model == "close_to_close":
                # Entry: signal_date close, Exit: horizon days later close
                if signal_date not in hist.index:
                    returns[horizon] = None
                    continue
                    
                future_dates = hist.index[hist.index > signal_date]
                if len(future_dates) >= horizon:
                    exit_date = future_dates[horizon - 1]
                    entry_price = hist.loc[signal_date, 'Close']
                    exit_price = hist.loc[exit_date, 'Close']
                    returns[horizon] = (exit_price / entry_price) - 1.0 if entry_price > 0 else None
                else:
                    returns[horizon] = None
                    
            elif execution_model == "next_open_to_close":
                # Entry: next open, Exit: same day close (current implementation)
                future_dates = hist.index[hist.index > signal_date]
                if len(future_dates) >= horizon:
                    entry_date = future_dates[horizon - 1]
                    entry_price = hist.loc[entry_date, 'Open']
                    exit_price = hist.loc[entry_date, 'Close']
                    returns[horizon] = (exit_price / entry_price) - 1.0 if entry_price > 0 else None
                else:
                    returns[horizon] = None
                    
            elif execution_model == "next_open_to_next_open":
                # Entry: next open, Exit: next next open
                future_dates = hist.index[hist.index > signal_date]
                if len(future_dates) >= horizon + 1:  # Need one extra day
                    entry_date = future_dates[horizon - 1]
                    exit_date = future_dates[horizon]  # Next day
                    entry_price = hist.loc[entry_date, 'Open']
                    exit_price = hist.loc[exit_date, 'Open']
                    returns[horizon] = (exit_price / entry_price) - 1.0 if entry_price > 0 else None
                else:
                    returns[horizon] = None
                    
            else:
                raise ValueError(f"Unknown execution_model: {execution_model}")
                
        except (KeyError, IndexError):
            returns[horizon] = None
    
    return returns


# =============================================================================
# Hit Rate Calculation
# =============================================================================

def calculate_hit_rate(
    signals: List[SignalResult],
    horizon_days: int = 1,
    signal_filter: int = 0,  # 1 = Buy only, -1 = Sell only, 0 = All
    min_signals: int = 5,
    use_alpha_signal: bool = False  # If True, use alpha_signal instead of signal
) -> Dict[str, Any]:
    """Calculate hit rate for given signal configuration."""
    # Filter signals based on which signal type to use
    signal_attr = 'alpha_signal' if use_alpha_signal else 'signal'
    
    if signal_filter != 0:
        filtered = [s for s in signals if getattr(s, signal_attr) == signal_filter]
    else:
        filtered = [s for s in signals if getattr(s, signal_attr) != 0]
    
    # Get forward returns for horizon
    forward_returns = []
    for sig in filtered:
        if horizon_days == 1:
            ret = sig.forward_return_1d
        elif horizon_days == 3:
            ret = sig.forward_return_3d
        elif horizon_days == 5:
            ret = sig.forward_return_5d
        else:
            ret = None
        
        if ret is not None:
            signal_direction = getattr(sig, signal_attr)
            adjusted_ret = ret * signal_direction
            forward_returns.append(adjusted_ret)
    
    if len(forward_returns) < min_signals:
        return {
            "hit_rate": None,
            "total_signals": len(filtered),
            "valid_signals": len(forward_returns),
            "profitable_signals": 0,
            "avg_return": None,
            "median_return": None,
            "status": "INSUFFICIENT_DATA"
        }
    
    profitable = sum(1 for r in forward_returns if r > 0)
    hit_rate = profitable / len(forward_returns)
    avg_return = np.mean(forward_returns)
    
    return {
        "hit_rate": hit_rate,
        "total_signals": len(filtered),
        "valid_signals": len(forward_returns),
        "profitable_signals": profitable,
        "avg_return": avg_return,
        "median_return": np.median(forward_returns),
        "std_return": np.std(forward_returns),
        "status": "VALID" if hit_rate >= 0.52 else "BELOW_THRESHOLD"
    }


# =============================================================================
# Single Stock Validation
# =============================================================================

async def validate_single_stock(
    symbol: str,
    start_date: str = "1996-01-01",
    end_date: str = "2025-01-01",
    execution_model: str = "next_open_to_close",
    alpha_threshold_mode: str = "quantile",
    q_long: float = 80.0,
    q_short: float = 20.0
) -> Tuple[List[SignalResult], pd.DataFrame]:
    """
    Generate signals and calculate forward returns for single stock.
    
    TIMING (as of 2026-01-24):
    - Signal computed DAILY at close using all data up to that day
    - Entry = Next trading day OPEN
    - Exit = SAME day CLOSE (intraday only)
    - Forward return = (Close / Open) - 1.0 on entry day
    
    Example:
    - signal_date = Monday Jan 8 close
    - Use data up to Monday Jan 8 close to compute indicators
    - Entry = Tuesday Jan 9 open
    - Exit = Tuesday Jan 9 close (same day)
    - Return = (Tue Close / Tue Open) - 1.0 (intraday)
    
    This tests: "Does Monday's signal predict Tuesday's intraday direction?"
    Signals generated EVERY trading day (~250/year).
    """
    print(f"\n{'='*60}")
    print(f"Validating: {symbol}")
    print(f"{'='*60}")
    
    # Fetch data
    hist = fetch_tiingo_ohlcv(symbol, start_date, end_date)
    if hist is None or len(hist) < 252:
        print(f"Insufficient data for {symbol}")
        return [], pd.DataFrame()
    
    # Fetch benchmark (SPY) data for relative strength calculation
    spy_hist = fetch_tiingo_ohlcv("SPY", start_date, end_date)
    benchmark_returns = {}
    if spy_hist is not None and len(spy_hist) >= 252:
        spy_calc = TechnicalIndicatorCalculator(spy_hist)
        benchmark_returns = spy_calc.calculate_momentum()
    
    # Generate signals DAILY (not weekly) - true daily trading strategy
    # Skip first 252 days for technical indicator warmup period
    daily_dates = hist.index[252:]
    
    signals = []
    historical_alpha_scores = []  # Track raw scores for temperature calculation
    historical_scaled_scores = []  # Track scaled scores for quantile thresholds
    
    for signal_date in daily_dates[:-5]:  # Leave room for forward returns
        # Use data UP TO signal_date close to compute indicators
        # Signal is computed at EOD, entry happens next day at open
        hist_slice = hist[hist.index <= signal_date].copy()
        
        if len(hist_slice) < 200:
            continue
        
        try:
            # Calculate technical indicators
            calc = TechnicalIndicatorCalculator(hist_slice)
            indicators = {
                'momentum': calc.calculate_momentum(),
                'trend': calc.calculate_trend_indicators(),
                'volume_liquidity': calc.calculate_volume_liquidity_indicators(),
                'volatility': calc.calculate_volatility_indicators(),
                'risk': calc.calculate_risk_indicators(),
                'price_events': calc.calculate_price_events(),
                'relative_strength': {},
                'benchmark_returns': benchmark_returns
            }
            
            # Calculate composite scores
            composites = {
                'momentum_composite': calculate_momentum_composite(indicators),
                'trend_composite': calculate_trend_composite(indicators),
                'volume_composite': calculate_volume_confirmation_composite(indicators),
                'risk_composite': calculate_risk_composite(indicators),
                'rs_composite': calculate_relative_strength_composite(indicators),
                'next_day_alpha': calculate_next_day_alpha_composite(indicators)
            }
            
            # Calculate regime
            regime = classify_market_regime(indicators)
            
            tier1_result = calculate_tier1_master_score(composites, regime)
            tier1_score = tier1_result['tier1_score']
            
            next_day_alpha_result = calculate_next_day_alpha_composite(
                indicators,
                tier1_context=tier1_result
            )
            next_day_alpha_score = next_day_alpha_result['score']
            next_day_decision = next_day_alpha_result['decision']
            next_day_regime = next_day_alpha_result['regime']
            tier1_confirmation = next_day_alpha_result.get('tier1_confirmation')
            
            # Apply temperature scaling to alpha score for better distribution spread
            if len(historical_alpha_scores) >= 60:  # Need minimum history
                # Calculate rolling std of historical alpha scores (temperature)
                rolling_scores = historical_alpha_scores[-252:] if len(historical_alpha_scores) >= 252 else historical_alpha_scores
                temperature = max(0.05, np.std(rolling_scores))
                
                # Apply tanh scaling: spreads out small values, compresses large ones
                alpha_score_scaled = np.tanh(next_day_alpha_score / temperature)
            else:
                # Fallback during warmup
                alpha_score_scaled = next_day_alpha_score
            
            # Track historical scores
            historical_alpha_scores.append(next_day_alpha_score)
            historical_scaled_scores.append(alpha_score_scaled)
            
            # Generate tier-1 signal using fixed thresholds on tier1_score
            tier1_signal = generate_binary_signal(tier1_score)
            
            # Generate alpha signal using quantile-based thresholds
            if alpha_threshold_mode == "quantile" and len(historical_scaled_scores) >= 60:  # Need minimum history for quantiles
                # Use rolling window of last 252 trading days (1 year) for quantiles on scaled scores
                rolling_scaled = historical_scaled_scores[-252:] if len(historical_scaled_scores) >= 252 else historical_scaled_scores
                q_long_threshold = np.percentile(rolling_scaled, q_long)   # q_long percentile for long
                q_short_threshold = np.percentile(rolling_scaled, q_short)  # q_short percentile for short
                
                if alpha_score_scaled >= q_long_threshold:
                    alpha_signal = 1  # LONG
                elif alpha_score_scaled <= q_short_threshold:
                    alpha_signal = -1  # SHORT
                else:
                    alpha_signal = 0  # HOLD
            elif alpha_threshold_mode == "fixed":
                # Use original decision-based logic with fixed thresholds
                alpha_signal = 1 if next_day_decision == 'LONG' else (-1 if next_day_decision == 'SHORT' else 0)
            else:
                # Fallback during warmup period
                alpha_signal = 1 if next_day_decision == 'LONG' else (-1 if next_day_decision == 'SHORT' else 0)
            
            # Track signals - use tier-1 signal for the main signal, alpha for comparison
            if tier1_signal != 0 or alpha_signal != 0:
                forward_rets = calculate_forward_returns(hist, signal_date, [1, 3, 5], execution_model)
                
                signal_result = SignalResult(
                    date=signal_date,
                    symbol=symbol,
                    signal=tier1_signal,
                    signal_strength=tier1_score,
                    alpha_signal=alpha_signal,
                    alpha_score=next_day_alpha_score,
                    alpha_score_scaled=alpha_score_scaled,
                    alpha_regime=next_day_regime,
                    forward_return_1d=forward_rets[1],
                    forward_return_3d=forward_rets[3],
                    forward_return_5d=forward_rets[5],
                    indicators={
                        **indicators,
                        'composites': composites,
                        'tier1_score': tier1_score,
                        'regime': regime
                    }
                )
                signals.append(signal_result)
        
        except Exception as e:
            print(f"Error on {signal_date}: {e}")
            continue
    
    print(f"Generated {len(signals)} signals for {symbol}")
    return signals, hist


# =============================================================================
# Multi-Stock Universe Validation
# =============================================================================

async def validate_mini_universe(
    symbols: List[str] = MINI_UNIVERSE,
    start_date: str = "1996-01-01",
    end_date: str = "2025-01-01",
    execution_model: str = "next_open_to_close",
    alpha_threshold_mode: str = "quantile",
    q_long: float = 80.0,
    q_short: float = 20.0
) -> Tuple[Dict[str, Any], List[SignalResult], Dict[str, List[SignalResult]]]:
    """
    Validate hit rates across mini-universe of stocks.
    
    Returns:
        (results_dict, all_signals, per_stock_signals)
    """
    print(f"\n{'='*60}")
    print(f"MINI-UNIVERSE HIT RATE VALIDATION")
    print(f"Stocks: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    all_signals = []
    stock_results = {}  # symbol -> list of signals
    stock_hist = {}  # symbol -> historical data for charting
    
    for symbol in symbols:
        signals, hist = await validate_single_stock(
            symbol, 
            start_date, 
            end_date, 
            execution_model,
            alpha_threshold_mode,
            q_long,
            q_short
        )
        all_signals.extend(signals)
        stock_results[symbol] = signals
        stock_hist[symbol] = hist
    
    # Calculate aggregate hit rates
    print(f"\n{'='*60}")
    print(f"AGGREGATE HIT RATE RESULTS")
    print(f"{'='*60}\n")
    
    results = {}
    
    for horizon in [1, 3, 5]:
        print(f"\n--- {horizon}-Day Forward Returns ---")
        
        # ===== OLD METHOD: Tier-1 Score =====
        print(f"\n  [TIER-1 SCORE (old method)]")
        buy_hr = calculate_hit_rate(all_signals, horizon, signal_filter=1, use_alpha_signal=False)
        print(f"    BUY:  {buy_hr['hit_rate']:.2%} ({buy_hr['valid_signals']} signals)" if buy_hr['hit_rate'] else f"    BUY:  N/A ({buy_hr['valid_signals']} signals)")
        
        sell_hr = calculate_hit_rate(all_signals, horizon, signal_filter=-1, use_alpha_signal=False)
        print(f"    SELL: {sell_hr['hit_rate']:.2%} ({sell_hr['valid_signals']} signals)" if sell_hr['hit_rate'] else f"    SELL: N/A ({sell_hr['valid_signals']} signals)")
        
        all_hr = calculate_hit_rate(all_signals, horizon, signal_filter=0, use_alpha_signal=False)
        print(f"    ALL:  {all_hr['hit_rate']:.2%} ({all_hr['valid_signals']} signals)" if all_hr['hit_rate'] else f"    ALL:  N/A ({all_hr['valid_signals']} signals)")
        
        # ===== NEW METHOD: Next-Day Alpha =====
        print(f"\n  [NEXT-DAY ALPHA (new method)]")
        alpha_long_hr = calculate_hit_rate(all_signals, horizon, signal_filter=1, use_alpha_signal=True)
        print(f"    LONG:  {alpha_long_hr['hit_rate']:.2%} ({alpha_long_hr['valid_signals']} signals, avg={alpha_long_hr['avg_return']:.2%})" if alpha_long_hr['hit_rate'] else f"    LONG:  N/A ({alpha_long_hr['valid_signals']} signals)")
        
        alpha_short_hr = calculate_hit_rate(all_signals, horizon, signal_filter=-1, use_alpha_signal=True)
        print(f"    SHORT: {alpha_short_hr['hit_rate']:.2%} ({alpha_short_hr['valid_signals']} signals, avg={alpha_short_hr['avg_return']:.2%})" if alpha_short_hr['hit_rate'] else f"    SHORT: N/A ({alpha_short_hr['valid_signals']} signals)")
        
        alpha_all_hr = calculate_hit_rate(all_signals, horizon, signal_filter=0, use_alpha_signal=True)
        print(f"    ALL:   {alpha_all_hr['hit_rate']:.2%} ({alpha_all_hr['valid_signals']} signals, avg={alpha_all_hr['avg_return']:.2%})" if alpha_all_hr['hit_rate'] else f"    ALL:   N/A ({alpha_all_hr['valid_signals']} signals)")
        
        results[f"{horizon}d"] = {
            "tier1": {"buy": buy_hr, "sell": sell_hr, "all": all_hr},
            "alpha": {"long": alpha_long_hr, "short": alpha_short_hr, "all": alpha_all_hr}
        }
    
    # Per-stock breakdown (comparing both methods)
    print(f"\n{'='*60}")
    print(f"PER-STOCK BREAKDOWN (1-Day Returns)")
    print(f"{'='*60}")
    print(f"{'Symbol':8s} | {'Tier-1 HR':>12s} | {'Alpha HR':>12s} | {'Alpha Signals':>14s}")
    print(f"{'-'*60}")
    
    for symbol, signals in stock_results.items():
        tier1_hr = calculate_hit_rate(signals, horizon_days=1, signal_filter=0, min_signals=5, use_alpha_signal=False)
        alpha_hr = calculate_hit_rate(signals, horizon_days=1, signal_filter=0, min_signals=3, use_alpha_signal=True)
        
        tier1_str = f"{tier1_hr['hit_rate']:.1%}" if tier1_hr['hit_rate'] else "N/A"
        alpha_str = f"{alpha_hr['hit_rate']:.1%}" if alpha_hr['hit_rate'] else "N/A"
        alpha_count = alpha_hr['valid_signals']
        
        print(f"{symbol:8s} | {tier1_str:>12s} | {alpha_str:>12s} | {alpha_count:>14d}")
    
    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"hit_rate_validation_{timestamp}.json"
    
    save_results = {
        "config": {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "total_signals": len(all_signals)
        },
        "aggregate_results": results,
        "per_stock": {
            symbol: calculate_hit_rate(sigs, 1, 0)
            for symbol, sigs in stock_results.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return save_results, all_signals, stock_results, stock_hist


# =============================================================================
# Visualization - Per Stock Charts
# =============================================================================

def plot_stock_analysis(
    symbol: str,
    signals: List[SignalResult],
    hist: pd.DataFrame,
    output_dir: Path
):
    """Generate hit rate analysis charts for a single stock."""
    if not signals:
        print(f"No signals for {symbol}, skipping chart")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{symbol} - Pure Signal Analysis', fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Chart 1: Hit Rate by Horizon
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    horizons = ['1d', '3d', '5d']
    hit_rates = []
    for h in [1, 3, 5]:
        hr = calculate_hit_rate(signals, h, signal_filter=0)
        hit_rates.append(hr['hit_rate'] or 0)
    
    colors = ['#2ecc71' if hr >= 0.52 else '#e74c3c' for hr in hit_rates]
    bars = ax1.bar(horizons, hit_rates, alpha=0.8, color=colors, edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
    ax1.axhline(y=0.50, color='gray', linestyle=':', label='Random (50%)', linewidth=1)
    
    ax1.set_xlabel('Forward Horizon', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Hit Rate by Horizon', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    
    for bar, v in zip(bars, hit_rates):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.1%}', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Chart 2: Signal Distribution
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    buy_count = sum(1 for s in signals if s.signal == 1)
    sell_count = sum(1 for s in signals if s.signal == -1)
    
    bars = ax2.bar(['Buy', 'Sell'], [buy_count, sell_count], 
                   alpha=0.8, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    
    ax2.set_xlabel('Signal Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signal Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    for bar, v in zip(bars, [buy_count, sell_count]):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 1, str(v), 
                ha='center', fontsize=10, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Chart 3: Return Distribution
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    returns = [s.forward_return_1d * 100 for s in signals 
               if s.forward_return_1d is not None and s.signal != 0]
    
    if returns:
        ax3.hist(returns, bins=25, alpha=0.7, color='#3498db', edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
        
        # Add mean line
        mean_ret = np.mean(returns)
        ax3.axvline(x=mean_ret, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_ret:.2f}%')
        ax3.legend()
    
    ax3.set_xlabel('1-Day Forward Return (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # -------------------------------------------------------------------------
    # Chart 4: Cumulative Returns - Strategy vs Buy & Hold
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    sorted_signals = sorted([s for s in signals if s.forward_return_1d is not None], 
                           key=lambda x: x.date)
    
    if sorted_signals and hist is not None and len(hist) > 0:
        # Strategy returns (following our signals)
        dates = [s.date for s in sorted_signals]
        cumulative = 1.0
        strategy_returns = []
        
        for s in sorted_signals:
            cumulative *= (1 + s.forward_return_1d)
            strategy_returns.append(cumulative)
        
        # Buy & Hold returns (underlying stock performance)
        # Get the stock price at first and last signal dates
        first_date = dates[0]
        # Filter hist to match signal date range
        hist_filtered = hist[(hist.index >= first_date) & (hist.index <= dates[-1])]
        if len(hist_filtered) > 0:
            buy_hold_base = hist_filtered['Close'].iloc[0]
            buy_hold_returns = hist_filtered['Close'] / buy_hold_base
            
            # Plot buy & hold
            ax4.plot(hist_filtered.index, buy_hold_returns, linewidth=2, 
                    color='#3498db', alpha=0.8, label='Buy & Hold')
        
        # Plot strategy
        strat_color = '#2ecc71' if strategy_returns[-1] > 1 else '#e74c3c'
        ax4.plot(dates, strategy_returns, linewidth=2, color=strat_color, 
                label='Signal Strategy', linestyle='-', marker='o', markersize=3, alpha=0.8)
        
        ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
        
        # Add final return labels
        strat_final = (strategy_returns[-1] - 1) * 100
        if len(hist_filtered) > 0:
            bh_final = (buy_hold_returns.iloc[-1] - 1) * 100
            ax4.text(0.02, 0.98, f'Strategy: {strat_final:+.1f}%\nBuy&Hold: {bh_final:+.1f}%', 
                    transform=ax4.transAxes, fontsize=11, fontweight='bold', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.02, 0.98, f'Strategy: {strat_final:+.1f}%', transform=ax4.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top')
        
        ax4.legend(loc='upper right', fontsize=9)
    
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax4.set_title('Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_file = output_dir / f"hit_rate_{symbol}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    
    plt.close('all')


def plot_aggregate_analysis(
    results: Dict[str, Any],
    all_signals: List[SignalResult],
    stock_results: Dict[str, List[SignalResult]],
    output_dir: Path
):
    """Generate aggregate analysis across all stocks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Aggregate Hit Rate Analysis - All Stocks', fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Chart 1: Per-Stock Hit Rates
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    symbols = list(stock_results.keys())
    hit_rates = []
    for sym in symbols:
        hr = calculate_hit_rate(stock_results[sym], 1, 0)
        hit_rates.append(hr['hit_rate'] or 0)
    
    colors = ['#2ecc71' if hr >= 0.52 else '#e74c3c' for hr in hit_rates]
    bars = ax1.bar(symbols, hit_rates, alpha=0.8, color=colors, edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
    ax1.axhline(y=0.50, color='gray', linestyle=':', linewidth=1)
    
    ax1.set_xlabel('Stock', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Hit Rate (1-Day)', fontsize=11, fontweight='bold')
    ax1.set_title('Per-Stock Hit Rates', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    
    for bar, v in zip(bars, hit_rates):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.1%}', 
                    ha='center', fontsize=9, fontweight='bold', rotation=0)
    
    # -------------------------------------------------------------------------
    # Chart 2: Signal Count by Stock
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    counts = [len(stock_results[sym]) for sym in symbols]
    bars = ax2.bar(symbols, counts, alpha=0.8, color='#3498db', edgecolor='black')
    
    ax2.set_xlabel('Stock', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Signal Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signals Generated Per Stock', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    for bar, v in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 1, str(v), 
                ha='center', fontsize=9, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Chart 3: Buy vs Sell Win Rates
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    buy_wrs = []
    sell_wrs = []
    for sym in symbols:
        buy_hr = calculate_hit_rate(stock_results[sym], 1, signal_filter=1)
        sell_hr = calculate_hit_rate(stock_results[sym], 1, signal_filter=-1)
        buy_wrs.append(buy_hr['hit_rate'] or 0)
        sell_wrs.append(sell_hr['hit_rate'] or 0)
    
    x = np.arange(len(symbols))
    width = 0.35
    
    ax3.bar(x - width/2, buy_wrs, width, label='Buy Signals', alpha=0.8, color='#2ecc71', edgecolor='black')
    ax3.bar(x + width/2, sell_wrs, width, label='Sell Signals', alpha=0.8, color='#e74c3c', edgecolor='black')
    ax3.axhline(y=0.52, color='green', linestyle='--', linewidth=2)
    ax3.axhline(y=0.50, color='gray', linestyle=':', linewidth=1)
    
    ax3.set_xlabel('Stock', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
    ax3.set_title('Buy vs Sell Win Rates', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(symbols)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_ylim([0, 0.8])
    
    # -------------------------------------------------------------------------
    # Chart 4: Aggregate Hit Rate by Horizon
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    horizons = ['1d', '3d', '5d']
    agg_hit_rates = []
    for h in [1, 3, 5]:
        hr = calculate_hit_rate(all_signals, h, signal_filter=0)
        agg_hit_rates.append(hr['hit_rate'] or 0)
    
    colors = ['#2ecc71' if hr >= 0.52 else '#e74c3c' for hr in agg_hit_rates]
    bars = ax4.bar(horizons, agg_hit_rates, alpha=0.8, color=colors, edgecolor='black')
    ax4.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
    ax4.axhline(y=0.50, color='gray', linestyle=':', linewidth=1)
    
    ax4.set_xlabel('Forward Horizon', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Aggregate Hit Rate (All Stocks)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(alpha=0.3, axis='y')
    ax4.set_ylim([0, 0.8])
    
    for bar, v in zip(bars, agg_hit_rates):
        if v > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.1%}', 
                    ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"hit_rate_aggregate_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    
    plt.close('all')


def plot_factor_analysis(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """Analyze individual factor correlation with forward returns."""
    valid_signals = [s for s in all_signals if s.forward_return_1d is not None and s.signal != 0]
    
    if len(valid_signals) < 20:
        print("Insufficient signals for factor analysis")
        return
    
    def get_composite_score(s, composite_name):
        comp = s.indicators.get('composites', {}).get(composite_name)
        if comp is None:
            return None
        if isinstance(comp, dict):
            return comp.get('score', 0)
        return comp
    
    data = {
        'Tier-1 Score': [s.indicators.get('tier1_score', 0) for s in valid_signals],
        'Momentum': [get_composite_score(s, 'momentum_composite') for s in valid_signals],
        'Trend': [get_composite_score(s, 'trend_composite') for s in valid_signals],
        'Volume': [get_composite_score(s, 'volume_composite') for s in valid_signals],
        'Risk': [get_composite_score(s, 'risk_composite') for s in valid_signals],
        'RSI': [s.indicators.get('trend', {}).get('rsi_14d', 50) for s in valid_signals],
        'ADX': [s.indicators.get('trend', {}).get('adx_14d', 0) for s in valid_signals],
        'MACD': [s.indicators.get('trend', {}).get('macd', 0) for s in valid_signals],
    }
    
    forward_returns = [s.forward_return_1d * 100 for s in valid_signals]
    
    # Calculate correlations
    correlations = {}
    for name, values in data.items():
        clean_vals = [v if v is not None else 0 for v in values]
        if len(set(clean_vals)) > 1:
            try:
                corr = np.corrcoef(clean_vals, forward_returns)[0, 1]
                if not np.isnan(corr):
                    correlations[name] = corr
            except:
                pass
    
    sorted_factors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle('Factor Correlation Analysis', fontsize=16, fontweight='bold')
    
    for idx, (factor_name, corr) in enumerate(sorted_factors[:8]):
        ax = axes[idx]
        
        factor_values = [v if v is not None else 0 for v in data[factor_name]]
        colors = ['green' if r > 0 else 'red' for r in forward_returns]
        
        ax.scatter(factor_values, forward_returns, alpha=0.4, c=colors, s=20, edgecolors='black', linewidth=0.2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        try:
            z = np.polyfit(factor_values, forward_returns, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(factor_values), max(factor_values), 100)
            ax.plot(x_trend, p(x_trend), "b--", linewidth=2, alpha=0.7)
        except:
            pass
        
        ax.set_xlabel(factor_name, fontsize=9, fontweight='bold')
        ax.set_ylabel('1D Return (%)', fontsize=9)
        ax.set_title(f'Corr: {corr:+.3f}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    
    for idx in range(len(sorted_factors), 8):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"factor_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("FACTOR CORRELATION SUMMARY")
    print("="*60)
    for factor_name, corr in sorted_factors:
        direction = "↑" if corr > 0 else "↓"
        print(f"  {factor_name:15s}: {corr:+.3f} {direction}")
    
    # Also calculate correlations with ABSOLUTE returns (magnitude, not direction)
    abs_returns = [abs(r) for r in forward_returns]
    abs_correlations = {}
    for name, values in data.items():
        clean_vals = [v if v is not None else 0 for v in values]
        if len(set(clean_vals)) > 1:
            try:
                corr = np.corrcoef(clean_vals, abs_returns)[0, 1]
                if not np.isnan(corr):
                    abs_correlations[name] = corr
            except:
                pass
    
    sorted_abs = sorted(abs_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n" + "="*60)
    print("FACTOR vs ABSOLUTE RETURNS (magnitude of move)")
    print("="*60)
    print("  (Positive = factor predicts bigger moves regardless of direction)")
    for factor_name, corr in sorted_abs:
        direction = "↑" if corr > 0 else "↓"
        print(f"  {factor_name:15s}: {corr:+.3f} {direction}")
    
    plt.close('all')


def plot_individual_factor_charts(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """
    Generate detailed individual factor analysis charts.
    Creates scatter plots for each factor vs forward returns with trend lines.
    """
    valid_signals = [s for s in all_signals if s.forward_return_1d is not None and s.signal != 0]
    
    if len(valid_signals) < 20:
        print("Insufficient signals for individual factor analysis")
        return
    
    def get_composite_score(s, composite_name):
        comp = s.indicators.get('composites', {}).get(composite_name)
        if comp is None:
            return None
        if isinstance(comp, dict):
            return comp.get('score', 0)
        return comp
    
    # Extended list of factors to analyze (including NEW z-scored features)
    data = {
        'Tier-1 Score': [s.indicators.get('tier1_score', 0) for s in valid_signals],
        'Next-Day Alpha': [s.alpha_score for s in valid_signals],
        'Momentum Composite': [get_composite_score(s, 'momentum_composite') for s in valid_signals],
        'Trend Composite': [get_composite_score(s, 'trend_composite') for s in valid_signals],
        'Volume Composite': [get_composite_score(s, 'volume_composite') for s in valid_signals],
        'Risk Composite': [get_composite_score(s, 'risk_composite') for s in valid_signals],
        # NEW: Z-scored returns
        'ret_1d_z': [s.indicators.get('momentum', {}).get('ret_1d_zscore', 0) for s in valid_signals],
        'ret_3d_z': [s.indicators.get('momentum', {}).get('ret_3d_zscore', 0) for s in valid_signals],
        'mom_3m_z': [s.indicators.get('momentum', {}).get('mom_3m_zscore', 0) for s in valid_signals],
        # NEW: Autocorrelation (regime detection)
        'autocorr_5d': [s.indicators.get('momentum', {}).get('autocorr_5d', 0) for s in valid_signals],
        'autocorr_10d': [s.indicators.get('momentum', {}).get('autocorr_10d', 0) for s in valid_signals],
        # Standard indicators
        'RSI (14d)': [s.indicators.get('trend', {}).get('rsi_14d', 50) for s in valid_signals],
        'ADX': [s.indicators.get('trend', {}).get('adx_14d', 20) for s in valid_signals],
        'MACD': [s.indicators.get('trend', {}).get('macd', 0) for s in valid_signals],
        'MACD Histogram': [s.indicators.get('trend', {}).get('macd_histogram', 0) for s in valid_signals],
        'ATR %': [s.indicators.get('volatility', {}).get('atr_14d_pct', 0) for s in valid_signals],
        'Volume Spike Z': [s.indicators.get('volume_liquidity', {}).get('volume_spike_z', 0) for s in valid_signals],
        'Vol Percentile': [s.indicators.get('volatility', {}).get('vol_percentile', 50) for s in valid_signals],
        'Vol Ratio': [s.indicators.get('volatility', {}).get('vol_ratio_21v252', 1.0) for s in valid_signals],
    }
    
    forward_returns = [s.forward_return_1d * 100 for s in valid_signals]
    
    # Calculate correlations for all factors
    correlations = {}
    for name, values in data.items():
        clean_vals = [v if v is not None else 0 for v in values]
        if len(set(clean_vals)) > 1:
            try:
                corr = np.corrcoef(clean_vals, forward_returns)[0, 1]
                if not np.isnan(corr):
                    correlations[name] = corr
            except:
                pass
    
    # Sort by absolute correlation
    sorted_factors = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create 3x4 grid for 12 factors
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()
    fig.suptitle('Individual Factor Analysis - Correlation with 1-Day Forward Returns', 
                 fontsize=16, fontweight='bold')
    
    for idx, (factor_name, corr) in enumerate(sorted_factors[:12]):
        ax = axes[idx]
        
        factor_values = [v if v is not None else 0 for v in data[factor_name]]
        
        # Color points by whether return was positive or negative
        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in forward_returns]
        
        ax.scatter(factor_values, forward_returns, alpha=0.5, c=colors, s=30, 
                  edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        
        # Add trend line
        try:
            z = np.polyfit(factor_values, forward_returns, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(factor_values), max(factor_values), 100)
            ax.plot(x_trend, p(x_trend), "b-", linewidth=2.5, alpha=0.8, 
                   label=f'y={z[0]:.3f}x+{z[1]:.2f}')
        except:
            pass
        
        # Format title with correlation and interpretation
        corr_color = '#2ecc71' if corr > 0.05 else '#e74c3c' if corr < -0.05 else '#7f8c8d'
        strength = 'Strong' if abs(corr) > 0.15 else 'Moderate' if abs(corr) > 0.08 else 'Weak'
        
        ax.set_xlabel(factor_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('1D Return (%)', fontsize=9)
        ax.set_title(f'{factor_name}\nCorr: {corr:+.3f} ({strength})', fontsize=10, fontweight='bold',
                    color=corr_color)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(sorted_factors), 12):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"individual_factor_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    
    plt.close('all')


def plot_quintile_analysis(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """
    Analyze performance by signal strength quintiles.
    Shows if stronger signals lead to better returns.
    """
    valid_signals = [s for s in all_signals if s.forward_return_1d is not None and s.signal != 0]
    
    if len(valid_signals) < 25:
        print("Insufficient signals for quintile analysis")
        return
    
    # Get signal strengths and returns
    strengths = [abs(s.signal_strength) for s in valid_signals]
    returns = [s.forward_return_1d * 100 for s in valid_signals]
    
    try:
        df = pd.DataFrame({'strength': strengths, 'return': returns})
        df['quintile'] = pd.qcut(df['strength'], q=5, labels=False, duplicates='drop')
        
        # Calculate stats per quintile
        quintile_stats = df.groupby('quintile').agg({
            'return': ['mean', 'std', 'count'],
            'strength': ['min', 'max']
        })
        
        quintile_stats.columns = ['avg_return', 'std_return', 'count', 'min_strength', 'max_strength']
        quintile_stats = quintile_stats.reset_index()
        quintile_stats = quintile_stats.sort_values('quintile')
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Signal Strength Quintile Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Average Return by Quintile
        ax1 = axes[0]
        
        labels = []
        for _, row in quintile_stats.iterrows():
            labels.append(f"Q{int(row['quintile'])+1}\n({row['min_strength']:.2f}-{row['max_strength']:.2f})")
        
        x = np.arange(len(quintile_stats))
        colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in quintile_stats['avg_return']]
        
        bars = ax1.bar(x, quintile_stats['avg_return'], alpha=0.8, color=colors, edgecolor='black')
        ax1.errorbar(x, quintile_stats['avg_return'], yerr=quintile_stats['std_return']/2, 
                    fmt='none', color='black', capsize=5)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Signal Strength Quintile (Score Range)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Average 1-Day Return (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Average Return by Signal Strength', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.grid(alpha=0.3, axis='y')
        
        for bar, row in zip(bars, quintile_stats.itertuples()):
            height = bar.get_height()
            offset = 0.15 if height >= 0 else -0.25
            ax1.text(bar.get_x() + bar.get_width()/2, height + offset, 
                    f'n={int(row.count)}', ha='center', fontsize=9)
        
        # Chart 2: Hit Rate by Quintile
        ax2 = axes[1]
        
        hit_rates = []
        for q in quintile_stats['quintile']:
            q_returns = df[df['quintile'] == q]['return']
            if len(q_returns) > 0:
                hr = (q_returns > 0).sum() / len(q_returns)
                hit_rates.append(hr)
            else:
                hit_rates.append(0)
        
        colors = ['#2ecc71' if hr >= 0.52 else '#e74c3c' for hr in hit_rates]
        bars = ax2.bar(x, hit_rates, alpha=0.8, color=colors, edgecolor='black')
        ax2.axhline(y=0.52, color='green', linestyle='--', label='Target (52%)', linewidth=2)
        ax2.axhline(y=0.50, color='gray', linestyle=':', label='Random (50%)', linewidth=1)
        
        ax2.set_xlabel('Signal Strength Quintile', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
        ax2.set_title('Hit Rate by Signal Strength', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(alpha=0.3, axis='y')
        ax2.set_ylim([0, 0.8])
        
        for bar, v in zip(bars, hit_rates):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.1%}', 
                        ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_dir / f"quintile_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {plot_file}")
        
        plt.close('all')
        
    except Exception as e:
        print(f"Error in quintile analysis: {e}")


def plot_regime_performance(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """Analyze hit rate performance by regime."""
    regimes = ['mean_reversion', 'continuation', 'mixed']
    
    # Filter alpha signals by regime
    regime_data = {}
    for regime in regimes:
        regime_signals = [s for s in all_signals if s.alpha_regime == regime and s.alpha_signal != 0]
        regime_data[regime] = regime_signals
    
    if sum(len(v) for v in regime_data.values()) < 10:
        print("Insufficient signals for regime analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance by Regime Type', fontsize=16, fontweight='bold')
    
    # Chart 1: Hit rates by regime
    ax1 = axes[0, 0]
    hit_rates = []
    counts = []
    for regime in regimes:
        signals = regime_data[regime]
        if signals:
            hr = calculate_hit_rate([s for s in signals if s.alpha_signal != 0], 1, signal_filter=0)
            hit_rates.append(hr['hit_rate'] or 0)
            counts.append(len(signals))
        else:
            hit_rates.append(0)
            counts.append(0)
    
    colors = ['#e74c3c', '#3498db', '#95a5a6']
    bars = ax1.bar(regimes, hit_rates, alpha=0.8, color=colors, edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', linewidth=2, label='Target')
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Hit Rate by Regime (1D)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    
    for bar, hr, cnt in zip(bars, hit_rates, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, hr + 0.02, 
                f'{hr:.1%}\n(n={cnt})', ha='center', fontsize=9, fontweight='bold')
    
    # Chart 2: Signal distribution by regime
    ax2 = axes[0, 1]
    ax2.bar(regimes, counts, alpha=0.8, color=colors, edgecolor='black')
    ax2.set_ylabel('Signal Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signals by Regime', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Chart 3: Average return by regime
    ax3 = axes[1, 0]
    avg_returns = []
    for regime in regimes:
        signals = regime_data[regime]
        if signals:
            returns = [s.forward_return_1d * 100 for s in signals if s.forward_return_1d is not None]
            avg_returns.append(np.mean(returns) if returns else 0)
        else:
            avg_returns.append(0)
    
    bars = ax3.bar(regimes, avg_returns, alpha=0.8, color=colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Avg Return (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Average 1D Return by Regime', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    for bar, ret in zip(bars, avg_returns):
        ax3.text(bar.get_x() + bar.get_width()/2, ret + 0.05 if ret > 0 else ret - 0.1,
                f'{ret:.2f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Chart 4: Return distributions
    ax4 = axes[1, 1]
    for regime, color in zip(regimes, colors):
        signals = regime_data[regime]
        if signals:
            returns = [s.forward_return_1d * 100 for s in signals if s.forward_return_1d is not None]
            if returns:
                ax4.hist(returns, bins=20, alpha=0.5, label=regime, color=color, edgecolor='black')
    
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('1D Return (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Return Distribution by Regime', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"regime_performance_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    plt.close('all')


def plot_confirmation_analysis(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """Analyze tier1-alpha confirmation impact."""
    # Extract confirmation data
    high_conf = []
    low_conf = []
    neutral_conf = []
    
    for s in all_signals:
        if s.alpha_signal == 0:
            continue
        
        conf = s.indicators.get('composites', {}).get('next_day_alpha', {}).get('tier1_confirmation')
        if conf:
            conf_level = conf.get('confidence', 'neutral')
            if conf_level == 'high':
                high_conf.append(s)
            elif conf_level == 'low':
                low_conf.append(s)
            else:
                neutral_conf.append(s)
    
    total_conf = len(high_conf) + len(low_conf) + len(neutral_conf)
    if total_conf == 0:
        print("No signals with confirmation data available")
        return
    elif total_conf < 10:
        print(f"⚠️  WARNING: Only {total_conf} signals with confirmation data (low sample size - interpret cautiously)")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tier1-Alpha Confirmation Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Hit rates by confidence level
    ax1 = axes[0, 0]
    conf_levels = ['High\n(Aligned)', 'Low\n(Diverged)', 'Neutral\n(Weak)']
    conf_data = [high_conf, low_conf, neutral_conf]
    hit_rates = []
    counts = []
    
    for signals in conf_data:
        if signals:
            hr = calculate_hit_rate(signals, 1, signal_filter=0)
            hit_rates.append(hr['hit_rate'] or 0)
            counts.append(len(signals))
        else:
            hit_rates.append(0)
            counts.append(0)
    
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    bars = ax1.bar(conf_levels, hit_rates, alpha=0.8, color=colors, edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', linewidth=2, label='Target')
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Hit Rate by Confirmation Level', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    
    for bar, hr, cnt in zip(bars, hit_rates, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, hr + 0.02,
                f'{hr:.1%}\n(n={cnt})', ha='center', fontsize=9, fontweight='bold')
    
    # Chart 2: Average returns by confidence
    ax2 = axes[0, 1]
    avg_returns = []
    for signals in conf_data:
        if signals:
            returns = [s.forward_return_1d * 100 for s in signals if s.forward_return_1d is not None]
            avg_returns.append(np.mean(returns) if returns else 0)
        else:
            avg_returns.append(0)
    
    bars = ax2.bar(conf_levels, avg_returns, alpha=0.8, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Avg Return (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Return by Confirmation', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Chart 3: Size multiplier validation
    ax3 = axes[1, 0]
    multipliers = [1.2, 0.7, 1.0]
    expected_returns = [avg * mult for avg, mult in zip(avg_returns, multipliers)]
    
    x = np.arange(len(conf_levels))
    width = 0.35
    ax3.bar(x - width/2, avg_returns, width, label='Raw Return', alpha=0.8, color='#3498db', edgecolor='black')
    ax3.bar(x + width/2, expected_returns, width, label='Size-Adjusted', alpha=0.8, color='#e67e22', edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Size Multiplier Impact (1.2x / 0.7x / 1.0x)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(conf_levels)
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # Chart 4: Distribution comparison
    ax4 = axes[1, 1]
    for signals, color, label in zip(conf_data, colors, ['High Conf', 'Low Conf', 'Neutral']):
        if signals:
            returns = [s.forward_return_1d * 100 for s in signals if s.forward_return_1d is not None]
            if returns:
                ax4.hist(returns, bins=20, alpha=0.5, label=label, color=color, edgecolor='black')
    
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('1D Return (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Return Distribution by Confidence', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"confirmation_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    plt.close('all')


def plot_method_comparison(
    all_signals: List[SignalResult],
    output_dir: Path
):
    """Compare Tier-1 vs Next-Day Alpha performance."""
    tier1_signals = [s for s in all_signals if s.signal != 0]
    alpha_signals = [s for s in all_signals if s.alpha_signal != 0]
    
    if not tier1_signals and not alpha_signals:
        print("Insufficient signals for method comparison")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Tier-1 vs Next-Day Alpha Comparison', fontsize=16, fontweight='bold')
    
    # Chart 1: Hit rates comparison
    ax1 = axes[0, 0]
    methods = ['Tier-1\n(Old)', 'Alpha\n(New)']
    hit_rates_1d = []
    hit_rates_3d = []
    
    for signals in [tier1_signals, alpha_signals]:
        hr_1d = calculate_hit_rate(signals, 1, signal_filter=0)
        hr_3d = calculate_hit_rate(signals, 3, signal_filter=0)
        hit_rates_1d.append(hr_1d['hit_rate'] or 0)
        hit_rates_3d.append(hr_3d['hit_rate'] or 0)
    
    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, hit_rates_1d, width, label='1D', alpha=0.8, color='#3498db', edgecolor='black')
    ax1.bar(x + width/2, hit_rates_3d, width, label='3D', alpha=0.8, color='#e67e22', edgecolor='black')
    ax1.axhline(y=0.52, color='green', linestyle='--', linewidth=2)
    ax1.set_ylabel('Hit Rate', fontsize=11, fontweight='bold')
    ax1.set_title('Hit Rate Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 0.8])
    
    # Chart 2: Signal counts
    ax2 = axes[0, 1]
    counts = [len(tier1_signals), len(alpha_signals)]
    ax2.bar(methods, counts, alpha=0.8, color=['#9b59b6', '#1abc9c'], edgecolor='black')
    ax2.set_ylabel('Signal Count', fontsize=11, fontweight='bold')
    ax2.set_title('Total Signals Generated', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    for i, v in enumerate(counts):
        ax2.text(i, v + 5, str(v), ha='center', fontsize=10, fontweight='bold')
    
    # Chart 3: Agreement matrix
    ax3 = axes[0, 2]
    agreement_matrix = np.zeros((3, 3))  # -1, 0, 1 for both
    
    for s in all_signals:
        tier1_idx = s.signal + 1  # Map -1,0,1 to 0,1,2
        alpha_idx = s.alpha_signal + 1
        agreement_matrix[tier1_idx, alpha_idx] += 1
    
    im = ax3.imshow(agreement_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks([0, 1, 2])
    ax3.set_yticks([0, 1, 2])
    ax3.set_xticklabels(['SHORT', 'HOLD', 'LONG'])
    ax3.set_yticklabels(['SELL', 'NEUTRAL', 'BUY'])
    ax3.set_xlabel('Alpha Decision', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Tier-1 Signal', fontsize=11, fontweight='bold')
    ax3.set_title('Agreement Matrix', fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            text = ax3.text(j, i, int(agreement_matrix[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Chart 4: Return distributions (Tier-1)
    ax4 = axes[1, 0]
    tier1_returns = [s.forward_return_1d * 100 for s in tier1_signals if s.forward_return_1d is not None]
    if tier1_returns:
        ax4.hist(tier1_returns, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax4.axvline(x=np.mean(tier1_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(tier1_returns):.2f}%')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('1D Return (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Tier-1 Return Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Chart 5: Return distributions (Alpha)
    ax5 = axes[1, 1]
    alpha_returns = [s.forward_return_1d * 100 for s in alpha_signals if s.forward_return_1d is not None]
    if alpha_returns:
        ax5.hist(alpha_returns, bins=30, alpha=0.7, color='#1abc9c', edgecolor='black')
        ax5.axvline(x=np.mean(alpha_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alpha_returns):.2f}%')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('1D Return (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Next-Day Alpha Return Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Chart 6: Sharpe comparison
    ax6 = axes[1, 2]
    sharpes = []
    for returns in [tier1_returns, alpha_returns]:
        if returns and len(returns) > 5:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            sharpes.append(sharpe)
        else:
            sharpes.append(0)
    
    bars = ax6.bar(methods, sharpes, alpha=0.8, color=['#9b59b6', '#1abc9c'], edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_ylabel('Sharpe Ratio (Annualized)', fontsize=11, fontweight='bold')
    ax6.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')
    
    for bar, sh in zip(bars, sharpes):
        ax6.text(bar.get_x() + bar.get_width()/2, sh + 0.1 if sh > 0 else sh - 0.2,
                f'{sh:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"method_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    plt.close('all')


def plot_baseline_comparison(
    all_signals: List[SignalResult],
    stock_hist: Dict[str, pd.DataFrame],
    output_dir: Path,
    execution_model: str = "next_open_to_close"
):
    """
    Compare signal performance vs market baseline (natural up/down day frequency).
    Shows if signals are actually beating random chance.
    """
    if not all_signals:
        print("No signals for baseline comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Signal Performance vs Market Baseline', fontsize=16, fontweight='bold')
    
    # Calculate baseline stats for each stock
    baseline_stats = {}
    for symbol, hist in stock_hist.items():
        if hist is not None and len(hist) > 0:
            # Calculate daily forward returns using same method as signals
            daily_returns = []
            for date in hist.index[:-1]:  # Exclude last date since we need forward returns
                ret = calculate_forward_return_label(hist, date, horizon=1, execution_model=execution_model)
                if ret is not None:
                    daily_returns.append(ret)
            
            daily_returns = np.array(daily_returns)
            baseline_stats[symbol] = {
                'up_pct': (daily_returns > 0).sum() / len(daily_returns),
                'down_pct': (daily_returns < 0).sum() / len(daily_returns),
                'avg_up': daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0,
                'avg_down': daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0,
                'total_days': len(daily_returns)
            }
    
    # Aggregate baseline (weighted by number of days)
    total_days = sum(s['total_days'] for s in baseline_stats.values())
    agg_up_pct = sum(s['up_pct'] * s['total_days'] for s in baseline_stats.values()) / total_days if total_days > 0 else 0
    agg_down_pct = sum(s['down_pct'] * s['total_days'] for s in baseline_stats.values()) / total_days if total_days > 0 else 0
    
    # Calculate signal performance
    long_signals = [s for s in all_signals if s.alpha_signal == 1 and s.forward_return_1d is not None]
    short_signals = [s for s in all_signals if s.alpha_signal == -1 and s.forward_return_1d is not None]
    
    long_hr = (sum(1 for s in long_signals if s.forward_return_1d > 0) / len(long_signals)) if long_signals else 0
    short_hr = (sum(1 for s in short_signals if s.forward_return_1d * -1 > 0) / len(short_signals)) if short_signals else 0  # Flip for shorts
    
    # Chart 1: Up/Down Day Frequency
    ax1 = axes[0, 0]
    x = np.arange(2)
    width = 0.35
    
    baseline_vals = [agg_up_pct * 100, agg_down_pct * 100]
    signal_vals = [long_hr * 100, short_hr * 100]
    
    bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Market Baseline', alpha=0.8, color='#95a5a6', edgecolor='black')
    bars2 = ax1.bar(x + width/2, signal_vals, width, label='Signal Hit Rate', alpha=0.8, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    
    ax1.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Signal Hit Rate vs Market Baseline', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['LONG\n(vs Up Days)', 'SHORT\n(vs Down Days)'])
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                    f'{height:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Add edge calculation
    long_edge = signal_vals[0] - baseline_vals[0]
    short_edge = signal_vals[1] - baseline_vals[1]
    ax1.text(0.02, 0.98, f'LONG Edge: {long_edge:+.1f}%\nSHORT Edge: {short_edge:+.1f}%',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Chart 2: Signal Count vs Available Opportunities
    ax2 = axes[0, 1]
    
    total_up_days = sum(s['total_days'] * s['up_pct'] for s in baseline_stats.values())
    total_down_days = sum(s['total_days'] * s['down_pct'] for s in baseline_stats.values())
    
    opportunity_counts = [total_up_days, total_down_days]
    signal_counts = [len(long_signals), len(short_signals)]
    
    x = np.arange(2)
    bars1 = ax2.bar(x - width/2, opportunity_counts, width, label='Available Opportunities', alpha=0.8, color='#3498db', edgecolor='black')
    bars2 = ax2.bar(x + width/2, signal_counts, width, label='Signals Fired', alpha=0.8, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Signal Selectivity', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['LONG\nOpportunities', 'SHORT\nOpportunities'])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(opportunity_counts)*0.02,
                    f'{int(height)}', ha='center', fontsize=9, fontweight='bold')
    
    # Add selectivity %
    long_sel = (len(long_signals) / total_up_days * 100) if total_up_days > 0 else 0
    short_sel = (len(short_signals) / total_down_days * 100) if total_down_days > 0 else 0
    ax2.text(0.02, 0.98, f'LONG Selectivity: {long_sel:.1f}%\nSHORT Selectivity: {short_sel:.1f}%',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Chart 3: Average Return Comparison
    ax3 = axes[1, 0]
    
    avg_long_ret = np.mean([s.forward_return_1d * 100 for s in long_signals]) if long_signals else 0
    avg_short_ret = np.mean([s.forward_return_1d * -100 for s in short_signals]) if short_signals else 0  # Flip for shorts
    
    avg_up_day = sum(s['avg_up'] * s['total_days'] * s['up_pct'] for s in baseline_stats.values()) / sum(s['total_days'] * s['up_pct'] for s in baseline_stats.values()) * 100 if baseline_stats else 0
    avg_down_day = sum(s['avg_down'] * s['total_days'] * s['down_pct'] for s in baseline_stats.values()) / sum(s['total_days'] * s['down_pct'] for s in baseline_stats.values()) * 100 if baseline_stats else 0
    
    baseline_rets = [avg_up_day, abs(avg_down_day)]  # Make down day positive for comparison
    signal_rets = [avg_long_ret, avg_short_ret]
    
    x = np.arange(2)
    bars1 = ax3.bar(x - width/2, baseline_rets, width, label='Market Baseline', alpha=0.8, color='#95a5a6', edgecolor='black')
    bars2 = ax3.bar(x + width/2, signal_rets, width, label='Signal Avg Return', alpha=0.8, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Average Return (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Return Magnitude: Signals vs Baseline', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['LONG', 'SHORT'])
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05 if height > 0 else height - 0.15,
                    f'{height:.2f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Chart 4: Expectancy Comparison
    ax4 = axes[1, 1]
    
    # Expectancy = Hit Rate * Avg Win - (1 - Hit Rate) * Avg Loss
    baseline_long_exp = (agg_up_pct * avg_up_day) * 100
    baseline_short_exp = (agg_down_pct * abs(avg_down_day)) * 100
    
    signal_long_exp = avg_long_ret * (long_hr / 100) if long_hr > 0 else 0
    signal_short_exp = avg_short_ret * (short_hr / 100) if short_hr > 0 else 0
    
    baseline_exps = [baseline_long_exp, baseline_short_exp]
    signal_exps = [signal_long_exp, signal_short_exp]
    
    x = np.arange(2)
    bars1 = ax4.bar(x - width/2, baseline_exps, width, label='Market Baseline', alpha=0.8, color='#95a5a6', edgecolor='black')
    bars2 = ax4.bar(x + width/2, signal_exps, width, label='Signal Expectancy', alpha=0.8, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Expected Return Per Trade (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Expectancy: Signals vs Random Entry', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['LONG', 'SHORT'])
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height > 0 else 'red'
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height > 0 else height - 0.08,
                    f'{height:.2f}%', ha='center', fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"baseline_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE vs SIGNAL COMPARISON")
    print("="*60)
    print(f"Market Baseline:")
    print(f"  Up Days:   {agg_up_pct*100:.1f}% (avg: {avg_up_day:+.2f}%)")
    print(f"  Down Days: {agg_down_pct*100:.1f}% (avg: {avg_down_day:.2f}%)")
    print(f"\nSignal Performance:")
    print(f"  LONG:  {long_hr*100:.1f}% hit rate, {avg_long_ret:+.2f}% avg (Edge: {long_edge:+.1f}%)")
    print(f"  SHORT: {short_hr*100:.1f}% hit rate, {avg_short_ret:+.2f}% avg (Edge: {short_edge:+.1f}%)")
    print(f"\nExpectancy:")
    print(f"  LONG:  Signal={signal_long_exp:+.2f}% vs Random={baseline_long_exp:+.2f}%")
    print(f"  SHORT: Signal={signal_short_exp:+.2f}% vs Random={baseline_short_exp:+.2f}%")
    
    plt.close('all')


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Run hit rate validation on mini-universe - pure signals only."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Hit Rate Validator for Technical Indicators")
    parser.add_argument(
        "--execution-model",
        choices=["close_to_close", "next_open_to_close", "next_open_to_next_open"],
        default="next_open_to_close",
        help="How to execute trades for forward return calculation"
    )
    parser.add_argument(
        "--alpha-threshold-mode",
        choices=["fixed", "quantile"],
        default="quantile",
        help="How to determine alpha signal thresholds"
    )
    parser.add_argument(
        "--q-long",
        type=float,
        default=80.0,
        help="Long threshold quantile percentile (default: 80th percentile)"
    )
    parser.add_argument(
        "--q-short", 
        type=float,
        default=20.0,
        help="Short threshold quantile percentile (default: 20th percentile)"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=MINI_UNIVERSE,
        help="Stock symbols to validate (default: MINI_UNIVERSE)"
    )
    parser.add_argument(
        "--start-date",
        default="1996-01-01",
        help="Start date for validation"
    )
    parser.add_argument(
        "--end-date", 
        default="2025-01-01",
        help="End date for validation"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PURE SIGNAL HIT RATE VALIDATION")
    print(f"Execution Model: {args.execution_model}")
    print(f"Alpha Threshold Mode: {args.alpha_threshold_mode}")
    if args.alpha_threshold_mode == "quantile":
        print(f"Quantile Thresholds: Long={args.q_long}th percentile, Short={args.q_short}th percentile")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print("="*80)
    
    results, all_signals, stock_results, stock_hist = await validate_mini_universe(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        execution_model=args.execution_model,
        alpha_threshold_mode=args.alpha_threshold_mode,
        q_long=args.q_long,
        q_short=args.q_short
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for horizon in ['1d', '3d', '5d']:
        print(f"\n--- {horizon.upper()} ---")
        
        # Tier-1 (old method)
        tier1_hr = results['aggregate_results'][horizon]['tier1']['all']
        if tier1_hr['hit_rate']:
            print(f"  TIER-1:     {tier1_hr['hit_rate']:.2%} ({tier1_hr['valid_signals']} signals, avg={tier1_hr['avg_return']:.2%})")
        else:
            print(f"  TIER-1:     N/A")
        
        # Alpha (new method)
        alpha_hr = results['aggregate_results'][horizon]['alpha']['all']
        if alpha_hr['hit_rate']:
            print(f"  NEXT-DAY α: {alpha_hr['hit_rate']:.2%} ({alpha_hr['valid_signals']} signals, avg={alpha_hr['avg_return']:.2%})")
        else:
            print(f"  NEXT-DAY α: N/A")
    
    print(f"\nTARGET: 52% hit rate")
    
    # Check both methods
    tier1_1d = results['aggregate_results']['1d']['tier1']['all']['hit_rate']
    alpha_1d = results['aggregate_results']['1d']['alpha']['all']['hit_rate']
    
    print(f"\n1-Day Performance:")
    print(f"  TIER-1 (old):     {'✓ PASSED' if (tier1_1d and tier1_1d >= 0.52) else '✗ BELOW THRESHOLD'}")
    print(f"  NEXT-DAY α (new): {'✓ PASSED' if (alpha_1d and alpha_1d >= 0.52) else '✗ BELOW THRESHOLD'}")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate chart for EACH stock
    for symbol, signals in stock_results.items():
        hist = stock_hist.get(symbol, pd.DataFrame())
        plot_stock_analysis(symbol, signals, hist, output_dir)
    
    # Generate aggregate analysis
    plot_aggregate_analysis(results, all_signals, stock_results, output_dir)
    
    # Generate factor correlation summary
    plot_factor_analysis(all_signals, output_dir)
    
    # Generate detailed individual factor charts
    plot_individual_factor_charts(all_signals, output_dir)
    
    # Generate quintile analysis
    plot_quintile_analysis(all_signals, output_dir)
    
    # NEW: Regime performance analysis
    plot_regime_performance(all_signals, output_dir)
    
    # NEW: Tier1-Alpha confirmation analysis
    plot_confirmation_analysis(all_signals, output_dir)
    
    # NEW: Method comparison (Tier-1 vs Alpha)
    plot_method_comparison(all_signals, output_dir)
    
    # NEW: Baseline vs Signal comparison (are we beating random?)
    plot_baseline_comparison(all_signals, stock_hist, output_dir, args.execution_model)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
