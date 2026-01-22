"""
Time-Series Diagnostic Backtest for Single Stock
=================================================

- LLM confidence does NOT affect the signal score
- Confidence is tracked as METADATA ONLY for post-hoc analysis
- Signal = what we believe (direction/conviction/ranking)
- Confidence = how certain we are (separate, testable hypothesis)

KEY PRINCIPLE: Signal = alpha (directional conviction)
               Confidence = meta-information (testable, not assumed)
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from technical_indicators_calculator import TechnicalIndicatorCalculator
from technical_json_pipeline import fetch_tiingo_ohlcv
from composite_scores import (
    calculate_momentum_composite,
    calculate_trend_composite,
    calculate_volume_confirmation_composite,
    calculate_risk_composite,
    calculate_relative_strength_composite
)
from deterministic_scoring import calculate_tier1_master_score
from technical_llm import run_technical_llm
from signal_fusion import fuse_signals
from regime_classifier import classify_market_regime


def get_weekly_rebalance_dates(hist: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract weekly rebalance dates from historical data.
    Uses last trading day of each week (typically Friday).
    
    Args:
        hist: DataFrame with DatetimeIndex
        
    Returns:
        DatetimeIndex of weekly dates
    """
    # Resample to weekly frequency, taking last trading day of each week
    weekly = hist.resample('W-FRI').last()

    weekly = weekly.dropna(subset=['Close'])
    return weekly.index


def get_price_at_date(hist: pd.DataFrame, target_date: pd.Timestamp) -> Optional[float]:
    """
    Get close price at or nearest to target date (forward fill).
    
    Args:
        hist: DataFrame with DatetimeIndex and 'Close' column
        target_date: Target date to get price for
        
    Returns:
        Close price or None if not found
    """
    if target_date in hist.index:
        return hist.loc[target_date, 'Close']
    
    # Find nearest date on or before target (no lookahead)
    valid_dates = hist.index[hist.index <= target_date]
    if len(valid_dates) == 0:
        return None
    
    nearest_date = valid_dates[-1]
    return hist.loc[nearest_date, 'Close']


def calculate_indicators_as_of(symbol: str, asof_date: pd.Timestamp, hist: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Calculate technical indicators using ONLY data up to DAY BEFORE asof_date.
    
    This simulates the realistic scenario:
    - Thursday's data is available by Thursday 8 PM
    - We generate signal Friday morning using Thursday's data
    - We execute at Friday's close via MOC order
    
    Args:
        symbol: Stock ticker
        asof_date: Target date (e.g., Friday) - we use data UP TO day before this
        hist: Full historical data (will be sliced)
        
    Returns:
        Indicators dict or None if insufficient data
    """
    # Use data UP TO day before target (Thursday for Friday execution)
    previous_day = asof_date - timedelta(days=1)
    hist_slice = hist[hist.index <= previous_day].copy()
    
    if len(hist_slice) < 252:
        return None
    
    # Get benchmark data (SPY) for same period (up to day before)
    lookback_start = (asof_date - timedelta(days=365*2)).strftime('%Y-%m-%d')
    previous_day_str = previous_day.strftime('%Y-%m-%d')
    
    spy_hist = fetch_tiingo_ohlcv('SPY', start=lookback_start, end=previous_day_str)
    benchmark_prices = spy_hist['Close'].values if spy_hist is not None and not spy_hist.empty else None
    benchmark_returns = spy_hist['Close'].pct_change().values if spy_hist is not None and not spy_hist.empty else None
    
    calc = TechnicalIndicatorCalculator(hist_slice)
    
    indicators = calc.calculate_all_indicators(
        symbol=symbol,
        sector="Unknown",
        industry="Unknown",
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns
    )
    
    return indicators


async def generate_signal_at_date(symbol: str, asof_date: pd.Timestamp, hist: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate fused_signal using data up to DAY BEFORE asof_date.
    
    Simulates: Use Thursday's data to generate signal, execute Friday via MOC.
    
    Args:
        symbol: Stock ticker
        asof_date: Target execution date (e.g., Friday)
        hist: Full historical data (will be sliced to exclude asof_date)
        
    Returns:
        Dictionary with signal details or None if generation fails
    """
    try:
        indicators = calculate_indicators_as_of(symbol, asof_date, hist)
        
        if indicators is None:
            return None
        
        # Calculate composites
        momentum_comp = calculate_momentum_composite(indicators)
        trend_comp = calculate_trend_composite(indicators)
        volume_comp = calculate_volume_confirmation_composite(indicators)
        risk_comp = calculate_risk_composite(indicators)
        relative_comp = calculate_relative_strength_composite(indicators)
        
        composite_scores_dict = {
            "momentum_composite": momentum_comp,
            "trend_composite": trend_comp,
            "volume_composite": volume_comp,
            "risk_composite": risk_comp,
            "rs_composite": relative_comp
        }
        
        regime = classify_market_regime(indicators)
        deterministic_result = calculate_tier1_master_score(composite_scores_dict, regime)
        llm_result = await run_technical_llm(indicators, mask_date=True)
        fusion_result = fuse_signals(tier1_result=deterministic_result, llm_result=llm_result)
        
        # Return both signal and confidence for tracking
        return {
            'signal': fusion_result.get("signal_final", 0.0),
            'confidence': fusion_result.get("confidence_final", 0.5),
            'llm_signal': llm_result.get('score', 0.0),
            'llm_confidence': llm_result.get('confidence', 0.5),
            'llm_tone': llm_result.get('tone', 'neutral'),
            'llm_summary': llm_result.get('summary', ''),
            'llm_action': llm_result.get('suggested_action', 'hold'),
            'deterministic_signal': deterministic_result.get('tier1_score', 0.0),
            'deterministic_regime': deterministic_result.get('regime', 'unknown'),
            'deterministic_confidence': deterministic_result.get('confidence', 0.0),
            # Composite score breakdown
            'comp_momentum': momentum_comp,
            'comp_trend': trend_comp,
            'comp_volume': volume_comp,
            'comp_risk': risk_comp,
            'comp_rs': relative_comp
        }
        
    except Exception as e:
        print(f"  Error generating signal: {e}")
        return None


async def run_time_series_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = "backtest_results",
    vol_target: Optional[float] = None,  
    transaction_cost_bps: float = 0.0,
    use_cache: bool = False
) -> Optional[pd.DataFrame]:
    """
    Time-series diagnostic backtest with weekly rebalancing and same-day execution.
    
    Execution Model:
    - Signal generated using data up to Thursday (day before Friday)
    - Position taken at Friday's close (via MOC order simulation)
    - Forward return from Friday to next Friday applied to this position
    
    This avoids lookahead bias while simulating realistic same-day execution.
    
    Args:
        symbol: Stock ticker
        start_date: Backtest start date
        end_date: Backtest end date
        output_dir: Directory to save results
        vol_target: Optional annual volatility target for position scaling
        transaction_cost_bps: Transaction cost in basis points
        use_cache: If True, load cached signals from previous run (fast)
        
    Returns:
        DataFrame with weekly results or None if failed
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Try loading cached signals first
    cache_file = output_path / f"{symbol}_signals_cache.pkl"
    
    if use_cache and cache_file.exists():
        print(f"\n{'='*60}")
        print(f"LOADING CACHED SIGNALS: {symbol}")
        print(f"{'='*60}\n")
        
        df = pd.read_pickle(cache_file)
        print(f"✓ Loaded {len(df)} cached weekly signals")
        print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Filter to requested date range if different
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        print(f"✓ Filtered to {len(df)} weeks in requested range\n")
        
        if len(df) < 2:
            print("❌ Insufficient data in requested range")
            return None
        
        # Re-apply position sizing with current parameters
        print(f"Applying position sizing...")
        print(f"  Vol Target: {vol_target*100:.0f}%" if vol_target else "  Vol Target: None")
        print(f"  Transaction Cost: {transaction_cost_bps:.1f} bps\n")
        
        # Position same day (signal uses Thursday data, executes Friday)
        # HOLD (signal=0) maintains previous position via forward-fill
        df['position'] = df['signal_clipped'].replace(0, np.nan).ffill().fillna(0.0)
        
        # Optional: Vol targeting
        if vol_target is not None and 'realized_vol' in df.columns:
            df['vol_scalar'] = vol_target / df['realized_vol']
            df['vol_scalar'] = df['vol_scalar'].clip(0, 2)
            df['position'] = df['position'] * df['vol_scalar']
        
        # Recalculate turnover and costs
        df['position_prev'] = df['position'].shift(1)
        df['turnover'] = (df['position'] - df['position_prev']).abs()
        df['tc_cost'] = df['turnover'] * (transaction_cost_bps / 10000)
        df['strategy_return'] = df['position'] * df['forward_return'] - df['tc_cost']
        df['buy_hold_return'] = df['forward_return']
        
        # Drop last row (no forward return available)
        df = df[:-1].copy()
        
        if len(df) == 0:
            print("❌ No valid backtest results")
            return None
        
        # Save results
        output_file = output_path / f"{symbol}_backtest.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to {output_file}")
        
        return df
    
    # === GENERATE NEW SIGNALS ===
    
    print(f"\n{'='*60}")
    print(f"TIME-SERIES DIAGNOSTIC BACKTEST: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Vol Target: {vol_target*100:.0f}%" if vol_target else "Vol Target: None")
    print(f"Transaction Cost: {transaction_cost_bps:.1f} bps")
    print(f"{'='*60}\n")
    
    # Fetch full history (extended for warmup period)
    extended_start = (pd.to_datetime(start_date) - timedelta(days=365*2)).strftime('%Y-%m-%d')
    extended_end = (pd.to_datetime(end_date) + timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Fetching price data from {extended_start} to {extended_end}...")
    hist = fetch_tiingo_ohlcv(symbol, start=extended_start, end=extended_end)
    
    if hist is None or hist.empty:
        print("❌ Failed to fetch historical data")
        return None
    
    print(f"✓ Loaded {len(hist)} daily prices")
    
    # Get weekly rebalance dates
    weekly_dates = get_weekly_rebalance_dates(hist)
    
    # Filter to backtest window
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    weekly_dates = weekly_dates[(weekly_dates >= start_dt) & (weekly_dates <= end_dt)]
    
    if len(weekly_dates) < 2:
        print("❌ Insufficient weekly dates in backtest period")
        return None
    
    print(f"✓ Found {len(weekly_dates)} weekly rebalance dates\n")
    
    # Generate signals for each week-end
    signal_data = []
    for i, week_date in enumerate(weekly_dates, 1):
        print(f"[{i}/{len(weekly_dates)}] {week_date.strftime('%Y-%m-%d')}", end=" ")
        
        result = await generate_signal_at_date(symbol, week_date, hist)
        
        if result is None:
            print("❌ Signal generation failed")
            signal_data.append({
                'date': week_date,
                'fused_signal': None,
                'confidence': None,
                'llm_signal': None,
                'llm_confidence': None,
                'llm_tone': None,
                'llm_summary': None,
                'llm_action': None,
                'deterministic_signal': None,
                'deterministic_regime': None,
                'deterministic_confidence': None,
                'comp_momentum': None,
                'comp_trend': None,
                'comp_volume': None,
                'comp_risk': None,
                'comp_rs': None
            })
        else:
            signal = result['signal']
            print(f"✓ Signal: {signal:+.3f}")
            signal_data.append({
                'date': week_date,
                'fused_signal': signal,
                'confidence': result['confidence'],
                'llm_signal': result['llm_signal'],
                'llm_confidence': result['llm_confidence'],
                'llm_tone': result['llm_tone'],
                'llm_summary': result['llm_summary'],
                'llm_action': result['llm_action'],
                'deterministic_signal': result['deterministic_signal'],
                'deterministic_regime': result['deterministic_regime'],
                'deterministic_confidence': result['deterministic_confidence'],
                'comp_momentum': result['comp_momentum'],
                'comp_trend': result['comp_trend'],
                'comp_volume': result['comp_volume'],
                'comp_risk': result['comp_risk'],
                'comp_rs': result['comp_rs']
            })
    
    # Build weekly time series
    df = pd.DataFrame(signal_data)
    
    # Drop rows with missing signals
    df = df.dropna(subset=['fused_signal']).copy()
    
    if len(df) < 2:
        print("\n❌ Insufficient valid signals")
        return None
    
    print(f"\n✓ Generated {len(df)} valid signals")
    
    # Get weekly prices
    df['price'] = df['date'].apply(lambda d: get_price_at_date(hist, d))
    df = df.dropna(subset=['price']).copy()
    
    # Calculate forward weekly returns: ret[t+1] = price[t+1]/price[t] - 1
    df['forward_return'] = df['price'].pct_change().shift(-1)
    
    # Clip signals to [-1, 1]
    df['signal_clipped'] = df['fused_signal'].clip(-1, 1)
    
    # Calculate rolling realized volatility for cache (even if not used now)
    daily_returns = hist['Close'].pct_change()
    rolling_vol = []
    for week_date in df['date']:
        past_returns = daily_returns[daily_returns.index <= week_date].tail(63)
        if len(past_returns) >= 20:
            realized_vol = past_returns.std() * np.sqrt(252)
            rolling_vol.append(realized_vol)
        else:
            rolling_vol.append(None)
    df['realized_vol'] = rolling_vol
    
    # === SAVE CACHE (before position sizing) ===
    # Include confidence as metadata for post-hoc analysis
    cache_columns = ['date', 'fused_signal', 'signal_clipped', 'price', 'forward_return', 'realized_vol',
                     'confidence', 'llm_signal', 'llm_confidence', 'llm_tone', 'llm_summary', 'llm_action',
                     'deterministic_signal', 'deterministic_regime', 'deterministic_confidence',
                     'comp_momentum', 'comp_trend', 'comp_volume', 'comp_risk', 'comp_rs']
    df[cache_columns].to_pickle(cache_file)
    print(f"\n✓ Cached signals saved to {cache_file}")
    print(f"  (Rerun with --use-cache to skip signal generation)\n")
    
    # Position at week t uses signal from week t (same-day execution via MOC)
    # Signal uses data up to Thursday, executes at Friday close
    # HOLD (signal=0) maintains previous position via forward-fill
    df['position'] = df['signal_clipped'].replace(0, np.nan).ffill().fillna(0.0)
    
    # Optional: Vol targeting
    if vol_target is not None:
        df['vol_scalar'] = vol_target / df['realized_vol']
        df['vol_scalar'] = df['vol_scalar'].clip(0, 2)  # Cap at 2x
        df['position'] = df['position'] * df['vol_scalar']
    
    # Calculate turnover
    df['position_prev'] = df['position'].shift(1)
    df['turnover'] = (df['position'] - df['position_prev']).abs()
    
    # Transaction costs
    df['tc_cost'] = df['turnover'] * (transaction_cost_bps / 10000)
    
    # Strategy return: strat_ret[t+1] = pos[t+1] * ret_fwd[t+1] - tc
    df['strategy_return'] = df['position'] * df['forward_return'] - df['tc_cost']
    
    # Buy-and-hold return (same forward returns)
    df['buy_hold_return'] = df['forward_return']
    
    # Drop last row (no forward return available)
    df = df[:-1].copy()
    
    if len(df) == 0:
        print("\n❌ No valid backtest results")
        return None
    
    # Save results
    output_file = output_path / f"{symbol}_backtest.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    return df


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from return series."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()


def analyze_results(df: pd.DataFrame, symbol: str, output_dir: str = "backtest_results"):
    """
    Comprehensive diagnostic analysis of backtest results.
    
    Outputs:
    - Performance metrics (CAGR, Sharpe, max drawdown)
    - Information Coefficient (IC) between signal and forward returns
    - Bucket test for monotonicity
    - Turnover analysis
    - Equity curves
    """
    print(f"\n{'='*60}")
    print(f"TIME-SERIES DIAGNOSTIC ANALYSIS: {symbol}")
    print(f"{'='*60}\n")
    
    n_weeks = len(df)
    n_years = n_weeks / 52
    
    # === PERFORMANCE METRICS ===
    
    # Strategy
    cum_strat = (1 + df['strategy_return']).cumprod()
    total_strat_ret = cum_strat.iloc[-1] - 1
    cagr_strat = (cum_strat.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
    vol_strat = df['strategy_return'].std() * np.sqrt(52)
    sharpe_strat = (df['strategy_return'].mean() * 52) / vol_strat if vol_strat > 0 else 0
    max_dd_strat = calculate_max_drawdown(df['strategy_return'])
    
    # Buy & Hold
    cum_bh = (1 + df['buy_hold_return']).cumprod()
    total_bh_ret = cum_bh.iloc[-1] - 1
    cagr_bh = (cum_bh.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
    vol_bh = df['buy_hold_return'].std() * np.sqrt(52)
    sharpe_bh = (df['buy_hold_return'].mean() * 52) / vol_bh if vol_bh > 0 else 0
    max_dd_bh = calculate_max_drawdown(df['buy_hold_return'])
    
    # Turnover
    avg_turnover = df['turnover'].mean()
    annual_turnover = avg_turnover * 52
    
    # === INFORMATION COEFFICIENT (IC) ===
    # Correlation between signal[t] and forward_return[t+1]
    # Note: position is signal shifted, so we use signal_clipped
    valid_pairs = df[['signal_clipped', 'forward_return']].dropna()
    
    if len(valid_pairs) > 3:
        ic_corr, ic_pval = pearsonr(valid_pairs['signal_clipped'], valid_pairs['forward_return'])
    else:
        ic_corr, ic_pval = 0, 1    
    # === HIT RATE CALCULATION ===
    # Calculate hit rate: % of signals with positive forward returns
    # Buy signals: signal_clipped > 0
    buy_signals = df[df['signal_clipped'] > 0].copy()
    buy_positive = (buy_signals['forward_return'] > 0).sum()
    buy_total = len(buy_signals)
    buy_hit_rate = buy_positive / buy_total if buy_total > 0 else 0
    
    # Sell signals: signal_clipped < 0 (success = negative return)
    sell_signals = df[df['signal_clipped'] < 0].copy()
    sell_positive = (sell_signals['forward_return'] < 0).sum()
    sell_total = len(sell_signals)
    sell_hit_rate = sell_positive / sell_total if sell_total > 0 else 0
    
    # All directional signals (non-zero)
    directional_signals = df[df['signal_clipped'] != 0].copy()
    # For long signals, positive return is success; for short, negative return is success
    directional_signals['hit'] = (
        (directional_signals['signal_clipped'] > 0) & (directional_signals['forward_return'] > 0)
    ) | (
        (directional_signals['signal_clipped'] < 0) & (directional_signals['forward_return'] < 0)
    )
    all_hit_rate = directional_signals['hit'].sum() / len(directional_signals) if len(directional_signals) > 0 else 0    
    # === BUCKET TEST ===
    # Average forward return by signal strength bins
    df['signal_abs'] = df['signal_clipped'].abs()
    
    # Define 5 bins by absolute signal strength
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    df['signal_bucket'] = pd.cut(df['signal_abs'], bins=bins, labels=labels, include_lowest=True)
    bucket_stats = df.groupby('signal_bucket', observed=True).agg({
        'forward_return': ['mean', 'count']
    }).round(4)
    
    # === PRINT DIAGNOSTICS ===
    
    print(f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Weeks: {n_weeks} ({n_years:.1f} years)")
    print(f"\n--- RETURNS ---")
    print(f"Strategy Total:  {total_strat_ret:+.1%}")
    print(f"Buy & Hold Total: {total_bh_ret:+.1%}")
    print(f"Outperformance:   {(total_strat_ret - total_bh_ret):+.1%}")
    
    print(f"\n--- ANNUALIZED METRICS ---")
    print(f"Strategy CAGR:    {cagr_strat:+.1%}")
    print(f"Strategy Vol:     {vol_strat:.1%}")
    print(f"Strategy Sharpe:  {sharpe_strat:.2f}")
    print(f"Strategy Max DD:  {max_dd_strat:.1%}")
    
    print(f"\nBuy & Hold CAGR:  {cagr_bh:+.1%}")
    print(f"Buy & Hold Vol:   {vol_bh:.1%}")
    print(f"Buy & Hold Sharpe: {sharpe_bh:.2f}")
    print(f"Buy & Hold Max DD: {max_dd_bh:.1%}")
    
    print(f"\n--- TURNOVER ---")
    print(f"Avg Weekly:       {avg_turnover:.2%}")
    print(f"Annualized:       {annual_turnover:.2f}x")
    
    print(f"\n--- INFORMATION COEFFICIENT ---")
    print(f"IC (Signal vs Fwd Return): {ic_corr:+.3f}")
    print(f"P-value:                   {ic_pval:.4f}")
    print(f"Significance: {'***' if ic_pval < 0.01 else '**' if ic_pval < 0.05 else '*' if ic_pval < 0.1 else 'ns'}")
    
    print(f"\n--- HIT RATE (52% TARGET) ---")
    print(f"BUY Signals:  {buy_hit_rate:.1%} ({buy_positive}/{buy_total} positive returns)")
    print(f"SELL Signals: {sell_hit_rate:.1%} ({sell_positive}/{sell_total} negative returns)")
    print(f"ALL Signals:  {all_hit_rate:.1%} ({directional_signals['hit'].sum()}/{len(directional_signals)} correct)")
    print(f"Status: {'✓ PASSED' if all_hit_rate >= 0.52 else '✗ BELOW 52% THRESHOLD'}")
    
    print(f"\n--- BUCKET TEST (Monotonicity) ---")
    print("Avg Forward Return by Absolute Signal Strength:")
    print(bucket_stats.to_string())
    
    # === CONFIDENCE ANALYSIS (METADATA) ===
    if 'confidence' in df.columns and df['confidence'].notna().any():
        print(f"\n--- CONFIDENCE ANALYSIS (Metadata Only) ---")
        print(f"Avg Confidence:    {df['confidence'].mean():.3f}")
        print(f"Avg LLM Confidence: {df['llm_confidence'].mean():.3f}")
        
        # IC of confidence with abs(forward_return) - does confidence predict magnitude?
        valid_conf = df[['confidence', 'forward_return']].dropna()
        if len(valid_conf) > 3:
            conf_ic, conf_pval = pearsonr(valid_conf['confidence'], valid_conf['forward_return'].abs())
            print(f"Confidence IC (predicts |return|): {conf_ic:+.3f} (p={conf_pval:.3f})")
        
        # Performance by confidence tercile
        df['conf_tercile'] = pd.qcut(df['confidence'], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        conf_perf = df.groupby('conf_tercile', observed=True).agg({
            'forward_return': 'mean',
            'strategy_return': 'mean'
        }).round(4)
        print(f"\nAvg Returns by Confidence Tercile:")
        print(conf_perf.to_string())
        print(f"\n(Note: Confidence does NOT affect signal/position - tracking for analysis only)")
    
    # === VISUALIZATIONS ===
    
    output_path = Path(output_dir)
    
    # Create 4x2 grid to accommodate confidence chart
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'{symbol} Time-Series Diagnostic Backtest', fontsize=16, fontweight='bold')
    
    # 1. Equity Curves
    ax = axes[0, 0]
    ax.plot(df['date'], cum_strat, label=f'Strategy (Return: {total_strat_ret:+.1%})', linewidth=2)
    ax.plot(df['date'], cum_bh, label=f'Buy & Hold (Return: {total_bh_ret:+.1%})', linewidth=2, alpha=0.7)
    ax.set_title('Equity Curves')
    ax.set_ylabel('Growth of $1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Signal vs Forward Return (IC scatter)
    ax = axes[0, 1]
    ax.scatter(df['signal_clipped'], df['forward_return'], alpha=0.4, s=20)
    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Signal (week t)')
    ax.set_ylabel('Forward Return (week t+1)')
    ax.set_title(f'IC = {ic_corr:+.3f} (p={ic_pval:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax = axes[1, 0]
    cum_strat_max = cum_strat.expanding().max()
    drawdown_strat = (cum_strat - cum_strat_max) / cum_strat_max
    ax.fill_between(df['date'], drawdown_strat, 0, alpha=0.3, color='red')
    ax.plot(df['date'], drawdown_strat, linewidth=1, color='darkred')
    ax.set_title(f'Strategy Drawdown (Max: {max_dd_strat:.1%})')
    ax.set_ylabel('Drawdown')
    ax.grid(True, alpha=0.3)
    
    # 4. Bucket Test (monotonicity)
    ax = axes[1, 1]
    bucket_means = df.groupby('signal_bucket', observed=True)['forward_return'].mean()
    bucket_means.plot(kind='bar', ax=ax, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', alpha=0.3)
    ax.set_title('Bucket Test: Avg Fwd Return by |Signal|')
    ax.set_xlabel('Absolute Signal Strength')
    ax.set_ylabel('Avg Forward Return')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Signal over time
    ax = axes[2, 0]
    ax.plot(df['date'], df['signal_clipped'], linewidth=1.5, color='purple')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_title('Signal Over Time')
    ax.set_ylabel('Signal')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 6. Turnover over time
    ax = axes[2, 1]
    ax.bar(df['date'], df['turnover'], width=5, alpha=0.6, color='orange')
    ax.axhline(avg_turnover, color='red', linestyle='--', label=f'Avg: {avg_turnover:.2%}')
    ax.set_title('Weekly Turnover')
    ax.set_ylabel('Turnover')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 7. Confidence vs Magnitude (Metadata Analysis)
    ax = axes[3, 0]
    if 'confidence' in df.columns and df['confidence'].notna().any():
        df['abs_return'] = df['forward_return'].abs()
        valid_conf = df[['confidence', 'abs_return']].dropna()
        
        if len(valid_conf) > 3:
            corr_mag, p_mag = pearsonr(valid_conf['confidence'], valid_conf['abs_return'])
            ax.scatter(valid_conf['confidence'], valid_conf['abs_return'], alpha=0.4, s=20, color='purple')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Absolute Forward Return')
            ax.set_title(f'Confidence vs Return Magnitude (IC={corr_mag:+.3f}, p={p_mag:.3f})')
            
            # Add trend line
            z = np.polyfit(valid_conf['confidence'], valid_conf['abs_return'], 1)
            p = np.poly1d(z)
            ax.plot(valid_conf['confidence'], p(valid_conf['confidence']), 
                   "r--", alpha=0.5, linewidth=2, label='Trend')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add interpretation text
            interpretation = "Confidence predicts magnitude" if corr_mag > 0.1 and p_mag < 0.05 else "No predictive power"
            ax.text(0.05, 0.95, interpretation, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Insufficient confidence data', 
                   ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No confidence data available', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 8. Confidence Distribution
    ax = axes[3, 1]
    if 'confidence' in df.columns and df['confidence'].notna().any():
        ax.hist(df['confidence'].dropna(), bins=30, alpha=0.6, color='steelblue', edgecolor='black')
        ax.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {df["confidence"].mean():.2f}')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No confidence data available', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    plot_file = output_path / f"{symbol}_diagnostic_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualizations saved to {plot_file}")
    
    plt.show()


async def main():
    """
    Main entry point for time-series diagnostic backtest.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run time-series diagnostic backtest for single stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python backtest_single_stock.py --symbol TSLA --start 2020-01-01 --end 2024-12-31
  
  # With vol targeting and transaction costs
  python backtest_single_stock.py --symbol AAPL --start 2021-01-01 --vol-target 0.15 --tc-bps 5.0
        """
    )
    
    parser.add_argument("--symbol", required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="backtest_results", help="Output directory")
    parser.add_argument("--vol-target", type=float, default=None, 
                        help="Optional volatility target (e.g., 0.15 for 15%% annual vol)")
    parser.add_argument("--tc-bps", type=float, default=0.0,
                        help="Transaction cost in basis points (e.g., 5.0 for 5 bps)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached signals from previous run (fast)")
    
    args = parser.parse_args()
    
    df = await run_time_series_backtest(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        vol_target=args.vol_target,
        transaction_cost_bps=args.tc_bps,
        use_cache=args.use_cache
    )
    
    if df is not None:
        analyze_results(df, args.symbol, args.output)
    else:
        print("\n❌ Backtest failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
