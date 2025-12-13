import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_DOLLAR_VOLUME = 10_000_000
MIN_LIQUIDITY_SCORE = 50
MAX_SPREAD_PCT = 0.05
MIN_AVG_VOLUME = 100_000

def check_liquidity_gate(indicators: Dict[str, Any]) -> Dict[str, Any]:
    vol_liq = indicators.get('volume_liquidity', {})
    
    avg_dollar_vol = vol_liq.get('avg_dollar_vol_21d')
    liquidity_score = vol_liq.get('liquidity_score_0_100')
    cs_spread = vol_liq.get('cs_spread_21')
    avg_volume = vol_liq.get('avg_vol_3m')
    
    checks = {}
    failed_reasons = []
    
    if avg_dollar_vol is not None:
        checks['min_dollar_volume_filter'] = avg_dollar_vol >= MIN_DOLLAR_VOLUME
        if not checks['min_dollar_volume_filter']:
            failed_reasons.append(
                f"Dollar volume ${avg_dollar_vol/1e6:.1f}M < ${MIN_DOLLAR_VOLUME/1e6:.0f}M minimum"
            )
    else:
        checks['min_dollar_volume_filter'] = False
        failed_reasons.append("Dollar volume data missing")
    
    if liquidity_score is not None:
        checks['min_liquidity_score_filter'] = liquidity_score >= MIN_LIQUIDITY_SCORE
        if not checks['min_liquidity_score_filter']:
            failed_reasons.append(
                f"Liquidity score {liquidity_score} < {MIN_LIQUIDITY_SCORE} minimum"
            )
    else:
        checks['min_liquidity_score_filter'] = False
        failed_reasons.append("Liquidity score data missing")
    
    if cs_spread is not None:
        checks['max_spread_filter'] = cs_spread <= MAX_SPREAD_PCT
        if not checks['max_spread_filter']:
            failed_reasons.append(
                f"Spread {cs_spread*100:.2f}% > {MAX_SPREAD_PCT*100:.1f}% maximum"
            )
    else:
        checks['max_spread_filter'] = True
    
    if avg_volume is not None:
        checks['min_volume_filter'] = avg_volume >= MIN_AVG_VOLUME
        if not checks['min_volume_filter']:
            failed_reasons.append(
                f"Avg volume {avg_volume/1e6:.2f}M < {MIN_AVG_VOLUME/1e6:.2f}M minimum"
            )
    else:
        checks['min_volume_filter'] = False
        failed_reasons.append("Volume data missing")
    
    passes = all(checks.values())
    
    result = {
        "pass": passes,
        "reason": "PASS - Meets all liquidity requirements" if passes else "; ".join(failed_reasons),
        "checks": checks,
        "metrics": {
            "avg_dollar_vol_21d": avg_dollar_vol,
            "liquidity_score": liquidity_score,
            "cs_spread_21": cs_spread,
            "avg_vol_3m": avg_volume
        }
    }
    
    return result


def check_data_sanity(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Data quality check: Ensure required fields are present and reasonable.
    
    Args:
        indicators: Complete technical indicators dictionary
    
    Returns:
        Dictionary with:
        {
            "pass": bool,
            "reason": str,
            "missing_fields": list,
            "invalid_fields": list
        }
    """
    missing_fields = []
    invalid_fields = []
    
    required_sections = ['momentum', 'trend', 'volatility', 'risk', 'volume_liquidity', 'events', 'relative_perf']
    for section in required_sections:
        if section not in indicators:
            missing_fields.append(section)
    
    if 'momentum' in indicators:
        mom = indicators['momentum']
        for field in ['mom_1m', 'mom_3m', 'mom_6m']:
            if mom.get(field) is None:
                missing_fields.append(f'momentum.{field}')
    
    if 'trend' in indicators:
        trend = indicators['trend']
        for field in ['rsi_14d', 'macd_histogram', 'adx_14d']:
            if trend.get(field) is None:
                missing_fields.append(f'trend.{field}')
    
    if 'volatility' in indicators:
        vol = indicators['volatility']
        if vol.get('realized_vol_252d') is None:
            missing_fields.append('volatility.realized_vol_252d')
        
        rvol = vol.get('realized_vol_252d')
        if rvol is not None and (rvol < 0 or rvol > 5.0):
            invalid_fields.append(f'volatility.realized_vol_252d={rvol:.2f} (unreasonable)')
    
    if 'risk' in indicators:
        risk = indicators['risk']
        sharpe = risk.get('sharpe_1y')
        if sharpe is not None and (sharpe < -10 or sharpe > 10):
            invalid_fields.append(f'risk.sharpe_1y={sharpe:.2f} (unreasonable)')
    
    passes = len(missing_fields) == 0 and len(invalid_fields) == 0
    
    reason = "PASS - All data fields valid"
    if not passes:
        reason_parts = []
        if missing_fields:
            reason_parts.append(f"Missing: {', '.join(missing_fields)}")
        if invalid_fields:
            reason_parts.append(f"Invalid: {', '.join(invalid_fields)}")
        reason = "; ".join(reason_parts)
    
    return {
        "pass": passes,
        "reason": reason,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields
    }


def apply_liquidity_gate(indicators: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Complete liquidity gate + data sanity check.
    
    This is the entry point for the first-stage filter.
    
    Args:
        indicators: Complete technical indicators dictionary
        verbose: If True, log detailed results
    
    Returns:
        Dictionary with:
        {
            "pass": bool,                           # Overall pass/fail
            "liquidity_gate": dict,                 # Liquidity gate results
            "data_sanity": dict,                    # Data sanity results
            "symbol": str,                          # Stock symbol
            "reason": str                           # Human-readable reason
        }
    """
    symbol = indicators.get('id', {}).get('symbol', 'UNKNOWN')
    
    if verbose:
        logger.info(f"Applying liquidity gate to {symbol}...")
    
    liq_result = check_liquidity_gate(indicators)
    
    sanity_result = check_data_sanity(indicators)
    
    overall_pass = liq_result['pass'] and sanity_result['pass']
    
    if overall_pass:
        reason = "PASS - Meets liquidity and data quality requirements"
    else:
        reasons = []
        if not liq_result['pass']:
            reasons.append(f"Liquidity: {liq_result['reason']}")
        if not sanity_result['pass']:
            reasons.append(f"Data Quality: {sanity_result['reason']}")
        reason = "; ".join(reasons)
    
    result = {
        "pass": overall_pass,
        "liquidity_gate": liq_result,
        "data_sanity": sanity_result,
        "symbol": symbol,
        "reason": reason
    }
    
    if verbose:
        status = "✓ PASS" if overall_pass else "✗ FAIL"
        logger.info(f"{symbol}: {status}")
        if not overall_pass:
            logger.info(f"  Reason: {reason}")
        else:
            metrics = liq_result['metrics']
            logger.info(f"  Dollar Vol: ${metrics['avg_dollar_vol_21d']/1e6:.1f}M")
            logger.info(f"  Liquidity Score: {metrics['liquidity_score']}/100")
    
    return result


def filter_universe_by_liquidity(
    indicators_dict: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Apply liquidity gate to multiple stocks and return filtered universe.
    
    Args:
        indicators_dict: Dictionary mapping symbol -> indicators
        verbose: If True, log results
    
    Returns:
        {
            "passed": list,      # Symbols that passed
            "failed": list,      # Symbols that failed
            "results": dict      # Detailed results for each symbol
        }
    """
    passed = []
    failed = []
    results = {}
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"LIQUIDITY GATE FILTER: Processing {len(indicators_dict)} symbols")
        logger.info(f"{'='*80}\n")
    
    for symbol, indicators in indicators_dict.items():
        result = apply_liquidity_gate(indicators, verbose=False)
        results[symbol] = result
        
        if result['pass']:
            passed.append(symbol)
        else:
            failed.append(symbol)
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"LIQUIDITY GATE RESULTS:")
        logger.info(f"  Passed: {len(passed)}/{len(indicators_dict)}")
        logger.info(f"  Failed: {len(failed)}/{len(indicators_dict)}")
        logger.info(f"{'='*80}")
        
        if passed:
            logger.info(f"\n✓ Passed: {', '.join(passed)}")
        if failed:
            logger.info(f"\n✗ Failed: {', '.join(failed)}")
            for symbol in failed:
                logger.info(f"  {symbol}: {results[symbol]['reason']}")
    
    return {
        "passed": passed,
        "failed": failed,
        "results": results
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    output_dir = Path("./output")
    
    if not output_dir.exists():
        logger.error("Output directory not found. Run calculate_all_indicators.py first.")
        exit(1)
    
    indicators_dict = {}
    for json_file in output_dir.glob("*_technical_indicators.json"):
        with open(json_file) as f:
            data = json.load(f)
            symbol = data['id']['symbol']
            indicators_dict[symbol] = data
    
    if not indicators_dict:
        logger.error("No indicator JSON files found in output directory.")
        exit(1)
    
    filter_results = filter_universe_by_liquidity(indicators_dict, verbose=True)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Universe Size: {len(indicators_dict)}")
    print(f"Passed Gate:   {len(filter_results['passed'])}")
    print(f"Failed Gate:   {len(filter_results['failed'])}")
    print(f"Pass Rate:     {len(filter_results['passed'])/len(indicators_dict)*100:.1f}%")
    print("="*80)
