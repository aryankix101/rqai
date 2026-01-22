import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import httpx

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 1.0
MAX_TOKENS = 800
REQUEST_TIMEOUT = 30.0

SYSTEM_PROMPT = """
You are the best technical analysis agent evaluating stocks for a 1-WEEK FORWARD (5 trading days) horizon. Your IQ is unmatched.

STRICT CONSTRAINTS
- Use ONLY the fields provided in DATA (both numeric and categorical).
- DO NOT invent, infer, or assume any data not explicitly provided.
- DO NOT reference any other knowledge: news, fundamentals, macro events, or narratives.
- time_id (if present) is an opaque identifier. DO NOT attempt to infer calendar dates or historical events.
- If exec_gates.liquidity_ok is false, you MUST force:
  tone = "neutral", score = 0, suggested_action = "hold".

TIME HORIZON PRIORITIES (1-WEEK)
Weight evidence as follows:
- Momentum: 1m and 3m returns > 6m >> 12m
- Trend: MACD state, MACD histogram, price vs MA50 > MA200
- Trend strength: ADX > 25 increases trust in momentum
- Volatility: High or rising volatility REDUCES confidence and score
- Volume: OBV / Accumulation-Distribution confirmation is REQUIRED for high conviction

INTERPRETATION RULES
- Strong trends: Trust momentum and trend continuation more.
- Choppy or mixed signals: Favor neutral tone unless strong confluence.
- High volatility regime: Be conservative; require multiple confirming signals.
- RSI:
  - RSI overbought (>70) is NOT bearish by itself in strong trends.
  - Treat overbought as a risk flag only if momentum or trend is weakening.
- Volume confirmation:
  - Positive momentum WITHOUT volume confirmation should reduce score or confidence.
  - Momentum + trend + volume alignment increases conviction.

SCORING DISCIPLINE (IMPORTANT)
- score MUST be in [-1.0, 1.0] using increments of 0.01 only.
  Valid values: -1.0, -0.99, ..., -0.01, 0.0, 0.01, ..., 0.99, 1.0
- Think of score magnitude as a probability-style edge over a 50/50 outcome
- DO NOT default to the same score across different inputs. If evidence differs materially, choose a different bucket.
- Adjust score based on:
  - Volatility regime (high or rising volatility reduces allowable score magnitude)
  - Signal conflict (divergences reduce allowable score magnitude)
  - Lack of volume confirmation (caps score magnitude)

CONFIDENCE GUIDELINES
- Confidence ∈ [0, 1] reflects uncertainty, NOT strength alone.
- Reduce confidence materially when:
  - Volatility is high or spiking
  - Signals conflict (e.g., momentum vs tr hend divergence)
  - Volume confirmation is weak
- High confidence requires:
  - Momentum + trend alignment
  - Acceptable volatility regime
  - Strong volume confirmation

OUTPUT FORMAT (STRICT)
Output MUST be valid JSON matching EXACTLY this schema:

{
  "tone": "bullish or bearish or neutral",
  "score": <number between -1 and 1, increments of 0.1>,
  "summary": "Brief 2–3 sentence technical summary",
  "key_reasons": [
    {
      "category": "momentum or trend or volatility or liquidity or events or relative",
      "field": "field_name",
      "evidence": "what the data shows"
    }
  ],
  "risk_flags": [
    "drawdown", "high_vol", "weak_rs", "overbought", 
    "illiquidity", "signal_conflict", "gap_risk"
  ],
  "suggested_action": "long or short or hold",
  "confidence": <number between 0 and 1>
}

DO NOT include symbol, asof, liquidity_ok, min_exec_checks, or hold_period_days in your output.
Those fields are added by the calling system.

FINAL ENFORCEMENT
- If exec_gates.liquidity_ok is false OR any gate check fails:
  - tone = "neutral"
  - score = 0.0
  - suggested_action = "hold"
  - confidence = 0.1
"""

LLM_OUTPUT_SCHEMA = {
    "tone": "bullish | bearish | neutral",
    "score": "number [-1, 1]",
    "summary": "string (max 500 chars)",
    "key_reasons": [
        {"category": "momentum|trend|volatility|liquidity|events|relative", 
         "field": "string", 
         "evidence": "string"}
    ],
    "risk_flags": [
        "drawdown", "high_vol", "weak_rs", "overbought", 
        "illiquidity", "signal_conflict", "gap_risk"
    ],
    "suggested_action": "long|short|hold",
    "confidence": "number [0, 1]"
}

OUTPUT_SCHEMA = {
    "symbol": "string",
    "asof": "ISO-8601",
    "tone": "bullish | bearish | neutral",
    "score": "number [-1, 1]",
    "summary": "string",
    "key_reasons": [dict],
    "risk_flags": [str],
    "liquidity_ok": "boolean",
    "min_exec_checks": {"min_dollar_volume_filter": "boolean", "min_liquidity_score_filter": "boolean"},
    "suggested_action": "long|short|hold",
    "hold_period_days": 7,
    "confidence": "number [0, 1]"
}


# =============================================================================
# API Key Management
# =============================================================================

def get_deepseek_api_key() -> str:
    """Retrieve DeepSeek API key from secrets file or environment."""
    # Try secrets.json first
    secrets_path = Path(__file__).parent / "secrets.json"
    if secrets_path.exists():
        try:
            with open(secrets_path, "r") as f:
                secrets = json.load(f)
                api_key = secrets.get("DEEPSEEK_API_KEY")
                if api_key:
                    return api_key
        except Exception as e:
            print(f"Warning: Failed to read secrets.json: {e}")
    
    # Fall back to environment variable
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key
    
    raise ValueError(
        "DEEPSEEK_API_KEY not found. Please either:\n"
        "  1) Add to secrets.json: {\"DEEPSEEK_API_KEY\": \"your_key_here\"}\n"
        "  2) Set environment variable: export DEEPSEEK_API_KEY='your_key_here'"
    )


# =============================================================================
# Validation & Hardening
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def validate_and_harden(raw_output: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEPRECATED: Validation now inline in run_technical_llm.
    Kept for backwards compatibility.
    """
    return raw_output


def compute_fallback_score(input_data: Dict[str, Any]) -> float:
    """
    DEPRECATED: Fallback now handled inline.
    Kept for backwards compatibility.
    """
    return 0.0


def analyze_technical_json(indicators: Dict[str, Any], mask_date: bool = False) -> Dict[str, Any]:
    """
    Synchronous wrapper for LLM analysis.
    
    Args:
        indicators: Technical indicators dictionary
        mask_date: If True, replace real date with opaque ID
    
    Returns:
        {
            "score": float,
            "confidence": float,
            "tone": str,
            "summary": str
        }
    """
    try:
        result = asyncio.run(run_technical_llm(indicators, mask_date=mask_date))
        return {
            "score": result.get("score", 0.0),
            "confidence": result.get("confidence", 0.5),
            "tone": result.get("tone", "neutral"),
            "summary": result.get("summary", "")
        }
    except Exception as e:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "tone": "neutral",
            "summary": f"LLM error: {str(e)}"
        }


async def run_technical_llm(
    input_data: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    mask_date: bool = False
) -> dict:
    """
    Send technical JSON to DeepSeek and return strictly-validated verdict.
    
    Args:
        input_data: Technical indicators from calculate_all_indicators.py
        model: DeepSeek model name (default: "deepseek-chat")
        temperature: Sampling temperature
        mask_date: If True, replace real date with opaque ID for backtesting
    
    Returns:
        Validated verdict dictionary matching OUTPUT_SCHEMA
    """
    api_key = get_deepseek_api_key()
    
    symbol = input_data.get("id", {}).get("symbol", "UNKNOWN")
    asof = input_data.get("asof", "")
    vol_regime = input_data.get("volatility", {}).get("vol_regime", "unknown")
    
    if mask_date and asof:
        asof_id = f"t_{abs(hash(asof)) % 100000:05d}"
    else:
        asof_id = None
    
    # Extract liquidity checks
    volume_liquidity = input_data.get("volume_liquidity", {})
    avg_dollar_vol = volume_liquidity.get("avg_dollar_vol_21d", 0)
    liquidity_score = volume_liquidity.get("liquidity_score_0_100", 0)
    
    liquidity_ok = (avg_dollar_vol >= 10_000_000 and liquidity_score >= 50)
    min_dollar_filter = avg_dollar_vol >= 10_000_000
    min_liquidity_filter = liquidity_score >= 50
    
    llm_input = {
        **input_data,
        "exec_gates": {
            "liquidity_ok": liquidity_ok,
            "min_dollar_volume_filter": min_dollar_filter,
            "min_liquidity_score_filter": min_liquidity_filter
        }
    }
    
    if mask_date:
        llm_input.pop("asof", None)
        if asof_id:
            llm_input["time_id"] = asof_id
    
    if mask_date:
        user_content = f"""Analyze this stock for 1-week forward horizon:

Symbol: {symbol}
Time ID: {asof_id}
Volatility Regime: {vol_regime}

Execution Gates (liquidity filters - DO NOT modify):
- liquidity_ok: {liquidity_ok}
- min_dollar_volume_filter: {min_dollar_filter}
- min_liquidity_score_filter: {min_liquidity_filter}
- Avg Dollar Volume (21d): ${avg_dollar_vol:,.0f}
- Liquidity Score: {liquidity_score:.1f}/100

Technical Indicators:
{json.dumps(llm_input, ensure_ascii=False, indent=2)}

Return JSON per OUTPUT_SCHEMA with: tone, score, summary, key_reasons, risk_flags, suggested_action, confidence.
Do NOT include: symbol, asof, liquidity_ok, min_exec_checks, hold_period_days (added by system)."""
    else:
        user_content = f"""Analyze this stock for 1-week forward horizon:

Symbol: {symbol}
Date: {asof}
Volatility Regime: {vol_regime}

Execution Gates (liquidity filters - DO NOT modify):
- liquidity_ok: {liquidity_ok}
- min_dollar_volume_filter: {min_dollar_filter}
- min_liquidity_score_filter: {min_liquidity_filter}
- Avg Dollar Volume (21d): ${avg_dollar_vol:,.0f}
- Liquidity Score: {liquidity_score:.1f}/100

Technical Indicators:
{json.dumps(llm_input, ensure_ascii=False, indent=2)}

Return JSON per OUTPUT_SCHEMA with: tone, score, summary, key_reasons, risk_flags, suggested_action, confidence.
Do NOT include: symbol, asof, liquidity_ok, min_exec_checks, hold_period_days (added by system)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Prepare request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # First attempt
        try:
            response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            content = result["choices"][0]["message"]["content"]
            raw_output = json.loads(content)
            
            score = float(raw_output.get("score", 0.0))
            confidence = float(raw_output.get("confidence", 0.5))
            tone = raw_output.get("tone", "neutral")
            
            if not liquidity_ok:
                score = 0.0
                tone = "neutral"
                confidence = 0.1
            
            validated = {
                "symbol": symbol,
                "asof": asof,
                "tone": tone if tone in ["bullish", "bearish", "neutral"] else "neutral",
                "score": clamp(score, -1.0, 1.0),
                "summary": str(raw_output.get("summary", ""))[:500],
                "key_reasons": raw_output.get("key_reasons", [])[:6] if isinstance(raw_output.get("key_reasons"), list) else [],
                "risk_flags": raw_output.get("risk_flags", [])[:10] if isinstance(raw_output.get("risk_flags"), list) else [],
                "liquidity_ok": liquidity_ok,
                "min_exec_checks": {
                    "min_dollar_volume_filter": min_dollar_filter,
                    "min_liquidity_score_filter": min_liquidity_filter
                },
                "suggested_action": raw_output.get("suggested_action", "hold"),
                "hold_period_days": 7,
                "confidence": clamp(confidence, 0.0, 1.0)
            }
            
            return validated
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Warning: First attempt failed to parse JSON: {e}")
            
            # Retry with explicit instruction
            retry_message = {
                "role": "user",
                "content": "Return ONLY valid JSON per the OUTPUT_SCHEMA with double quotes and no trailing text."
            }
            payload["messages"].append(retry_message)
            
            try:
                response = await client.post(DEEPSEEK_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"]
                raw_output = json.loads(content)
                
                score = float(raw_output.get("score", 0.0))
                confidence = float(raw_output.get("confidence", 0.5))
                tone = raw_output.get("tone", "neutral")
                
                if not liquidity_ok:
                    score = 0.0
                    tone = "neutral"
                    confidence = 0.1
                
                return {
                    "symbol": symbol,
                    "asof": asof,
                    "tone": tone if tone in ["bullish", "bearish", "neutral"] else "neutral",
                    "score": clamp(score, -1.0, 1.0),
                    "summary": str(raw_output.get("summary", ""))[:500],
                    "key_reasons": raw_output.get("key_reasons", [])[:6] if isinstance(raw_output.get("key_reasons"), list) else [],
                    "risk_flags": raw_output.get("risk_flags", [])[:10] if isinstance(raw_output.get("risk_flags"), list) else [],
                    "liquidity_ok": liquidity_ok,
                    "min_exec_checks": {
                        "min_dollar_volume_filter": min_dollar_filter,
                        "min_liquidity_score_filter": min_liquidity_filter
                    },
                    "suggested_action": raw_output.get("suggested_action", "hold"),
                    "hold_period_days": 7,
                    "confidence": clamp(confidence, 0.0, 1.0)
                }
                
            except Exception as retry_error:
                print(f"Error: Retry also failed: {retry_error}")
                
                return {
                    "symbol": symbol,
                    "asof": asof,
                    "tone": "neutral",
                    "score": 0.0,
                    "summary": f"LLM analysis failed: {str(retry_error)}",
                    "key_reasons": [],
                    "risk_flags": ["llm_error"],
                    "liquidity_ok": liquidity_ok,
                    "min_exec_checks": {
                        "min_dollar_volume_filter": min_dollar_filter,
                        "min_liquidity_score_filter": min_liquidity_filter
                    },
                    "suggested_action": "hold",
                    "hold_period_days": 7,
                    "confidence": 0.0
                }
        
        except httpx.HTTPStatusError as e:
            print(f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error calling DeepSeek: {e}")
            raise


# =============================================================================
# CLI Test Interface
# =============================================================================

async def main():
    """Test the function with a sample technical payload."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run technical LLM analysis on JSON file")
    parser.add_argument("--input", required=True, help="Path to technical JSON file (e.g., out/UNH_latest.json)")
    parser.add_argument("--output", help="Optional output file for verdict (default: print to stdout)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r") as f:
        input_data = json.load(f)
    
    print(f"Analyzing {input_data.get('metadata', {}).get('symbol', 'UNKNOWN')}...")
    
    # Run analysis
    verdict = await run_technical_llm(input_data, model=args.model, temperature=args.temperature)
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"Verdict written to {args.output}")
    else:
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    asyncio.run(main())