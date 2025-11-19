"""
Technical LLM Analysis Module
Sends technical JSON payloads to DeepSeek and returns strictly-validated verdicts.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import httpx

# =============================================================================
# Configuration
# =============================================================================

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2
MAX_TOKENS = 800
REQUEST_TIMEOUT = 30.0

SYSTEM_PROMPT = """You are a technical "Dynamics" analyst. Using ONLY the numeric fields in DATA, write a brief interpretation of momentum/trend, volatility/risk, volume/liquidity, price events, and relative performance. DO NOT invent data. DO NOT reference news or fundamentals. 

Output MUST be valid JSON matching this exact schema:
{
  "symbol": "string",
  "asof": "YYYY-MM-DD",
  "tone": "bullish or bearish or neutral",
  "score": <number between -1 and 1>,
  "summary": "Brief 2-3 sentence technical summary",
  "key_reasons": [
    {"category": "momentum or trend or volatility or liquidity or events or relative", "field": "field_name", "evidence": "what the data shows"}
  ],
  "risk_flags": ["array of risk identifiers like drawdown, high_vol, gap_down, weak_rs, overbought, illiquidity"],
  "liquidity_ok": <boolean from input>,
  "min_exec_checks": {"min_dollar_volume_filter": <boolean>, "min_liquidity_score_filter": <boolean>},
  "suggested_action": "long or short or hold",
  "hold_period_days": 7,
  "confidence": <number between 0 and 1>
}

Provide a tone (bullish/bearish/neutral) and a weekly score in [-1, 1] suitable for ranking. If liquidity gates fail, force tone="neutral" and score=0."""

OUTPUT_SCHEMA = {
    "symbol": "string",
    "asof": "ISO-8601",
    "tone": "bullish | bearish | neutral",
    "score": "number",
    "summary": "string",
    "key_reasons": [
        {"category": "momentum|trend|volatility|liquidity|events|relative", "field": "string", "evidence": "string"}
    ],
    "risk_flags": ["drawdown|high_vol|gap_down|weak_rs|overbought|illiquidity"],
    "liquidity_ok": "boolean",
    "min_exec_checks": {"min_dollar_volume_filter": "boolean", "min_liquidity_score_filter": "boolean"},
    "suggested_action": "long|short|hold",
    "hold_period_days": 7,
    "confidence": "number"
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
    Post-parse hardening and validation of LLM output.
    Ensures all required fields exist and applies business rules.
    """
    # Extract liquidity filters from input
    volume_liquidity = input_data.get("Volume_Liquidity", {})
    min_dollar_filter = volume_liquidity.get("min_dollar_volume_filter", False)
    min_liquidity_filter = volume_liquidity.get("min_liquidity_score_filter", False)
    liquidity_ok = min_dollar_filter and min_liquidity_filter
    
    # Extract and validate each field from LLM output
    validated = {}
    
    # Symbol and date from input metadata
    validated["symbol"] = raw_output.get("symbol", input_data.get("metadata", {}).get("symbol", "UNKNOWN"))
    validated["asof"] = raw_output.get("asof", input_data.get("metadata", {}).get("asof", ""))
    
    # Validate tone (must be one of three values)
    tone = raw_output.get("tone", "neutral")
    validated["tone"] = tone if tone in ["bullish", "bearish", "neutral"] else "neutral"
    
    # Validate score (must be numeric in [-1, 1])
    try:
        score = float(raw_output.get("score", 0.0))
        validated["score"] = clamp(score, -1.0, 1.0)
    except (TypeError, ValueError):
        validated["score"] = 0.0
    
    # Summary (required text field)
    validated["summary"] = str(raw_output.get("summary", ""))[:500]
    
    # Key reasons (array of objects)
    key_reasons = raw_output.get("key_reasons", [])
    if isinstance(key_reasons, list):
        validated["key_reasons"] = key_reasons[:6]
    else:
        validated["key_reasons"] = []
    
    # Risk flags (array of strings)
    risk_flags = raw_output.get("risk_flags", [])
    if isinstance(risk_flags, list):
        validated["risk_flags"] = [str(flag) for flag in risk_flags][:10]
    else:
        validated["risk_flags"] = []
    
    # Liquidity checks (from input data, not LLM)
    validated["liquidity_ok"] = liquidity_ok
    validated["min_exec_checks"] = {
        "min_dollar_volume_filter": min_dollar_filter,
        "min_liquidity_score_filter": min_liquidity_filter
    }
    
    # Suggested action (must be one of three values)
    action = raw_output.get("suggested_action", "hold")
    validated["suggested_action"] = action if action in ["long", "short", "hold"] else "hold"
    
    # Hold period (integer)
    try:
        validated["hold_period_days"] = int(raw_output.get("hold_period_days", 7))
    except (TypeError, ValueError):
        validated["hold_period_days"] = 7
    
    # Confidence (must be numeric in [0, 1])
    try:
        confidence = float(raw_output.get("confidence", 0.5))
        validated["confidence"] = clamp(confidence, 0.0, 1.0)
    except (TypeError, ValueError):
        validated["confidence"] = 0.5
    
    # Apply liquidity gate OVERRIDE - force neutral if illiquid
    if not liquidity_ok:
        validated["tone"] = "neutral"
        validated["score"] = 0.0
        validated["suggested_action"] = "hold"
    
    return validated


def compute_fallback_score(input_data: Dict[str, Any]) -> float:
    """
    Compute a fallback score from composites if LLM output is invalid.
    Maps tier1_master_score [0,100] → [-1,1] or blends momentum/trend/rs.
    """
    composites = input_data.get("Composites", {})
    
    # Try tier1_master_score first
    if "tier1_master_score" in composites and composites["tier1_master_score"] is not None:
        master = composites["tier1_master_score"]
        # Map [0, 100] → [-1, 1]: score = (master - 50) / 50
        return clamp((master - 50.0) / 50.0, -1.0, 1.0)
    
    # Fallback: blend momentum, trend, relative strength
    momentum_score = composites.get("momentum_score", 0)
    trend_score = composites.get("trend_score", 0)
    rs_score = composites.get("relative_strength_score", 0)
    
    # Average and normalize from [-100, 100] → [-1, 1]
    avg_score = (momentum_score + trend_score + rs_score) / 3.0
    return clamp(avg_score / 100.0, -1.0, 1.0)


# =============================================================================
# Main LLM Function
# =============================================================================

async def run_technical_llm(
    input_data: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE
) -> dict:
    """
    Send technical JSON to DeepSeek and return strictly-validated verdict.
    
    Args:
        input_data: Technical analysis JSON (e.g., UNH payload)
        model: DeepSeek model name (default: "deepseek-chat")
        temperature: Sampling temperature (default: 0.2)
    
    Returns:
        Validated verdict dictionary matching OUTPUT_SCHEMA
    """
    api_key = get_deepseek_api_key()
    
    # Build messages
    user_content = f"DATA:\n{json.dumps(input_data, ensure_ascii=False, indent=2)}"
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
        "response_format": {"type": "json_object"}  # Request JSON mode
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
            
            # Validate and harden
            return validate_and_harden(raw_output, input_data)
            
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
                
                return validate_and_harden(raw_output, input_data)
                
            except Exception as retry_error:
                print(f"Error: Retry also failed: {retry_error}")
                
                # Return minimal fallback with computed score
                fallback_score = compute_fallback_score(input_data)
                volume_liquidity = input_data.get("Volume_Liquidity", {})
                liquidity_ok = (
                    volume_liquidity.get("min_dollar_volume_filter", False) and
                    volume_liquidity.get("min_liquidity_score_filter", False)
                )
                
                return {
                    "symbol": input_data.get("metadata", {}).get("symbol", "UNKNOWN"),
                    "asof": input_data.get("metadata", {}).get("asof", ""),
                    "tone": "neutral" if not liquidity_ok else ("bullish" if fallback_score > 0.2 else "bearish" if fallback_score < -0.2 else "neutral"),
                    "score": 0.0 if not liquidity_ok else fallback_score,
                    "summary": "LLM analysis unavailable. Using fallback scoring from composites.",
                    "key_reasons": [],
                    "risk_flags": ["llm_unavailable"],
                    "liquidity_ok": liquidity_ok,
                    "min_exec_checks": {
                        "min_dollar_volume_filter": volume_liquidity.get("min_dollar_volume_filter", False),
                        "min_liquidity_score_filter": volume_liquidity.get("min_liquidity_score_filter", False)
                    },
                    "suggested_action": "hold" if not liquidity_ok else "long" if fallback_score > 0.2 else "short" if fallback_score < -0.2 else "hold",
                    "hold_period_days": 7,
                    "confidence": 0.3
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