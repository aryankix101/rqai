"""Shared portfolio-level trading guardrails."""

import math


MIN_TRADE_PRICE = 5.0


def has_min_trade_price(price) -> bool:
    """Return True when the candidate price clears the trading floor."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > MIN_TRADE_PRICE
