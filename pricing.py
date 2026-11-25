"""Model prices in US$ per million tokens.

ESTIMATES. Claude rows are Anthropic list prices as of November 2026.
"Other provider" rows are placeholders. Check the providers' pricing pages
before quoting any figure to a customer.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    id: str
    name: str
    provider: str
    tier: str  # "large" | "medium" | "small"
    input: float
    output: float
    cache_read: float


CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute cache writes cost 1.25x base input
BATCH_DISCOUNT = 0.50  # batch API: 50% off every token

PRICES = [
    ModelPrice("claude-opus-5-5", "Claude Opus 5.5", "Anthropic", "large", 4.00, 20.00, 0.20),
    ModelPrice("claude-sonnet-5-5", "Claude Sonnet 5.5", "Anthropic", "medium", 2.00, 10.00, 0.10),
    ModelPrice("claude-haiku-5-5", "Claude Haiku 5.5", "Anthropic", "small", 0.10, 0.50, 0.01),
    ModelPrice("other-large", "Other provider, large", "Other", "large", 3.00, 12.00, 0.30),
    ModelPrice("other-small", "Other provider, small", "Other", "small", 0.25, 1.00, 0.03),
]
PRICE_BY_ID = {p.id: p for p in PRICES}


def small_model_for(model_id: str) -> ModelPrice:
    """Cheapest model from the same provider: the routing target."""
    p = PRICE_BY_ID[model_id]
    return min((q for q in PRICES if q.provider == p.provider), key=lambda q: q.input)
