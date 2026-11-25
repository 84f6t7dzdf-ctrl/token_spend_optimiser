"""Cost model for company-wide LLM token spend and savings levers.

Pure functions, no Streamlit, so it can be unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from pricing import BATCH_DISCOUNT, CACHE_WRITE_MULTIPLIER, PRICE_BY_ID, ModelPrice, small_model_for


@dataclass
class UseCase:
    team: str
    use_case: str
    model: str
    requests_per_month: float
    avg_input_tokens: float
    avg_output_tokens: float
    rag_tokens: float = 0.0  # retrieved context, part of avg_input_tokens
    stable_prefix_share: float = 0.5  # share of non-RAG input that repeats (cacheable)
    batchable: bool = False  # nobody waits on the answer
    routable: bool = False  # simple enough that a small model may do


@dataclass
class Levers:
    """None means the lever is off."""
    cache_hit_rate: float | None = None  # 0..1 on the stable prefix
    batch_share: float | None = None  # 0..1 of batchable traffic
    trim: float | None = None  # 0..1 cut in non-RAG input
    rag_cap: float | None = None  # max retrieved tokens per request
    output_cut: float | None = None  # 0..1 cut in output tokens
    route_share: float | None = None  # 0..1 of routable traffic to small model


# Free wins first (no quality cost), then tradeoffs that need an eval.
LEVER_ORDER = [
    ("cache_hit_rate", "Prompt caching", "free win"),
    ("batch_share", "Batch processing", "free win"),
    ("trim", "Prompt trimming", "free win"),
    ("rag_cap", "RAG context limit", "tradeoff"),
    ("output_cut", "Output caps", "tradeoff"),
    ("route_share", "Model routing", "tradeoff"),
]

NO_LEVERS = Levers()


def _per_request(static: float, prefix_share: float, rag: float, out: float,
                 p: ModelPrice, hit: float | None) -> float:
    prefix = static * prefix_share
    rest = static - prefix + rag
    if hit is None:
        in_cost = (prefix + rest) * p.input
    else:
        in_cost = prefix * (hit * p.cache_read + (1 - hit) * CACHE_WRITE_MULTIPLIER * p.input) + rest * p.input
    return (in_cost + out * p.output) / 1e6


def monthly_cost(uc: UseCase, lv: Levers = NO_LEVERS) -> float:
    rag = min(uc.rag_tokens, uc.avg_input_tokens)
    static = uc.avg_input_tokens - rag
    out = uc.avg_output_tokens
    if lv.trim is not None:
        static *= 1 - lv.trim
    if lv.rag_cap is not None:
        rag = min(rag, lv.rag_cap)
    if lv.output_cut is not None:
        out *= 1 - lv.output_cut

    big = PRICE_BY_ID[uc.model]
    share = lv.route_share if (uc.routable and lv.route_share is not None) else 0.0
    cost = uc.requests_per_month * (
        (1 - share) * _per_request(static, uc.stable_prefix_share, rag, out, big, lv.cache_hit_rate)
        + share * _per_request(static, uc.stable_prefix_share, rag, out, small_model_for(uc.model), lv.cache_hit_rate)
    )
    if uc.batchable and lv.batch_share is not None:
        cost *= 1 - BATCH_DISCOUNT * lv.batch_share
    return cost


def total(cases: list[UseCase], lv: Levers = NO_LEVERS) -> float:
    return sum(monthly_cost(c, lv) for c in cases)


def waterfall(cases: list[UseCase], lv: Levers) -> pd.DataFrame:
    """Saving added by each active lever, applied in LEVER_ORDER."""
    rows, applied, prev = [], Levers(), total(cases)
    for field, name, kind in LEVER_ORDER:
        val = getattr(lv, field)
        if val is None:
            rows.append({"lever": name, "type": kind, "saving": 0.0, "spend_after": prev})
            continue
        applied = replace(applied, **{field: val})
        now = total(cases, applied)
        rows.append({"lever": name, "type": kind, "saving": prev - now, "spend_after": now})
        prev = now
    return pd.DataFrame(rows)


def breakdown(cases: list[UseCase], lv: Levers) -> pd.DataFrame:
    rows = []
    for c in cases:
        base, opt = monthly_cost(c), monthly_cost(c, lv)
        rows.append({
            "team": c.team, "use_case": c.use_case, "model": PRICE_BY_ID[c.model].name,
            "requests": c.requests_per_month,
            "tokens": c.requests_per_month * (c.avg_input_tokens + c.avg_output_tokens),
            "cost_per_request": base / c.requests_per_month if c.requests_per_month else 0.0,
            "current": base, "after": opt,
        })
    return pd.DataFrame(rows).sort_values("current", ascending=False)


def chargeback(cases: list[UseCase], lv: Levers, platform_cost: float) -> pd.DataFrame:
    """Each team pays its token cost plus a token-weighted share of the shared platform cost."""
    df = breakdown(cases, lv).groupby("team", as_index=False)[["tokens", "after"]].sum()
    tot = df["tokens"].sum() or 1
    df["platform_share"] = platform_cost * df["tokens"] / tot
    df["charge"] = df["after"] + df["platform_share"]
    return df.sort_values("charge", ascending=False)


def recommendations(cases: list[UseCase]) -> list[str]:
    recs = []
    for c in sorted(cases, key=monthly_cost, reverse=True):
        p = PRICE_BY_ID[c.model]
        if p.tier == "large" and c.routable and c.avg_output_tokens <= 200:
            recs.append(f"Route '{c.use_case}' to a small model: {int(c.avg_output_tokens)}-token outputs on {p.name} look like classification.")
        if c.stable_prefix_share >= 0.6 and c.avg_input_tokens >= 5000:
            recs.append(f"Cache the prompt prefix for '{c.use_case}': {c.stable_prefix_share:.0%} of a {int(c.avg_input_tokens):,}-token prompt repeats.")
        if c.batchable:
            recs.append(f"Move '{c.use_case}' to the batch API: nobody waits on it, so it runs at half price.")
        if c.rag_tokens >= 8000:
            recs.append(f"Cap retrieved context for '{c.use_case}': {int(c.rag_tokens):,} RAG tokens per request.")
    return recs[:5]


SAMPLE = [
    UseCase("Customer Support", "Support chatbot", "claude-sonnet-5-5", 900_000, 6000, 350, 2500, 0.8, False, True),
    UseCase("Customer Support", "Ticket triage", "claude-opus-5-5", 400_000, 1500, 60, 0, 0.9, True, True),
    UseCase("Engineering", "Coding agent", "claude-opus-5-5", 120_000, 40_000, 2500, 0, 0.9, False, False),
    UseCase("Engineering", "PR review bot", "claude-opus-5-5", 30_000, 25_000, 1500, 0, 0.5, True, False),
    UseCase("Marketing", "Campaign copy", "claude-opus-5-5", 60_000, 3000, 1800, 0, 0.6, True, True),
    UseCase("Legal", "Contract review", "claude-opus-5-5", 8000, 60_000, 3000, 40_000, 0.7, True, False),
    UseCase("Data & Analytics", "Nightly doc summaries", "claude-sonnet-5-5", 250_000, 8000, 600, 0, 0.2, True, True),
    UseCase("Data & Analytics", "SQL assistant", "other-large", 40_000, 5000, 500, 0, 0.7, False, False),
    UseCase("People (HR)", "Policy Q&A", "claude-sonnet-5-5", 150_000, 12_000, 400, 10_000, 0.8, False, True),
]

CSV_COLUMNS = ["team", "use_case", "model", "requests_per_month", "avg_input_tokens", "avg_output_tokens",
               "rag_tokens", "stable_prefix_share", "batchable", "routable"]


def to_frame(cases: list[UseCase]) -> pd.DataFrame:
    return pd.DataFrame([vars(c) for c in cases])[CSV_COLUMNS]


def from_frame(df: pd.DataFrame) -> list[UseCase]:
    missing = [c for c in CSV_COLUMNS[:6] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    out = []
    for r in df.to_dict("records"):
        model = r["model"] if r["model"] in PRICE_BY_ID else "claude-sonnet-5-5"
        as_bool = lambda v: str(v).strip().lower() in {"true", "yes", "1", "y"}
        out.append(UseCase(
            str(r["team"]), str(r["use_case"]), model, float(r["requests_per_month"]),
            float(r["avg_input_tokens"]), float(r["avg_output_tokens"]),
            float(r.get("rag_tokens", 0) or 0), float(r.get("stable_prefix_share", 0.5) or 0.5),
            as_bool(r.get("batchable", False)), as_bool(r.get("routable", False)),
        ))
    return out
