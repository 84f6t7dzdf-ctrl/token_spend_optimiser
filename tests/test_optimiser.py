import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from optimiser import SAMPLE, Levers, UseCase, chargeback, monthly_cost, total, waterfall


def uc(**kw):
    base = dict(team="T", use_case="U", model="claude-sonnet-5-5", requests_per_month=1_000_000,
                avg_input_tokens=1000, avg_output_tokens=100)
    base.update(kw)
    return UseCase(**base)


def test_baseline_cost_matches_list_price():
    # 1B input tokens at $2/M + 100M output at $10/M = $2,000 + $1,000
    assert monthly_cost(uc()) == pytest.approx(3000)


def test_batch_halves_batchable_only():
    assert monthly_cost(uc(batchable=True), Levers(batch_share=1.0)) == pytest.approx(1500)
    assert monthly_cost(uc(batchable=False), Levers(batch_share=1.0)) == pytest.approx(3000)


def test_full_cache_hits_charge_cache_read():
    c = uc(stable_prefix_share=1.0)
    # input becomes 1B * $0.10/M = $100, output stays $1,000
    assert monthly_cost(c, Levers(cache_hit_rate=1.0)) == pytest.approx(1100)


def test_zero_cache_hits_cost_more_than_no_caching():
    c = uc(stable_prefix_share=1.0)
    assert monthly_cost(c, Levers(cache_hit_rate=0.0)) > monthly_cost(c)


def test_routing_moves_share_to_small_model():
    c = uc(routable=True)
    small = 1000 * 0.10 + 100 * 0.50  # Haiku cost for the same 1M requests
    expected = 0.5 * 3000 + 0.5 * small
    assert monthly_cost(c, Levers(route_share=0.5)) == pytest.approx(expected)


def test_rag_cap_only_trims_retrieved_tokens():
    c = uc(avg_input_tokens=10_000, rag_tokens=8000)
    capped = monthly_cost(c, Levers(rag_cap=2000))
    assert capped == pytest.approx(1_000_000 * ((2000 + 2000) * 2 + 100 * 10) / 1e6)


def test_waterfall_sums_to_total_saving():
    lv = Levers(0.7, 0.8, 0.15, 4000, 0.25, 0.5)
    wf = waterfall(SAMPLE, lv)
    assert wf["saving"].sum() == pytest.approx(total(SAMPLE) - total(SAMPLE, lv))
    assert (wf["saving"] >= 0).all()


def test_chargeback_includes_platform_cost():
    lv = Levers()
    cb = chargeback(SAMPLE, lv, 6000)
    assert cb["platform_share"].sum() == pytest.approx(6000)
    assert cb["charge"].sum() == pytest.approx(total(SAMPLE) + 6000)
