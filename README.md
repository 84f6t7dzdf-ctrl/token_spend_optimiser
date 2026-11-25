# token_spend_optimiser

Create a tool to help a company optimise its token spend.


Analyses a company's LLM token spend by team, model and use case, models six savings
levers, and shows team budgets and chargeback.

**Prices are estimates** (`pricing.py`). Claude rows are list prices as of November 2026;
"Other provider" rows are placeholders. Check pricing pages before quoting a number.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
pip install -r requirements-dev.txt && pytest -q   # 8 tests
```

Deploy: on share.streamlit.io pick this repo, branch `main` and main file `app.py`.

## Files

| File | What it does |
|---|---|
| `pricing.py` | Model price table, cache write multiplier (1.25x), batch discount (50%) |
| `optimiser.py` | Pure cost model: `monthly_cost`, `waterfall`, `breakdown`, `chargeback`, `recommendations`, sample data, CSV in/out |
| `app.py` | Streamlit UI: KPIs, Overview, Savings levers, Budgets and chargeback, Data and prices |
| `tests/test_optimiser.py` | Hand-checked unit tests for each lever |

## Cost model (per use case, per month)

1. Split input into stable prefix (cacheable), the rest of the prompt, and RAG context.
2. Trimming cuts non-RAG input. The RAG cap limits retrieved tokens. Output caps cut output tokens.
3. Caching: prefix tokens cost `hit * cache_read + (1 - hit) * 1.25 * input`.
4. Routing: a share of routable traffic is priced on the provider's cheapest model.
5. Batch: a share of batchable traffic gets 50% off.

Levers are applied in order: free wins (caching, batch, trimming) and then tradeoffs
(RAG cap, output caps, routing), which need an eval before rollout.

## CSV columns

`team, use_case, model, requests_per_month, avg_input_tokens, avg_output_tokens,
rag_tokens, stable_prefix_share, batchable, routable`
