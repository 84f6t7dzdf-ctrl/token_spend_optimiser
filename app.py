"""Token Spend Optimiser: Streamlit app.

Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from optimiser import (SAMPLE, Levers, breakdown, chargeback, from_frame, recommendations,
                       to_frame, total, waterfall)
from pricing import PRICES

st.set_page_config(page_title="Token Spend Optimiser", layout="wide")
st.title("Token Spend Optimiser")
st.caption("Where a company's LLM token spend goes, and what each savings lever is worth. "
           "Sample data. **Prices are estimates**: check providers' pricing pages before quoting.")

# ---- Sidebar: levers -------------------------------------------------------
st.sidebar.header("Savings levers")
st.sidebar.caption("Free wins first, then tradeoffs that need an eval before rollout.")


def lever(label, key, default_on, minv, maxv, default, step, help_text, pct=True):
    on = st.sidebar.checkbox(label, value=default_on, key=f"on_{key}", help=help_text)
    val = st.sidebar.slider(f"{label} setting", minv, maxv, default, step, key=f"sl_{key}",
                            disabled=not on, label_visibility="collapsed")
    return (val / 100 if pct else val) if on else None


lv = Levers(
    cache_hit_rate=lever("Prompt caching (hit rate %)", "cache", True, 0, 95, 70, 5, "Free win. Reuse the stable prompt prefix."),
    batch_share=lever("Batch processing (% of batchable)", "batch", True, 0, 100, 80, 5, "Free win. 50% off for work nobody waits on."),
    trim=lever("Prompt trimming (% cut)", "trim", True, 0, 50, 15, 5, "Free win. Remove prompt bloat."),
    rag_cap=lever("RAG context limit (tokens)", "rag", True, 1000, 40000, 4000, 1000, "Tradeoff. Rerank, keep top chunks.", pct=False),
    output_cut=lever("Output caps (% cut)", "out", True, 0, 60, 25, 5, "Tradeoff. Concise answers, max_tokens, lower effort."),
    route_share=lever("Model routing (% to small model)", "route", True, 0, 90, 50, 5, "Tradeoff. Simple requests to the small model."),
)

# ---- Data ------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = to_frame(SAMPLE)

tab_over, tab_levers, tab_gov, tab_data = st.tabs(["Overview", "Savings levers", "Budgets and chargeback", "Data and prices"])

with tab_data:
    st.subheader("Workload")
    st.caption("One row per use case, monthly volume. Input tokens include RAG context. "
               "Stable prefix share = share of non-RAG input that repeats (cacheable).")
    up = st.file_uploader("Import a usage CSV", type="csv")
    if up is not None:
        try:
            st.session_state.data = to_frame(from_frame(pd.read_csv(up)))
            st.success("Imported.")
        except Exception as e:  # show the reason, keep the old data
            st.error(f"Could not import: {e}")
    edited = st.data_editor(st.session_state.data, num_rows="dynamic", width="stretch",
                            column_config={"model": st.column_config.SelectboxColumn(options=[p.id for p in PRICES])})
    st.download_button("Download CSV", edited.to_csv(index=False), "token_usage.csv", "text/csv")
    st.subheader("Model prices (US$ per million tokens, estimates)")
    st.dataframe(pd.DataFrame([vars(p) for p in PRICES]), width="stretch", hide_index=True)

try:
    cases = from_frame(edited)
except Exception as e:
    st.error(f"Data problem: {e}")
    st.stop()

base, opt = total(cases), total(cases, lv)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current spend / month", f"${base:,.0f}")
c2.metric("After levers / month", f"${opt:,.0f}")
c3.metric("Savings", f"{(base - opt) / base:.0%}" if base else "0%", f"${base - opt:,.0f} / month")
c4.metric("Annualised savings", f"${(base - opt) * 12:,.0f}")

bd = breakdown(cases, lv)

with tab_over:
    a, b = st.columns(2)
    a.subheader("Spend by team")
    a.bar_chart(bd.groupby("team")[["current", "after"]].sum().sort_values("current"), horizontal=True, stack=False)
    b.subheader("Spend by model")
    b.bar_chart(bd.groupby("model")[["current", "after"]].sum().sort_values("current"), horizontal=True, stack=False)
    st.subheader("Use cases by current spend")
    show = bd.copy()
    show["share"] = show["current"] / base
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "cost_per_request": st.column_config.NumberColumn(format="$%.4f"),
        "current": st.column_config.NumberColumn(format="$%.0f"),
        "after": st.column_config.NumberColumn(format="$%.0f"),
        "share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
    })
    st.subheader("Top recommendations")
    for r in recommendations(cases):
        st.markdown(f"- {r}")

with tab_levers:
    st.subheader("Savings waterfall")
    st.caption("Each row is the saving a lever adds on top of the ones above it.")
    wf = waterfall(cases, lv)
    st.bar_chart(wf.set_index("lever")["saving"], horizontal=True)
    st.dataframe(wf, width="stretch", hide_index=True, column_config={
        "saving": st.column_config.NumberColumn(format="$%.0f"),
        "spend_after": st.column_config.NumberColumn(format="$%.0f")})

with tab_gov:
    st.subheader("Team budgets")
    teams = bd.groupby("team")[["current", "after"]].sum().reset_index()
    teams["budget"] = (teams["current"] * 0.8 / 500).round() * 500
    teams = st.data_editor(teams, disabled=["team", "current", "after"], hide_index=True, width="stretch")
    teams["use_of_budget"] = teams["after"] / teams["budget"].where(teams["budget"] > 0)
    teams["status"] = teams["use_of_budget"].map(lambda u: "Over budget" if u > 1 else "Alert at 80%" if u >= 0.8 else "On track")
    st.dataframe(teams[["team", "budget", "after", "use_of_budget", "status"]], hide_index=True, width="stretch")
    st.subheader("Chargeback")
    platform = st.number_input("Shared platform cost per month ($)", 0, 1_000_000, 6000, 500)
    st.dataframe(chargeback(cases, lv, platform), hide_index=True, width="stretch")
