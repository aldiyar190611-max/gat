from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from core.data import (
    generate_data, get_state, ACCOUNTS, FX_RATES, CLEARING_DAYS, CHANNEL_RELIABILITY
)
from core.ml import CashFlowForecaster
from core.engine import RiskEngine, LiquidityOptimizer, compute_whatif, SEV_COLORS

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinFlow AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
}
.finflow-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #58a6ff 0%, #3fb950 50%, #d29922 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.card-green  { background:#0d2818; border:1.5px solid #3fb950; border-radius:10px; padding:14px; margin:4px; }
.card-yellow { background:#2d2200; border:1.5px solid #d29922; border-radius:10px; padding:14px; margin:4px; }
.card-red    { background:#2d0f0f; border:1.5px solid #f85149; border-radius:10px; padding:14px; margin:4px; }
.sev-critical { background:#2d0f0f; border-left:4px solid #f85149; padding:12px 14px; border-radius:6px; margin:6px 0; }
.sev-high     { background:#2d1a00; border-left:4px solid #d29922; padding:12px 14px; border-radius:6px; margin:6px 0; }
.sev-medium   { background:#2d2a00; border-left:4px solid #e3b341; padding:12px 14px; border-radius:6px; margin:6px 0; }
.sev-low      { background:#0d2818; border-left:4px solid #3fb950; padding:12px 14px; border-radius:6px; margin:6px 0; }
.rec-card    { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; margin:8px 0; }
.rec-done    { background:#0d2818; border:1px solid #3fb950; border-radius:10px; padding:16px; margin:8px 0; }
.tag-urgent  { background:#f85149; color:#fff; font-size:11px; font-weight:700; padding:2px 8px; border-radius:20px; }
.tag-normal  { background:#d29922; color:#000; font-size:11px; font-weight:700; padding:2px 8px; border-radius:20px; }
.factor-row  { display:flex; justify-content:space-between; padding:5px 10px; background:#161b22; border-radius:5px; margin:2px 0; font-size:13px; }
.prob-bar-wrap { background:#21262d; border-radius:8px; height:10px; margin:4px 0; overflow:hidden; }
.sidebar-badge { background:#1f6feb; color:#fff; border-radius:6px; padding:4px 10px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "user_accounts": [],
    "use_custom_data": False,
    "custom_fx": dict(FX_RATES),
    "confirmed_transfers": {},   # id → rec dict
    "demo_mode": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Загрузка данных…")
def _load_df(accounts_json: str = "") -> pd.DataFrame:
    if accounts_json:
        return generate_data(months=12, accounts=json.loads(accounts_json))
    return generate_data(months=12)

@st.cache_resource(show_spinner="Обучение ML-моделей…")
def _load_model(df_hash: int, accounts_json: str = "") -> CashFlowForecaster:
    df = _load_df(accounts_json)
    fc = CashFlowForecaster()
    fc.train(df)
    return fc

def _gen_accounts():
    if st.session_state.use_custom_data and st.session_state.user_accounts:
        return [{k: v for k, v in a.items() if k != "current_balance"}
                for a in st.session_state.user_accounts]
    return None

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="finflow-title">💹 FinFlow AI</div>', unsafe_allow_html=True)
    st.caption("Treasury Intelligence Platform v2.0")
    st.divider()

    horizon = st.slider("Горизонт прогноза (дни)", 1, 7, 3)

    st.markdown("**Фильтр валют**")
    c1, c2, c3 = st.columns(3)
    show_usd = c1.checkbox("USD", value=True)
    show_eur = c2.checkbox("EUR", value=True)
    show_gbp = c3.checkbox("GBP", value=True)
    currencies = [c for c, v in [("USD", show_usd), ("EUR", show_eur), ("GBP", show_gbp)] if v]

    st.divider()

    demo_btn_label = "⏹ Остановить Demo" if st.session_state.demo_mode else "🔥 WOW Demo: SWIFT Outage"
    demo_btn_type  = "secondary" if st.session_state.demo_mode else "primary"
    if st.button(demo_btn_label, use_container_width=True, type=demo_btn_type):
        st.session_state.demo_mode = not st.session_state.demo_mode
        st.session_state.confirmed_transfers = {}
        st.rerun()

    if st.button("🔄 Обновить данные", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    if st.session_state.demo_mode:
        st.error("🔴 DEMO: SWIFT отключён")
    if st.session_state.use_custom_data and st.session_state.user_accounts:
        st.success(f"Свои данные: {len(st.session_state.user_accounts)} счёт(а)")
    else:
        st.caption("Источник: синтетические данные")

    st.caption("FinTech Hackathon 2026 | by FinFlow Team")

# ══════════════════════════════════════════════════════════════════════════════
# Load & compute
# ══════════════════════════════════════════════════════════════════════════════
_accs = _gen_accounts()
_acc_json = json.dumps(_accs, ensure_ascii=False, sort_keys=True) if _accs else ""
_fx = st.session_state.custom_fx

df      = _load_df(_acc_json)
model   = _load_model(hash(str(df.shape) + _acc_json), _acc_json)
state   = get_state(df)

# Apply user overrides
if _accs:
    for acc in st.session_state.user_accounts:
        cb = float(acc.get("current_balance", 0))
        if cb > 0:
            m = state["account_id"] == acc["id"]
            if m.any():
                pi = state.loc[m, "pending_inflow"].values[0]
                state.loc[m, ["balance", "available_balance", "usd_equivalent",
                               "excess", "deficit"]] = [
                    cb, cb + pi, cb * _fx.get(acc["currency"], 1.0),
                    max(0.0, cb - acc["target_balance"]),
                    max(0.0, acc["min_balance"] - cb),
                ]

# Apply WOW Demo: SWIFT accounts → 35% balance
if st.session_state.demo_mode:
    for idx, row in state.iterrows():
        if row["payment_system"] == "SWIFT":
            nb = row["balance"] * 0.35
            state.loc[idx, "balance"]       = nb
            state.loc[idx, "usd_equivalent"]= nb * _fx.get(row["currency"], 1.0)
            state.loc[idx, "deficit"]       = max(0.0, row["min_balance"] - nb)
            state.loc[idx, "excess"]        = 0.0

# Apply confirmed transfers to balances
for tid, rec in st.session_state.confirmed_transfers.items():
    fm = state["account_id"] == rec["from_id"]
    tm = state["account_id"] == rec["to_id"]
    if fm.any():
        nb = max(0.0, state.loc[fm, "balance"].values[0] - rec["amount"])
        state.loc[fm, "balance"]        = nb
        state.loc[fm, "usd_equivalent"] = nb * _fx.get(rec["currency_from"], 1.0)
        state.loc[fm, "deficit"]        = max(0.0, state.loc[fm, "min_balance"].values[0] - nb)
    if tm.any():
        nb = state.loc[tm, "balance"].values[0] + rec["amount_dest"]
        state.loc[tm, "balance"]        = nb
        state.loc[tm, "usd_equivalent"] = nb * _fx.get(rec["currency_to"], 1.0)
        state.loc[tm, "excess"]         = max(0.0, nb - state.loc[tm, "target_balance"].values[0])

state_f = state[state["currency"].isin(currencies)].copy()
balances = dict(zip(state["account_id"], state["balance"]))
forecasts = model.forecast_all(days=horizon, current_balances=balances)
fc_f = forecasts[forecasts["currency"].isin(currencies)] if not forecasts.empty else pd.DataFrame()

risk_engine = RiskEngine()
alerts      = risk_engine.generate_alerts(state_f, fc_f)
alert_sum   = risk_engine.summary(alerts)
optimizer   = LiquidityOptimizer()
recs        = optimizer.recommend(state_f, fc_f)
idle        = optimizer.idle_report(state_f)

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.demo_mode:
    st.markdown('<h1 style="color:#f85149">🔴 FinFlow AI — CRITICAL: SWIFT Outage</h1>', unsafe_allow_html=True)
    st.error("**ML-модель обнаружила**: риск дефицита ликвидности через 48 часов на SWIFT-счетах. Перейдите в **Rebalancing Hub** для подтверждения переводов.")
else:
    st.markdown('<h1 class="finflow-title">💹 FinFlow AI — Treasury Intelligence Platform</h1>', unsafe_allow_html=True)
    st.caption(f"{df['date'].max().strftime('%d %B %Y')}  ·  Счетов: {len(state_f)}  ·  Горизонт: {horizon} дн.")

# KPI row
total_liq = state_f["usd_equivalent"].sum()
frozen    = idle["total_idle_usd"]
opp_cost  = idle["annual_opp_cost_usd"]
crit_cnt  = alert_sum.get("CRITICAL", 0) + alert_sum.get("HIGH", 0)
pend_usd  = (state_f["pending_inflow"] * state_f["currency"].map(_fx)).sum()
efficiency= (total_liq - frozen) / total_liq * 100 if total_liq else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Общая ликвидность",  f"${total_liq/1e6:.2f}M")
k2.metric("Эффективность",      f"{efficiency:.1f}%",
          delta=f"{efficiency-75:.1f}% vs норма",
          delta_color="normal" if efficiency >= 75 else "inverse")
k3.metric("Критических алертов", str(crit_cnt),
          delta=f"+{crit_cnt}" if crit_cnt else None,
          delta_color="inverse" if crit_cnt else "off")
k4.metric("Заморожено (idle)",  f"${frozen/1e6:.2f}M")
k5.metric("Упущ. доход/год",    f"${opp_cost/1e3:.0f}K")
k6.metric("Pending клиринг",    f"${pend_usd/1e6:.2f}M")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Дашборд",
    "📈 Прогноз",
    "🚨 Risk Engine",
    "⚖️ Rebalancing Hub",
    "🔬 What-If",
    "📊 KPI & Analytics",
    "📝 Данные",
])

# ─────────────────────────────── TAB 1: Dashboard ───────────────────────────
with tab1:
    st.subheader("Global Liquidity Map")

    cols3 = st.columns(3)
    for i, (_, row) in enumerate(state_f.iterrows()):
        bal, min_b, tgt_b = row["balance"], row["min_balance"], row["target_balance"]
        if bal < min_b:
            css, status, status_color = "card-red",    "🔴 ДЕФИЦИТ",  "#f85149"
        elif bal < tgt_b * 0.8:
            css, status, status_color = "card-yellow", "🟡 ВНИМАНИЕ", "#d29922"
        else:
            css, status, status_color = "card-green",  "🟢 НОРМА",    "#3fb950"
        fill = min(100, bal / tgt_b * 100)
        bar_color = "#3fb950" if fill >= 80 else "#d29922" if fill >= 50 else "#f85149"
        with cols3[i % 3]:
            st.markdown(f"""
<div class="{css}">
  <div style="font-size:13px;color:#8b949e;margin-bottom:4px">{row['payment_system']} · {row['currency']}</div>
  <div style="font-weight:700;font-size:15px">{row['account_name']}</div>
  <div style="font-size:26px;font-weight:700;margin:8px 0;color:#e6edf3">{bal:,.0f} <span style="font-size:15px;color:#8b949e">{row['currency']}</span></div>
  <div class="prob-bar-wrap"><div style="width:{fill:.0f}%;background:{bar_color};height:10px;border-radius:8px"></div></div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:12px;color:#8b949e">
    <span>МИН {min_b:,.0f}</span><span style="color:{status_color};font-weight:600">{status}</span><span>ЦЕЛЬ {tgt_b:,.0f}</span>
  </div>
  {'<div style="font-size:12px;color:#58a6ff;margin-top:4px">⏳ Клиринг: '+f"{row['pending_inflow']:,.0f}"+'</div>' if row['pending_inflow'] > 0 else ''}
</div>""", unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Динамика ликвидности (90 дней)")
        h90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
        h90 = h90[h90["currency"].isin(currencies)]
        daily = h90.groupby("date").apply(
            lambda g: (g["balance"] * g["currency"].map(_fx)).sum()
        ).reset_index(name="usd")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["usd"], fill="tozeroy",
                                  line=dict(color="#58a6ff", width=2),
                                  fillcolor="rgba(88,166,255,0.12)", name="Ликвидность"))
        fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                          xaxis_title="", yaxis_title="USD", margin=dict(t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("По валютам")
        ccy = state_f.groupby("currency")["usd_equivalent"].sum().reset_index()
        fig2 = px.pie(ccy, values="usd_equivalent", names="currency", hole=0.55,
                      color_discrete_map={"USD":"#58a6ff","EUR":"#3fb950","GBP":"#d29922"})
        fig2.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6edf3", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Статус алертов**")
        for sev in ["CRITICAL","HIGH","MEDIUM","LOW"]:
            cnt = alert_sum.get(sev, 0)
            color = SEV_COLORS[sev]
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 10px;background:#161b22;border-left:3px solid {color};border-radius:4px;margin:2px 0"><span style="color:{color};font-size:13px">{sev}</span><strong>{cnt}</strong></div>', unsafe_allow_html=True)

    # Channel reliability
    st.divider()
    st.subheader("Bank & Channel Risk Engine")
    ch_cols = st.columns(4)
    for i, (ch, rel) in enumerate(CHANNEL_RELIABILITY.items()):
        color = "#3fb950" if rel >= 0.98 else "#d29922" if rel >= 0.95 else "#f85149"
        score = int(rel * 100)
        with ch_cols[i]:
            st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;text-align:center">
  <div style="font-size:13px;color:#8b949e">{ch}</div>
  <div style="font-size:30px;font-weight:700;color:{color};margin:8px 0">{score}%</div>
  <div class="prob-bar-wrap"><div style="width:{score}%;background:{color};height:10px;border-radius:8px"></div></div>
  <div style="font-size:11px;color:#8b949e;margin-top:4px">надёжность</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────── TAB 2: Forecast ─────────────────────────────
with tab2:
    st.subheader("Прогноз Cash Flow — квантильные сценарии")

    if fc_f.empty:
        st.warning("Нет данных прогноза. Выберите хотя бы одну валюту.")
    else:
        sel_acc = st.selectbox(
            "Счёт",
            options=state_f["account_id"].tolist(),
            format_func=lambda x: state_f[state_f["account_id"]==x]["account_name"].iloc[0],
        )
        acc_fc    = fc_f[fc_f["account_id"] == sel_acc]
        acc_state = state_f[state_f["account_id"] == sel_acc].iloc[0]
        hist14    = df[(df["account_id"]==sel_acc) & (df["date"] >= df["date"].max()-pd.Timedelta(days=14))]

        # Probability of shortage
        p_short = model.p_shortage(sel_acc, acc_state["min_balance"], fc_f)
        p_color = "#f85149" if p_short > 0.3 else "#d29922" if p_short > 0.1 else "#3fb950"

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Текущий баланс", f"{acc_state['balance']:,.0f} {acc_state['currency']}")
        pm2.metric("Прогноз (q50)",  f"{acc_fc.iloc[-1]['q50']:,.0f}" if not acc_fc.empty else "—")
        pm3.metric("P(дефицит)",     f"{p_short*100:.1f}%",
                   delta="HIGH RISK" if p_short > 0.3 else "OK",
                   delta_color="inverse" if p_short > 0.3 else "normal")
        pm4.metric("Горизонт",       f"{horizon} дн.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist14["date"], y=hist14["balance"],
                                  name="Факт", line=dict(color="#58a6ff", width=2)))

        if not acc_fc.empty:
            # q90 optimistic
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q90"],
                                      name="q90 Оптимистичный", line=dict(color="rgba(63,185,80,0)"),
                                      fillcolor="rgba(63,185,80,0.12)", fill="tonexty", showlegend=True,
                                      legendgroup="ci"))
            # q10 pessimistic
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q10"],
                                      name="q10 Пессимистичный", line=dict(color="rgba(248,81,73,0)"),
                                      showlegend=True, legendgroup="ci"))
            # q50 median
            fig.add_trace(go.Scatter(x=acc_fc["date"], y=acc_fc["q50"],
                                      name="q50 Ожидаемый", mode="lines+markers",
                                      line=dict(color="#d29922", width=2, dash="dash")))

        fig.add_hline(y=acc_state["min_balance"],   line_dash="dash", line_color="#f85149", annotation_text="МИН")
        fig.add_hline(y=acc_state["target_balance"], line_dash="dot",  line_color="#3fb950", annotation_text="ЦЕЛЬ")
        fig.add_vline(x=df["date"].max().timestamp()*1000, line_color="#8b949e", annotation_text="Сегодня")

        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                          hovermode="x unified", legend=dict(orientation="h"),
                          xaxis_title="", yaxis_title=acc_state["currency"])
        st.plotly_chart(fig, use_container_width=True)

        if not acc_fc.empty:
            st.subheader("Таблица прогноза")
            tbl = acc_fc[["date","q10","q50","q90","predicted_inflow","predicted_outflow"]].copy()
            tbl.columns = ["Дата","q10 Пессим.","q50 Ожид.","q90 Оптим.","Приток","Отток"]
            for c in tbl.columns[1:]:
                tbl[c] = tbl[c].apply(lambda x: f"{x:,.0f}")
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────── TAB 3: Risk Engine ─────────────────────────
with tab3:
    st.subheader("Risk Engine — вероятность дефицита")

    # P(shortage) per account
    risk_rows = []
    for _, row in state_f.iterrows():
        p = model.p_shortage(row["account_id"], row["min_balance"], fc_f)
        risk_rows.append({
            "Счёт": row["account_name"],
            "Валюта": row["currency"],
            "Система": row["payment_system"],
            "P(дефицит)": p,
            "Уровень": "CRITICAL" if p > 0.3 else "HIGH" if p > 0.15 else "MEDIUM" if p > 0.05 else "LOW",
        })
    risk_df = pd.DataFrame(risk_rows)

    fig_risk = go.Figure()
    colors_p = [SEV_COLORS["CRITICAL"] if p > 0.3 else SEV_COLORS["HIGH"] if p > 0.15
                else SEV_COLORS["MEDIUM"] if p > 0.05 else SEV_COLORS["LOW"]
                for p in risk_df["P(дефицит)"]]
    fig_risk.add_trace(go.Bar(
        x=risk_df["Счёт"], y=risk_df["P(дефицит)"] * 100,
        marker_color=colors_p,
        text=[f"{v*100:.1f}%" for v in risk_df["P(дефицит)"]],
        textposition="outside",
    ))
    fig_risk.add_hline(y=30, line_dash="dash", line_color="#f85149", annotation_text="Порог CRITICAL (30%)")
    fig_risk.add_hline(y=10, line_dash="dot",  line_color="#d29922", annotation_text="Порог HIGH (10%)")
    fig_risk.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                           yaxis_title="P(дефицит) %", xaxis_title="",
                           xaxis_tickangle=-20)
    st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()
    st.subheader(f"Алерты с Explainable AI ({len(alerts)} шт.)")

    if not alerts:
        st.success("Нет активных алертов. Все счета в норме.")
    else:
        sev_cols = st.columns(4)
        for i, s in enumerate(["CRITICAL","HIGH","MEDIUM","LOW"]):
            sev_cols[i].metric(s, alert_sum.get(s, 0))
        st.divider()

        for al in alerts:
            sev = al["severity"]
            css = f"sev-{sev.lower()}"
            color = SEV_COLORS[sev]
            expl  = al.get("explanation", {})
            factors = expl.get("factors", [])
            risk_pct = expl.get("total_risk_pct", 0)
            time_str = f"{al['time_to_breach_h']}ч" if al.get("time_to_breach_h") else "Сейчас"

            st.markdown(f"""
<div class="{css}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <strong style="color:{color}">[{sev}] {al['type']}</strong>
    <span style="color:#8b949e;font-size:12px">⏱ {time_str} · {al['account_name']} ({al['currency']})</span>
  </div>
  <div style="margin-top:6px">{al['message']}</div>
  <div style="margin-top:4px;font-size:12px;color:#8b949e">💡 {al['action']}</div>
</div>""", unsafe_allow_html=True)

            if factors:
                with st.expander(f"🔍 Explainable AI — почему риск вырос на {risk_pct}%"):
                    for fname, fval in factors:
                        w = min(100, fval * 3)
                        fcolor = "#f85149" if fval >= 20 else "#d29922" if fval >= 10 else "#3fb950"
                        st.markdown(f"""
<div class="factor-row">
  <span>{fname}</span>
  <span style="color:{fcolor};font-weight:700">+{fval}%</span>
</div>
<div class="prob-bar-wrap"><div style="width:{w}%;background:{fcolor};height:10px;border-radius:8px"></div></div>
""", unsafe_allow_html=True)
                    st.markdown(f"**Общий вклад факторов: +{risk_pct}% к риску дефицита**")


# ─────────────────────────────── TAB 4: Rebalancing Hub ─────────────────────
with tab4:
    st.subheader("Rebalancing Hub — оптимизация ликвидности")

    ir1, ir2, ir3 = st.columns(3)
    ir1.metric("Всего ликвидности",    f"${idle['total_liquidity_usd']/1e6:.2f}M")
    ir2.metric("Заморожено (idle)",    f"${idle['total_idle_usd']/1e6:.2f}M",
               f"{idle['idle_pct']:.1f}% от общего")
    ir3.metric("Упущ. доход/год",      f"${idle['annual_opp_cost_usd']/1e3:.0f}K",
               "при 4.5% годовых")

    st.divider()

    if not recs:
        st.success("Все счета сбалансированы. Перераспределение не требуется.")
    else:
        confirmed_cnt = sum(1 for r in recs if r["id"] in st.session_state.confirmed_transfers)
        st.markdown(f"**{len(recs)} рекомендаций** · ✅ Подтверждено: {confirmed_cnt} / {len(recs)}")

        for i, rec in enumerate(recs):
            is_confirmed = rec["id"] in st.session_state.confirmed_transfers
            card_css = "rec-done" if is_confirmed else "rec-card"
            urg_css  = "tag-urgent" if rec["urgency"] == "НЕМЕДЛЕННО" else "tag-normal"
            fx_note  = f'→ {rec["amount_dest"]:,.0f} {rec["currency_to"]}' if rec["currency_from"] != rec["currency_to"] else ""

            st.markdown(f"""
<div class="{card_css}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <strong style="font-size:15px">{rec['from_account']} → {rec['to_account']}</strong>
    <span class="{urg_css}">{rec['urgency']}</span>
  </div>
  <div style="font-size:20px;font-weight:700;margin:10px 0;color:#e6edf3">
    💰 {rec['amount']:,.0f} {rec['currency_from']} {fx_note}
  </div>
  <div style="color:#8b949e;font-size:13px">
    ⏱ Время перевода: {rec['transfer_time_days']} дн. &nbsp;|&nbsp; Стоимость: ~{rec['estimated_cost']:,.0f} {rec['currency_from']} ({rec['cost_bps']} bps)
  </div>
  <div style="margin-top:4px;font-size:13px;color:#8b949e">{rec['reason']}</div>
</div>""", unsafe_allow_html=True)

            if not is_confirmed:
                btn_col, _ = st.columns([2, 5])
                if btn_col.button(f"✅ Подтвердить перевод #{i+1}", key=f"confirm_{i}", type="primary"):
                    st.session_state.confirmed_transfers[rec["id"]] = rec
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.success("✅ Execution request отправлен — перевод в процессе")


# ─────────────────────────────── TAB 5: What-If ──────────────────────────────
with tab5:
    st.subheader("What-If Симулятор — интерактивное моделирование")

    col_ctrl, col_res = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("### Параметры сценария")

        swift_off      = st.toggle("🔴 Отключить SWIFT", value=False)
        sepa_delay     = st.slider("SEPA доп. задержка (дни)", 0, 5, 0)
        card_delay     = st.slider("Card доп. задержка (дни)", 0, 5, 0)
        vol_spike      = st.slider("Пик объёма исходящих (%)", 0, 200, 0)
        eur_shock      = st.slider("EUR/USD шок (%)", -20, 20, 0)
        gbp_shock      = st.slider("GBP/USD шок (%)", -20, 20, 0)
        outage_names   = st.multiselect("Banking Outage (счета)", state_f["account_name"].tolist())
        outage_ids     = state_f[state_f["account_name"].isin(outage_names)]["account_id"].tolist()

        scenario_active = any([swift_off, sepa_delay, card_delay, vol_spike,
                                eur_shock != 0, gbp_shock != 0, outage_ids])
        if scenario_active:
            st.info("Результаты обновляются в реальном времени →")
        else:
            st.caption("Измените параметры чтобы запустить симуляцию")

    wi_df = compute_whatif(state_f, fc_f, {
        "swift_disabled":  swift_off,
        "sepa_extra_delay": sepa_delay,
        "card_extra_delay": card_delay,
        "volume_spike_pct": vol_spike,
        "eur_shock_pct":   eur_shock,
        "gbp_shock_pct":   gbp_shock,
        "outage_accounts": outage_ids,
    })

    with col_res:
        if wi_df.empty:
            st.info("Нет данных для симуляции.")
        else:
            total_delta  = wi_df["delta_usd"].sum()
            at_risk_cnt  = int(wi_df["at_risk"].sum())
            shortfall    = wi_df["shortfall_usd"].sum()
            sev_label    = ("🔴 КРИТИЧЕСКИЙ" if at_risk_cnt >= 3 or shortfall > 2e6
                            else "🟠 ВЫСОКИЙ" if at_risk_cnt >= 2 or shortfall > 5e5
                            else "🟡 СРЕДНИЙ" if at_risk_cnt >= 1
                            else "🟢 НИЗКИЙ")
            sev_color    = ("#f85149" if "КРИТИЧЕСКИЙ" in sev_label else
                            "#d29922" if "ВЫСОКИЙ" in sev_label or "СРЕДНИЙ" in sev_label else "#3fb950")

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Удар (USD)",       f"${total_delta/1e6:.2f}M",  delta_color="inverse" if total_delta < 0 else "normal")
            w2.metric("Счетов в риске",   str(at_risk_cnt))
            w3.metric("Дефицит (USD)",    f"${shortfall/1e3:.0f}K")
            w4.metric("Уровень риска",    sev_label)

            fig_wi = go.Figure()
            fig_wi.add_trace(go.Bar(name="Базовый прогноз", x=wi_df["account_name"],
                                    y=wi_df["base_end"], marker_color="#58a6ff"))
            fig_wi.add_trace(go.Bar(name="Стресс-сценарий",  x=wi_df["account_name"],
                                    y=wi_df["stressed_end"],
                                    marker_color=["#f85149" if r else "#d29922" for r in wi_df["at_risk"]]))
            for _, r in wi_df.iterrows():
                fig_wi.add_shape(type="line",
                                 x0=r["account_name"], x1=r["account_name"],
                                 y0=0, y1=r["min_balance"],
                                 line=dict(color="#f85149", width=1, dash="dot"))
            fig_wi.update_layout(barmode="group", height=350,
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="#e6edf3",
                                 xaxis_tickangle=-20, hovermode="x unified")
            st.plotly_chart(fig_wi, use_container_width=True)

            # Impact table
            tbl = wi_df[["account_name","currency","base_end","stressed_end","delta","shortfall","at_risk"]].copy()
            tbl.columns = ["Счёт","Валюта","База","Стресс","Изменение","Дефицит","Риск"]
            for c in ["База","Стресс","Изменение","Дефицит"]:
                tbl[c] = tbl[c].apply(lambda x: f"{x:,.0f}")
            tbl["Риск"] = tbl["Риск"].apply(lambda x: "🔴 ДА" if x else "🟢 НЕТ")
            st.dataframe(tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────── TAB 6: KPI & Analytics ─────────────────────
with tab6:
    st.subheader("KPI & Analytics")

    kl, kr = st.columns(2)
    with kl:
        # Liquidity efficiency over time
        st.markdown("#### Эффективность ликвидности (90 дн.)")
        h90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
        h90 = h90[h90["currency"].isin(currencies)]
        eff_series = h90.groupby("date").apply(lambda g: (
            (g["balance"] * g["currency"].map(_fx)).sum() /
            max((g["target_balance"] * g["currency"].map(_fx)).sum(), 1) * 100
        )).reset_index(name="eff")
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Scatter(x=eff_series["date"], y=eff_series["eff"],
                                      fill="tozeroy", line=dict(color="#3fb950", width=2),
                                      fillcolor="rgba(63,185,80,0.1)"))
        fig_eff.add_hline(y=100, line_dash="dot", line_color="#8b949e", annotation_text="100%")
        fig_eff.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                               yaxis_title="% от целевого")
        st.plotly_chart(fig_eff, use_container_width=True)

        # Idle capital breakdown
        st.markdown("#### Замороженный капитал по счетам")
        idle_df = pd.DataFrame(idle["details"])
        if not idle_df.empty:
            fig_idle = px.bar(idle_df.sort_values("idle_usd", ascending=False),
                              x="account", y=["balance_usd", "idle_usd"],
                              barmode="overlay",
                              color_discrete_map={"balance_usd":"#58a6ff","idle_usd":"#f85149"})
            fig_idle.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                                   xaxis_tickangle=-20, yaxis_title="USD")
            st.plotly_chart(fig_idle, use_container_width=True)

    with kr:
        # Inflow/Outflow heatmap (last 14 days × accounts)
        st.markdown("#### Приток/Отток (последние 14 дней)")
        h14 = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=14)) &
                  (df["currency"].isin(currencies))]
        pivot = h14.pivot_table(index="account_name", columns="date",
                                 values="net_flow", aggfunc="sum").fillna(0)
        if not pivot.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c.date()) for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale=[[0,"#f85149"],[0.5,"#21262d"],[1,"#3fb950"]],
                zmid=0,
            ))
            fig_hm.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e6edf3", margin=dict(t=10))
            st.plotly_chart(fig_hm, use_container_width=True)

        # KPI summary metrics
        st.markdown("#### Ключевые метрики системы")
        avg_pending = state_f["pending_inflow"].mean()
        avg_balance = state_f["balance"].mean()
        settlement_eff = max(0, 1 - avg_pending / max(avg_balance, 1)) * 100

        kpi_data = {
            "Эффективность ликвидности": f"{efficiency:.1f}%",
            "Settlement efficiency":      f"{settlement_eff:.1f}%",
            "Упущ. доход/год (idle)":    f"${idle['annual_opp_cost_usd']/1e3:.0f}K",
            "Счетов в дефиците":         str(int((state_f["deficit"] > 0).sum())),
            "Avg. clearing delay":        f"{np.mean([CLEARING_DAYS[ps] for ps in state_f['payment_system']]):.1f} дн.",
            "Алертов сгенерировано":     str(len(alerts)),
        }
        for label, val in kpi_data.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 12px;background:#161b22;border-radius:6px;margin:3px 0"><span style="color:#8b949e">{label}</span><strong style="color:#e6edf3">{val}</strong></div>', unsafe_allow_html=True)


# ─────────────────────────────── TAB 7: Data Input ───────────────────────────
with tab7:
    st.subheader("Управление данными")

    mode_idx = 1 if st.session_state.use_custom_data else 0
    data_mode = st.radio("Источник данных", ["🎲 Demo данные", "✏️ Свои данные"],
                          horizontal=True, index=mode_idx)
    new_mode = data_mode.startswith("✏️")
    if new_mode != st.session_state.use_custom_data:
        st.session_state.use_custom_data = new_mode
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    if not st.session_state.use_custom_data:
        st.markdown("### Demo счета")
        demo_tbl = pd.DataFrame([{
            "Счёт": a["name"], "Валюта": a["currency"], "Система": a["payment_system"],
            "Мин. баланс": f"{a['min_balance']:,.0f}", "Цел. баланс": f"{a['target_balance']:,.0f}",
            "Дн. объём": f"{a['daily_volume']:,.0f}",
        } for a in ACCOUNTS])
        st.dataframe(demo_tbl, use_container_width=True, hide_index=True)
        st.info("Переключитесь в **Свои данные** чтобы добавить свои счета.")
    else:
        l_col, r_col = st.columns([1, 1], gap="large")
        with l_col:
            st.markdown("### Добавить счёт")
            with st.form("add_acc_form", clear_on_submit=True):
                name = st.text_input("Название счёта *", placeholder="Мой банк USD")
                a1, a2 = st.columns(2)
                ccy = a1.selectbox("Валюта", ["USD","EUR","GBP"])
                ps  = a2.selectbox("Платёжная система", ["SWIFT","SEPA","CARD","LOCAL"])
                b1, b2 = st.columns(2)
                min_b   = b1.number_input("Мин. баланс",     0, value=100_000, step=10_000)
                tgt_b   = b2.number_input("Цел. баланс",     0, value=500_000, step=10_000)
                c1_, c2_ = st.columns(2)
                cur_b   = c1_.number_input("Текущий баланс", 0, value=350_000, step=10_000)
                dv      = c2_.number_input("Дн. объём",      1, value=200_000, step=10_000)
                if st.form_submit_button("➕ Добавить", type="primary", use_container_width=True):
                    if not name.strip():
                        st.error("Введите название счёта")
                    elif min_b >= tgt_b:
                        st.error("Целевой баланс > минимального")
                    else:
                        acc_id = f"user_{len(st.session_state.user_accounts)+1}_{name.strip()[:15].replace(' ','_').upper()}"
                        st.session_state.user_accounts.append({
                            "id": acc_id, "name": name.strip(), "currency": ccy,
                            "payment_system": ps, "min_balance": float(min_b),
                            "target_balance": float(tgt_b), "current_balance": float(cur_b),
                            "daily_volume": float(dv),
                        })
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.success(f"Счёт **{name.strip()}** добавлен!")
                        st.rerun()

        with r_col:
            st.markdown("### Курсы валют (к USD)")
            with st.form("fx_form"):
                eur_r = st.number_input("EUR/USD", 0.5, 2.0,
                                        value=st.session_state.custom_fx.get("EUR", 1.08),
                                        step=0.01, format="%.3f")
                gbp_r = st.number_input("GBP/USD", 0.5, 2.0,
                                        value=st.session_state.custom_fx.get("GBP", 1.27),
                                        step=0.01, format="%.3f")
                if st.form_submit_button("Применить", use_container_width=True):
                    st.session_state.custom_fx = {"USD":1.0,"EUR":eur_r,"GBP":gbp_r}
                    st.rerun()

        st.divider()
        if not st.session_state.user_accounts:
            st.info("Добавьте счёт слева.")
        else:
            st.markdown(f"### Мои счета ({len(st.session_state.user_accounts)})")
            for i, acc in enumerate(st.session_state.user_accounts):
                bal = acc["current_balance"]
                ok = "🟢" if bal >= acc["target_balance"]*0.8 else "🟡" if bal >= acc["min_balance"] else "🔴"
                with st.expander(f"{ok} {acc['name']} — {bal:,.0f} {acc['currency']}"):
                    ea, eb = st.columns([2, 2])
                    new_b = ea.number_input("Текущий баланс", 0.0, value=float(bal),
                                            step=float(max(1000, acc["daily_volume"]*0.01)),
                                            key=f"nb_{i}", format="%.0f")
                    eb.metric("Мин.", f"{acc['min_balance']:,.0f}")
                    st.caption(f"Цел: {acc['target_balance']:,.0f} | Система: {acc['payment_system']} | Дн.объём: {acc['daily_volume']:,.0f}")
                    uc, dc = st.columns([3,1])
                    if uc.button("💾 Сохранить", key=f"save_{i}", use_container_width=True):
                        st.session_state.user_accounts[i]["current_balance"] = new_b
                        st.rerun()
                    if dc.button("🗑 Удалить", key=f"del_{i}", use_container_width=True, type="secondary"):
                        st.session_state.user_accounts.pop(i)
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.rerun()
            if st.button("🗑 Очистить все счета", type="secondary"):
                st.session_state.user_accounts = []
                st.cache_data.clear(); st.cache_resource.clear()
                st.rerun()

        st.divider()
        st.markdown("### Загрузить CSV")
        st.caption("Формат: `date, account_name, currency, payment_system, inflow, outflow, balance`")
        uploaded = st.file_uploader("CSV файл", type=["csv"])
        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
                missing = {"date","account_name","currency","payment_system","inflow","outflow","balance"} - set(csv_df.columns)
                if missing:
                    st.error(f"Отсутствуют столбцы: {', '.join(missing)}")
                else:
                    st.success(f"Загружено {len(csv_df)} строк, {csv_df['account_name'].nunique()} счётов")
                    st.dataframe(csv_df.head(5), use_container_width=True, hide_index=True)
                    if st.button("Импортировать счета из CSV", type="primary"):
                        new_accs = []
                        for aname, grp in csv_df.groupby("account_name"):
                            grp = grp.sort_values("date")
                            new_accs.append({
                                "id": f"csv_{str(aname)[:15].replace(' ','_').upper()}",
                                "name": str(aname),
                                "currency": grp["currency"].iloc[0] if grp["currency"].iloc[0] in ("USD","EUR","GBP") else "USD",
                                "payment_system": grp["payment_system"].iloc[0] if grp["payment_system"].iloc[0] in ("SWIFT","SEPA","CARD","LOCAL") else "SWIFT",
                                "min_balance":    float(grp["balance"].quantile(0.05)),
                                "target_balance": float(grp["balance"].median()),
                                "current_balance":float(grp["balance"].iloc[-1]),
                                "daily_volume":   float((grp["inflow"]+grp["outflow"]).mean()),
                            })
                        st.session_state.user_accounts = new_accs
                        st.session_state.use_custom_data = True
                        st.cache_data.clear(); st.cache_resource.clear()
                        st.success(f"Создано {len(new_accs)} счётов")
                        st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div style="text-align:center;color:#8b949e;font-size:13px">FinFlow AI v2.0 · FinTech Hackathon 2026 · AI-driven Treasury Intelligence Platform</div>', unsafe_allow_html=True)
