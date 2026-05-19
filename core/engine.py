from __future__ import annotations
import pandas as pd
import numpy as np
from core.data import FX_RATES, CLEARING_DAYS, CHANNEL_RELIABILITY

# ── Severity helpers ──────────────────────────────────────────────────────────
SEV_COLORS = {"CRITICAL": "#FF4444", "HIGH": "#FF8800", "MEDIUM": "#FFCC00", "LOW": "#44AA44"}

def _sev(deficit_pct: float) -> str:
    if deficit_pct >= 0.5: return "CRITICAL"
    if deficit_pct >= 0.25: return "HIGH"
    if deficit_pct >= 0.10: return "MEDIUM"
    return "LOW"


# ── Explainable AI ────────────────────────────────────────────────────────────
def explain_risk(state_row: pd.Series, forecasts_acc: pd.DataFrame) -> dict:
    factors = []
    ps = state_row["payment_system"]
    bal = state_row["balance"]
    min_b = state_row["min_balance"]
    pi = state_row["pending_inflow"]

    if ps == "SWIFT":
        factors.append(("Задержки SWIFT (3 дня клиринг)", 18))
    elif ps == "CARD":
        factors.append(("Клиринг карт (до 5 дней)", 14))
    elif ps == "SEPA":
        factors.append(("SEPA задержка (1 день)", 6))

    if pi > bal * 0.5:
        factors.append(("Высокая зависимость от клиринга", 22))
    elif pi > bal * 0.3:
        factors.append(("Умеренный клиринговый риск", 10))

    buf = (bal - min_b) / max(min_b, 1)
    if buf < 0.1:
        factors.append(("Критически малый буфер", 28))
    elif buf < 0.3:
        factors.append(("Небольшой запас над минимумом", 16))

    if not forecasts_acc.empty:
        trend = forecasts_acc["predicted_net"].mean()
        if trend < -min_b * 0.05:
            factors.append(("Отрицательный тренд потоков", 15))
        elif trend < 0:
            factors.append(("Слабый отрицательный тренд", 7))

    rel = CHANNEL_RELIABILITY.get(ps, 0.97)
    if rel < 0.95:
        factors.append((f"Надёжность канала {ps}: {rel*100:.0f}%", int((1 - rel) * 200)))

    total_risk = min(100, sum(f[1] for f in factors))
    return {"total_risk_pct": total_risk, "factors": factors}


# ── Alert System ──────────────────────────────────────────────────────────────
class RiskEngine:
    def generate_alerts(self, state: pd.DataFrame, forecasts: pd.DataFrame) -> list[dict]:
        alerts = []
        for _, cs in state.iterrows():
            acc_id  = cs["account_id"]
            min_bal = cs["min_balance"]
            tgt_bal = cs["target_balance"]
            bal     = cs["balance"]
            acc_fc  = forecasts[forecasts["account_id"] == acc_id].sort_values("date") if not forecasts.empty else pd.DataFrame()

            if bal < min_bal:
                deficit = min_bal - bal
                expl = explain_risk(cs, acc_fc)
                alerts.append({
                    "account_id": acc_id, "account_name": cs["account_name"],
                    "currency": cs["currency"], "payment_system": cs["payment_system"],
                    "severity": _sev(deficit / max(min_bal, 1)),
                    "type": "CURRENT_DEFICIT",
                    "message": f"Баланс ниже минимума: {bal:,.0f} < {min_bal:,.0f} {cs['currency']}",
                    "deficit": deficit, "time_to_breach_h": 0,
                    "action": f"Немедленный перевод {deficit:,.0f} {cs['currency']}",
                    "explanation": expl,
                })

            for _, fc_row in acc_fc.iterrows():
                proj = fc_row["q50"]
                if proj < min_bal:
                    deficit = min_bal - proj
                    days = (fc_row["date"] - (acc_fc["date"].min() if not acc_fc.empty else fc_row["date"])).days + 1
                    expl = explain_risk(cs, acc_fc)
                    alerts.append({
                        "account_id": acc_id, "account_name": cs["account_name"],
                        "currency": cs["currency"], "payment_system": cs["payment_system"],
                        "severity": _sev(deficit / max(min_bal, 1)),
                        "type": "FORECAST_DEFICIT",
                        "message": f"Прогноз дефицита через {days} дн.: {proj:,.0f} < {min_bal:,.0f} {cs['currency']}",
                        "deficit": deficit, "time_to_breach_h": days * 24,
                        "action": f"Подготовить перевод {deficit:,.0f} {cs['currency']} не позднее чем через {max(0, days-1)} дн.",
                        "explanation": expl,
                    })

            if cs["pending_inflow"] > bal * 0.8 and bal < min_bal * 1.2:
                alerts.append({
                    "account_id": acc_id, "account_name": cs["account_name"],
                    "currency": cs["currency"], "payment_system": cs["payment_system"],
                    "severity": "MEDIUM", "type": "CLEARING_RISK",
                    "message": f"Высокий риск клиринга: {cs['pending_inflow']:,.0f} {cs['currency']} в обработке",
                    "deficit": 0, "time_to_breach_h": 24,
                    "action": "Проверить статус клиринга и держать резерв",
                    "explanation": explain_risk(cs, acc_fc),
                })

            if cs["excess"] > tgt_bal * 0.5:
                alerts.append({
                    "account_id": acc_id, "account_name": cs["account_name"],
                    "currency": cs["currency"], "payment_system": cs["payment_system"],
                    "severity": "LOW", "type": "EXCESS_IDLE",
                    "message": f"Избыточный остаток: {cs['excess']:,.0f} {cs['currency']} не работает",
                    "deficit": -cs["excess"], "time_to_breach_h": None,
                    "action": "Перевести излишек на доходный счёт",
                    "explanation": {"total_risk_pct": 5, "factors": [("Заморожен капитал", 5)]},
                })

        alerts.sort(key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}[x["severity"]])
        return alerts

    def summary(self, alerts: list[dict]) -> dict[str, int]:
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in alerts:
            counts[a["severity"]] = counts.get(a["severity"], 0) + 1
        return counts


# ── Liquidity Optimizer ───────────────────────────────────────────────────────
TRANSFER_COST_BPS = {
    ("USD","USD"):2, ("EUR","EUR"):1, ("GBP","GBP"):1,
    ("USD","EUR"):15,("USD","GBP"):18,("EUR","USD"):15,
    ("EUR","GBP"):12,("GBP","USD"):18,("GBP","EUR"):12,
}
TRANSFER_TIME = {"LOCAL":0,"SEPA":1,"SWIFT":2,"CARD":0}


class LiquidityOptimizer:
    def recommend(self, state: pd.DataFrame, forecasts: pd.DataFrame | None = None) -> list[dict]:
        surplus, deficit = [], []
        for _, row in state.iterrows():
            buf = row["min_balance"] * 0.15
            proj = row["balance"]
            if forecasts is not None and not forecasts.empty:
                fc = forecasts[forecasts["account_id"] == row["account_id"]]
                if not fc.empty:
                    proj = fc.iloc[-1]["q50"]
            ex = proj - (row["target_balance"] + buf)
            sh = (row["min_balance"] + buf) - proj
            if ex > 50_000:
                surplus.append({**row, "available_excess": ex, "proj_balance": proj})
            elif sh > 50_000:
                deficit.append({**row, "needed": sh, "proj_balance": proj})

        recs = []
        for d in deficit:
            needed = d["needed"]
            for s in surplus:
                if s["available_excess"] < 1_000:
                    continue
                amt = min(needed, s["available_excess"])
                cbps = TRANSFER_COST_BPS.get((s["currency"], d["currency"]),
                       TRANSFER_COST_BPS.get((d["currency"], s["currency"]), 20))
                ttime = max(TRANSFER_TIME[s["payment_system"]], TRANSFER_TIME[d["payment_system"]])
                amt_dest = amt * FX_RATES[s["currency"]] / FX_RATES[d["currency"]] if s["currency"] != d["currency"] else amt
                cost = amt * cbps / 10_000
                urgency = "НЕМЕДЛЕННО" if d["needed"] >= d["proj_balance"] * 0.5 else "В ТЕЧЕНИЕ 24Ч"
                recs.append({
                    "id": f"{s['account_id']}_{d['account_id']}_{amt:.0f}",
                    "from_account": s["account_name"], "from_id": s["account_id"],
                    "to_account": d["account_name"],   "to_id":   d["account_id"],
                    "amount": amt, "amount_dest": amt_dest,
                    "currency_from": s["currency"], "currency_to": d["currency"],
                    "transfer_time_days": ttime, "cost_bps": cbps, "estimated_cost": cost,
                    "urgency": urgency,
                    "reason": f"{d['account_name']} дефицит {d['needed']:,.0f} {d['currency']}. Перевод за {ttime} дн.",
                    "roi": (d["needed"] * 0.05 / 365) / max(cost, 1),
                })
                s["available_excess"] -= amt
                needed -= amt
                if needed <= 0:
                    break
        recs.sort(key=lambda x: (x["urgency"], -x["roi"]))
        return recs

    def idle_report(self, state: pd.DataFrame) -> dict:
        total_idle = total_liq = 0.0
        details = []
        for _, row in state.iterrows():
            usd = row["balance"] * FX_RATES[row["currency"]]
            tgt_usd = row["target_balance"] * FX_RATES[row["currency"]]
            idle = max(0.0, usd - tgt_usd)
            total_idle += idle
            total_liq  += usd
            details.append({"account": row["account_name"], "balance_usd": usd,
                             "idle_usd": idle, "idle_pct": idle/usd*100 if usd else 0})
        return {
            "total_liquidity_usd": total_liq,
            "total_idle_usd": total_idle,
            "idle_pct": total_idle / total_liq * 100 if total_liq else 0,
            "annual_opp_cost_usd": total_idle * 0.045,
            "details": details,
        }


# ── What-If Simulator ─────────────────────────────────────────────────────────
def compute_whatif(state: pd.DataFrame, forecasts: pd.DataFrame, params: dict) -> pd.DataFrame:
    rows = []
    for _, row in state.iterrows():
        acc_id = row["account_id"]
        ps = row["payment_system"]
        fc = forecasts[forecasts["account_id"] == acc_id] if not forecasts.empty else pd.DataFrame()
        base_end = fc.iloc[-1]["q50"] if not fc.empty else row["balance"]
        daily_in  = fc["predicted_inflow"].mean()  if not fc.empty else row["daily_volume"] * 0.53
        daily_out = fc["predicted_outflow"].mean() if not fc.empty else row["daily_volume"] * 0.47
        stressed = base_end

        if params.get("swift_disabled") and ps == "SWIFT":
            stressed -= daily_in * 3
        if ps == "SEPA" and params.get("sepa_extra_delay", 0) > 0:
            stressed -= daily_in * params["sepa_extra_delay"]
        if ps == "CARD" and params.get("card_extra_delay", 0) > 0:
            stressed -= daily_in * params["card_extra_delay"]
        if params.get("volume_spike_pct", 0) > 0:
            stressed -= daily_out * (params["volume_spike_pct"] / 100) * 3
        if row["currency"] == "EUR" and params.get("eur_shock_pct", 0) != 0:
            stressed *= (1 + params["eur_shock_pct"] / 100)
        if row["currency"] == "GBP" and params.get("gbp_shock_pct", 0) != 0:
            stressed *= (1 + params["gbp_shock_pct"] / 100)
        if acc_id in params.get("outage_accounts", []):
            stressed = 0.0

        stressed = max(0.0, stressed)
        min_bal  = row["min_balance"]
        rows.append({
            "account_id": acc_id, "account_name": row["account_name"],
            "currency": row["currency"], "payment_system": ps,
            "base_end": base_end, "stressed_end": stressed,
            "delta": stressed - base_end,
            "delta_usd": (stressed - base_end) * FX_RATES[row["currency"]],
            "at_risk": stressed < min_bal,
            "min_balance": min_bal,
            "shortfall": max(0.0, min_bal - stressed),
            "shortfall_usd": max(0.0, min_bal - stressed) * FX_RATES[row["currency"]],
        })
    return pd.DataFrame(rows)
