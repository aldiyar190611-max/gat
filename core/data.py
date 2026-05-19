from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date, timedelta

ACCOUNTS = [
    {"id": "USD_JPM_SWIFT",   "name": "JPMorgan USD (SWIFT)",    "currency": "USD", "payment_system": "SWIFT", "min_balance": 600_000,  "target_balance": 2_500_000, "daily_volume": 1_000_000},
    {"id": "USD_STRIPE_CARD", "name": "Stripe USD (Card)",        "currency": "USD", "payment_system": "CARD",  "min_balance": 250_000,  "target_balance": 1_200_000, "daily_volume": 550_000},
    {"id": "EUR_DB_SEPA",     "name": "Deutsche Bank EUR (SEPA)", "currency": "EUR", "payment_system": "SEPA",  "min_balance": 400_000,  "target_balance": 1_800_000, "daily_volume": 720_000},
    {"id": "EUR_ADYEN_CARD",  "name": "Adyen EUR (Card)",         "currency": "EUR", "payment_system": "CARD",  "min_balance": 200_000,  "target_balance": 1_000_000, "daily_volume": 420_000},
    {"id": "GBP_BARCLAYS",    "name": "Barclays GBP (Local)",     "currency": "GBP", "payment_system": "LOCAL", "min_balance": 150_000,  "target_balance": 700_000,   "daily_volume": 280_000},
    {"id": "GBP_HSBC_SWIFT",  "name": "HSBC GBP (SWIFT)",         "currency": "GBP", "payment_system": "SWIFT", "min_balance": 120_000,  "target_balance": 500_000,   "daily_volume": 200_000},
]

FX_RATES     = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
CLEARING_DAYS = {"LOCAL": 0, "SEPA": 1, "SWIFT": 3, "CARD": 5}

CHANNEL_RELIABILITY = {"SWIFT": 0.94, "SEPA": 0.99, "CARD": 0.97, "LOCAL": 0.999}

BANK_HOLIDAYS = {
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 1), date(2024, 5, 1),
    date(2024, 5, 27), date(2024, 12, 25), date(2024, 12, 26),
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 12, 25), date(2025, 12, 26),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 6),
    date(2026, 5, 1), date(2026, 12, 25), date(2026, 12, 26),
}


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in BANK_HOLIDAYS


def _day_cashflow(account: dict, d: date, seed: int):
    rng = np.random.default_rng(seed)
    if not is_business_day(d):
        return rng.exponential(account["daily_volume"] * 0.04), rng.exponential(account["daily_volume"] * 0.02)
    dow_mult   = [1.20, 1.10, 1.00, 0.95, 0.85][d.weekday()]
    month_mult = {11: 1.25, 12: 1.35, 1: 0.80, 8: 0.85}.get(d.month, 1.0)
    period_mult = 1.5 if d.day >= 25 else (0.90 if d.day <= 5 else 1.0)
    vol = account["daily_volume"] * dow_mult * month_mult * period_mult
    return max(0.0, rng.normal(vol * 0.53, vol * 0.14)), max(0.0, rng.normal(vol * 0.47, vol * 0.12))


def generate_data(months: int = 12, end_date: date | None = None, accounts=None) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)
    accs = accounts if accounts is not None else ACCOUNTS

    records = []
    for ai, acc in enumerate(accs):
        rng0 = np.random.default_rng(ai * 999)
        balance = acc["target_balance"] * rng0.uniform(0.85, 1.15)
        cd = CLEARING_DAYS[acc["payment_system"]]
        pending_queue: list[tuple[date, float]] = []

        d = start_date
        while d <= end_date:
            seed = ai * 100_000 + (d - start_date).days
            inflow_raw, outflow = _day_cashflow(acc, d, seed)

            if cd > 0:
                vd = d + timedelta(days=cd)
                while not is_business_day(vd):
                    vd += timedelta(days=1)
                pending_queue.append((vd, inflow_raw))
                settled = sum(a for vd2, a in pending_queue if vd2 <= d)
                pending_queue = [(vd2, a) for vd2, a in pending_queue if vd2 > d]
            else:
                settled = inflow_raw

            pending_total = sum(a for _, a in pending_queue)
            balance = max(0.0, balance + settled - outflow)

            rng_s = np.random.default_rng(seed + 77)
            if rng_s.random() < 0.015:
                shock = acc["target_balance"] * rng_s.uniform(0.08, 0.25)
                balance = max(0.0, balance + rng_s.choice([-1, 1]) * shock)

            records.append({
                "date": pd.Timestamp(d),
                "account_id": acc["id"],
                "account_name": acc["name"],
                "currency": acc["currency"],
                "payment_system": acc["payment_system"],
                "inflow": settled,
                "outflow": outflow,
                "net_flow": settled - outflow,
                "pending_inflow": pending_total,
                "balance": balance,
                "min_balance": acc["min_balance"],
                "target_balance": acc["target_balance"],
                "daily_volume": acc["daily_volume"],
                "is_business_day": is_business_day(d),
            })
            d += timedelta(days=1)

    return pd.DataFrame(records)


def get_state(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = generate_data()
    last_date = df["date"].max()
    today = df[df["date"] == last_date].copy()
    rows = []
    for _, row in today.iterrows():
        bal = row["balance"]
        rows.append({
            "account_id":       row["account_id"],
            "account_name":     row["account_name"],
            "currency":         row["currency"],
            "payment_system":   row["payment_system"],
            "balance":          bal,
            "pending_inflow":   row["pending_inflow"],
            "available_balance": bal + row["pending_inflow"],
            "min_balance":      row["min_balance"],
            "target_balance":   row["target_balance"],
            "daily_volume":     row["daily_volume"],
            "excess":           max(0.0, bal - row["target_balance"]),
            "deficit":          max(0.0, row["min_balance"] - bal),
            "usd_equivalent":   bal * FX_RATES[row["currency"]],
        })
    return pd.DataFrame(rows)
