from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import norm as sp_norm

from core.data import CLEARING_DAYS, is_business_day

FEATURE_COLS = [
    "dow", "dom", "month", "is_weekend", "is_month_end", "is_month_start", "is_biz",
    "net_lag1", "net_lag2", "net_lag3", "net_lag7", "net_lag14",
    "in_lag1",  "in_lag2",  "in_lag3",  "in_lag7",
    "out_lag1", "out_lag2", "out_lag3", "out_lag7",
    "net_roll7", "net_roll14", "net_roll30",
    "out_roll7", "in_roll7", "bal_lag1",
]


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["dow"]           = df["date"].dt.dayofweek
    df["dom"]           = df["date"].dt.day
    df["month"]         = df["date"].dt.month
    df["is_weekend"]    = (df["dow"] >= 5).astype(int)
    df["is_month_end"]  = (df["dom"] >= 25).astype(int)
    df["is_month_start"]= (df["dom"] <= 5).astype(int)
    df["is_biz"]        = df["is_business_day"].astype(int)
    for lag in [1, 2, 3, 7, 14]:
        df[f"net_lag{lag}"] = df["net_flow"].shift(lag)
        df[f"in_lag{lag}"]  = df["inflow"].shift(lag)
        df[f"out_lag{lag}"] = df["outflow"].shift(lag)
    df["net_roll7"]  = df["net_flow"].shift(1).rolling(7).mean()
    df["net_roll14"] = df["net_flow"].shift(1).rolling(14).mean()
    df["net_roll30"] = df["net_flow"].shift(1).rolling(30).mean()
    df["out_roll7"]  = df["outflow"].shift(1).rolling(7).mean()
    df["in_roll7"]   = df["inflow"].shift(1).rolling(7).mean()
    df["bal_lag1"]   = df["balance"].shift(1)
    return df.dropna()


class CashFlowForecaster:
    def __init__(self):
        self._rf:    dict[str, RandomForestRegressor]       = {}
        self._gbm10: dict[str, GradientBoostingRegressor]   = {}
        self._gbm90: dict[str, GradientBoostingRegressor]   = {}
        self._hist:  dict[str, pd.DataFrame]                = {}
        self.trained = False

    def train(self, df: pd.DataFrame) -> None:
        for acc_id in df["account_id"].unique():
            sub = _make_features(df[df["account_id"] == acc_id].copy())
            if len(sub) < 30:
                continue
            X = sub[FEATURE_COLS]
            y = sub["net_flow"]
            self._rf[acc_id]    = RandomForestRegressor(n_estimators=120, max_depth=7, random_state=42, n_jobs=-1).fit(X, y)
            self._gbm10[acc_id] = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=80, max_depth=4, random_state=42).fit(X, y)
            self._gbm90[acc_id] = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=80, max_depth=4, random_state=42).fit(X, y)
            self._hist[acc_id]  = sub.tail(30)
        self.trained = True

    def forecast(self, acc_id: str, days: int = 3, current_balance: float | None = None) -> pd.DataFrame:
        if acc_id not in self._rf:
            return pd.DataFrame()

        hist = self._hist[acc_id].copy()
        acc_name = hist["account_name"].iloc[0]
        acc_ccy  = hist["currency"].iloc[0]
        acc_ps   = hist["payment_system"].iloc[0]
        cd       = CLEARING_DAYS[acc_ps]
        bal      = current_balance if current_balance is not None else hist["balance"].iloc[-1]

        rows, last_date = [], hist["date"].iloc[-1]
        for i in range(1, days + 1):
            fdate  = last_date + timedelta(days=i)
            is_biz = is_business_day(fdate.date())
            tail   = hist.tail(14)

            feats = {
                "dow": fdate.dayofweek, "dom": fdate.day, "month": fdate.month,
                "is_weekend": int(fdate.dayofweek >= 5),
                "is_month_end": int(fdate.day >= 25),
                "is_month_start": int(fdate.day <= 5),
                "is_biz": int(is_biz),
                **{f"net_lag{l}": tail["net_flow"].values[-l] if len(tail) >= l else 0 for l in [1,2,3,7,14]},
                **{f"in_lag{l}":  tail["inflow"].values[-l]   if len(tail) >= l else 0 for l in [1,2,3,7]},
                **{f"out_lag{l}": tail["outflow"].values[-l]  if len(tail) >= l else 0 for l in [1,2,3,7]},
                "net_roll7":  float(np.mean(tail["net_flow"].values[-7:])) if len(tail) >= 7 else float(np.mean(tail["net_flow"].values)),
                "net_roll14": float(np.mean(tail["net_flow"].values[-14:])) if len(tail) >= 14 else float(np.mean(tail["net_flow"].values)),
                "net_roll30": float(np.mean(tail["net_flow"].values)),
                "out_roll7":  float(np.mean(tail["outflow"].values[-7:])) if len(tail) >= 7 else float(np.mean(tail["outflow"].values)),
                "in_roll7":   float(np.mean(tail["inflow"].values[-7:])) if len(tail) >= 7 else float(np.mean(tail["inflow"].values)),
                "bal_lag1":   bal,
            }

            X = pd.DataFrame([feats])[FEATURE_COLS]
            pred_net = float(self._rf[acc_id].predict(X)[0])
            net_q10  = float(self._gbm10[acc_id].predict(X)[0])
            net_q90  = float(self._gbm90[acc_id].predict(X)[0])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree_preds = np.array([t.predict(X.values)[0] for t in self._rf[acc_id].estimators_])
            std = float(tree_preds.std())

            eff = pred_net if cd == 0 else pred_net * max(0, 1 - cd * 0.15)
            bal = max(0.0, bal + eff)

            rows.append({
                "date": fdate, "account_id": acc_id,
                "account_name": acc_name, "currency": acc_ccy, "payment_system": acc_ps,
                "q50": bal,
                "q10": max(0.0, bal + (net_q10 - pred_net) * days),
                "q90": max(0.0, bal + (net_q90 - pred_net) * days),
                "std": max(std, 1.0),
                "predicted_inflow":  max(0.0, float(hist["inflow"].tail(7).mean())),
                "predicted_outflow": max(0.0, float(hist["outflow"].tail(7).mean())),
                "predicted_net": pred_net,
                "clearing_delay": cd,
            })

            new_row = hist.iloc[-1].copy()
            new_row["date"] = fdate
            new_row["net_flow"] = pred_net
            new_row["balance"]  = bal
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

        return pd.DataFrame(rows)

    def forecast_all(self, days: int = 3, current_balances: dict | None = None) -> pd.DataFrame:
        frames = []
        for acc_id in self._rf:
            bal = current_balances.get(acc_id) if current_balances else None
            frames.append(self.forecast(acc_id, days=days, current_balance=bal))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def p_shortage(self, acc_id: str, min_balance: float, forecasts: pd.DataFrame) -> float:
        fc = forecasts[forecasts["account_id"] == acc_id]
        if fc.empty:
            return 0.0
        last = fc.iloc[-1]
        if last["std"] <= 0:
            return 1.0 if last["q50"] < min_balance else 0.0
        return float(sp_norm.cdf(min_balance, loc=last["q50"], scale=last["std"] * np.sqrt(len(fc))))
