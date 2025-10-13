"""
metrics_pipeline.py — ETL & KPI pipeline (synthetic)
----------------------------------------------------
- Generates synthetic events and spend
- Computes KPIs: DAU/WAU/MAU, ARPU, MRR, churn, CAC, LTV, payback
- Simple CLI: export KPIs to CSV

Run examples
- python metrics_pipeline.py --preview
- python metrics_pipeline.py --export kpis.csv

Author: You
License: MIT
"""
from __future__ import annotations
import argparse, math, numpy as np, pandas as pd
from dataclasses import dataclass

np.random.seed(11)

@dataclass
class SimParams:
    start:str="2024-01-01"
    months:int=12
    base_users:int=12000
    growth:float=0.06
    arpu_mean:float=24.0
    arpu_sd:float=5.0
    churn_monthly:float=0.055
    cac:float=48.0

def simulate(p:SimParams=SimParams()):
    start = pd.Timestamp(p.start)
    months = [start + pd.DateOffset(months=i) for i in range(p.months)]
    users = []
    signups = []
    actives = []
    revenue = []
    churns = []
    spend = []

    current_users = p.base_users
    for i, m in enumerate(months):
        # growth with noise
        g = p.growth + np.random.normal(0, 0.01)
        new = int(max(0, current_users * (g)))
        signups.append(new)
        # churn
        churn = int((current_users + new) * max(0.01, p.churn_monthly + np.random.normal(0, 0.005)))
        churns.append(churn)
        current_users = current_users + new - churn
        users.append(max(0, current_users))
        # active users
        dau = int(current_users * (0.28 + np.random.uniform(-0.03, 0.03)))
        wau = int(current_users * (0.55 + np.random.uniform(-0.04, 0.04)))
        mau = int(current_users * (0.78 + np.random.uniform(-0.05, 0.05)))
        actives.append((dau, wau, mau))
        # revenue
        mrr = max(0, np.random.normal(p.arpu_mean, p.arpu_sd)) * mau * 0.25  # 25% of MAU are paid
        revenue.append(mrr)
        # spend
        spend.append(max(0, new * (p.cac + np.random.normal(0, 5))))

    df = pd.DataFrame({
        "month":[pd.Period(m, "M").strftime("%Y-%m") for m in months],
        "signups":signups,
        "churn":churns,
        "users_eom":users,
        "dau":[a[0] for a in actives],
        "wau":[a[1] for a in actives],
        "mau":[a[2] for a in actives],
        "mrr":[round(x,2) for x in revenue],
        "spend_acq":[round(x,2) for x in spend],
    })
    df["arpu"] = (df["mrr"] / df["mau"].clip(lower=1)).round(2)
    df["cac"] = (df["spend_acq"] / df["signups"].clip(lower=1)).round(2)
    df["ltv_simple"] = (df["arpu"] / max(0.01, 0.9 - 0.9*(1 - 0.055))).round(2)  # playful simple proxy
    df["payback_months"] = (df["cac"] / df["arpu"].replace(0, np.nan)).round(2)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--export", type=str, help="CSV path to export KPIs")
    args = ap.parse_args()
    df = simulate()
    if args.preview:
        print(df.head())
    if args.export:
        df.to_csv(args.export, index=False)

if __name__ == "__main__":
    main()
