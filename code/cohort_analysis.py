"""
cohort_analysis.py — Cohort retention & LTV (synthetic, self-contained)
-----------------------------------------------------------------------
What this does
- Generates realistic synthetic signup + order data
- Builds monthly cohorts
- Computes retention matrix (% active by months since cohort)
- Estimates per-cohort LTV with simple assumptions
- Exposes helper functions and a CLI entrypoint

Run examples
- python cohort_analysis.py --preview
- python cohort_analysis.py --export retention.csv ltv_by_cohort.csv

Author: You
License: MIT
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

np.random.seed(7)

@dataclass
class SimParams:
    n_users:int=2500
    start:str="2024-01-01"
    months:int=12
    base_retention:float=0.8        # baseline monthly retention probability
    cohort_decay:float=0.012        # slightly lower retention for later cohorts
    arpu_mean:float=32.0
    arpu_sd:float=7.0

def _month_floor(dt:pd.Series)->pd.Series:
    return pd.to_datetime(dt).values.astype('datetime64[M]').astype('datetime64[ns]')

def simulate_data(p:SimParams=SimParams())->Tuple[pd.DataFrame,pd.DataFrame]:
    start = pd.Timestamp(p.start)
    # distribute signups with mild growth
    probs = np.linspace(1.0, 1.8, p.months)
    probs = probs / probs.sum()
    signup_months = np.random.choice(p.months, size=p.n_users, p=probs)
    signups = pd.DataFrame({
        "user_id": np.arange(1, p.n_users+1),
        "signup_dt": [start + pd.DateOffset(months=int(m), days=int(np.random.randint(0,28))) for m in signup_months]
    })
    signups["cohort_month"] = _month_floor(signups["signup_dt"])

    # monthly activity & revenue
    rows = []
    for _, r in signups.iterrows():
        cohort_ix = ((r.signup_dt.to_period("M") - start.to_period("M")).n)
        for m in range(cohort_ix, p.months):
            month = start + pd.DateOffset(months=m)
            rel = m - cohort_ix
            pr = (p.base_retention**rel) * (1 - p.cohort_decay*cohort_ix)
            if np.random.rand() < pr:
                revenue = max(0, np.random.normal(p.arpu_mean, p.arpu_sd))
                rows.append((r.user_id, month, revenue))
    activity = pd.DataFrame(rows, columns=["user_id","month","revenue"])
    activity["month"] = _month_floor(activity["month"])
    return signups, activity

def build_retention(signups:pd.DataFrame, activity:pd.DataFrame)->pd.DataFrame:
    activity = activity.copy()
    activity["cohort_month"] = activity["user_id"].map(signups.set_index("user_id")["cohort_month"])
    activity["period"] = ((activity["month"].dt.to_period("M") - activity["cohort_month"].dt.to_period("M")).apply(lambda p: p.n)).astype(int)
    cohort_sizes = signups.groupby("cohort_month")["user_id"].nunique()
    actives = activity.groupby(["cohort_month","period"])["user_id"].nunique().unstack(fill_value=0)
    retention = actives.div(cohort_sizes, axis=0).round(3)
    retention.index.name = "cohort_month"
    return retention

def ltv_by_cohort(signups:pd.DataFrame, activity:pd.DataFrame)->pd.DataFrame:
    activity = activity.copy()
    activity["cohort_month"] = activity["user_id"].map(signups.set_index("user_id")["cohort_month"])
    ltv = activity.groupby("cohort_month")["revenue"].sum().div(
        signups.groupby("cohort_month")["user_id"].nunique()
    ).to_frame("ltv").round(2)
    ltv.index.name = "cohort_month"
    return ltv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="Print head/tail of outputs")
    ap.add_argument("--export", nargs="+", help="Export retention and ltv CSVs (provide 1 or 2 paths).")
    args = ap.parse_args()

    signups, activity = simulate_data()
    retention = build_retention(signups, activity)
    ltv = ltv_by_cohort(signups, activity)

    if args.preview:
        print("Retention (first 6 rows):")
        print(retention.head(6))
        print("\nLTV by Cohort:")
        print(ltv.head(10))

    if args.export:
        if len(args.export)==1:
            retention.to_csv(args.export[0], index=True)
        elif len(args.export)>=2:
            retention.to_csv(args.export[0], index=True)
            ltv.to_csv(args.export[1], index=True)

if __name__ == "__main__":
    main()
