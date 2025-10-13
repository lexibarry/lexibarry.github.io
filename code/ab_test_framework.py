"""
ab_test_framework.py — Two-proportion tests, CIs, power & sample size
---------------------------------------------------------------------
- Compute two-proportion z-tests (A/B conversion rates)
- Wilson score intervals
- Power and minimum sample size per variant
- Simple API + CLI

Run examples
- python ab_test_framework.py --rates 0.06 0.075 --n 2000 2000
- python ab_test_framework.py --mde 0.15 --baseline 0.06 --alpha 0.05 --power 0.8
"""
from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Tuple
from math import ceil
from statistics import NormalDist

@dataclass
class ABResult:
    p1: float; p2: float; n1: int; n2: int
    z: float; p_value: float
    ci1: Tuple[float,float]; ci2: Tuple[float,float]

def _z(p:float)->float:
    return NormalDist().inv_cdf(p)

def wilson_ci(k:int, n:int, alpha:float=0.05)->Tuple[float,float]:
    if n==0: return (0.0, 0.0)
    z = _z(1 - alpha/2)
    phat = k/n
    denom = 1 + z**2/n
    center = (phat + z*z/(2*n))/denom
    margin = (z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)))/denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def two_prop_z(k1:int, n1:int, k2:int, n2:int, alternative:str="two-sided")->ABResult:
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1 + k2)/(n1 + n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (p2 - p1)/se
    nd = NormalDist()
    if alternative=="two-sided":
        pval = 2*(1 - nd.cdf(abs(z)))
    elif alternative=="greater":
        pval = 1 - nd.cdf(z)
    else:
        pval = nd.cdf(z)
    return ABResult(
        p1=p1, p2=p2, n1=n1, n2=n2, z=z, p_value=pval,
        ci1=wilson_ci(k1,n1), ci2=wilson_ci(k2,n2)
    )

def required_n_per_variant(baseline:float, mde:float, alpha:float=0.05, power:float=0.8)->int:
    p1 = baseline
    p2 = baseline*(1+mde)
    z_alpha = _z(1 - alpha/2)
    z_beta = _z(power)
    se = math.sqrt(p1*(1-p1) + p2*(1-p2))
    n = ((z_alpha + z_beta)**2 * se**2) / (p2 - p1)**2
    return int(ceil(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs=2, type=float, help="p1 p2 (e.g., 0.06 0.075)")
    ap.add_argument("--n", nargs=2, type=int, help="n1 n2 sample sizes")
    ap.add_argument("--alternative", default="two-sided", choices=["two-sided","greater","less"])
    ap.add_argument("--mde", type=float, help="Relative lift (e.g., 0.1 = +10%)")
    ap.add_argument("--baseline", type=float, help="Baseline conversion")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    args = ap.parse_args()

    if args.rates and args.n:
        p1, p2 = args.rates
        n1, n2 = args.n
        k1, k2 = int(round(p1*n1)), int(round(p2*n2))
        res = two_prop_z(k1, n1, k2, n2, args.alternative)
        print(f"A: {res.p1:.4f} (n={res.n1}), CI={res.ci1[0]:.4f}-{res.ci1[1]:.4f}")
        print(f"B: {res.p2:.4f} (n={res.n2}), CI={res.ci2[0]:.4f}-{res.ci2[1]:.4f}")
        print(f"z={res.z:.3f}, p-value={res.p_value:.4f} ({args.alternative})")
    elif args.mde is not None and args.baseline is not None:
        n = required_n_per_variant(args.baseline, args.mde, args.alpha, args.power)
        print(f"Required n per variant ≈ {n} for baseline={args.baseline}, mde={args.mde}, α={args.alpha}, power={args.power}")
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
