#!/usr/bin/env python3
"""Check lemma 4.2: |lb(θ)-lb(θ*)| ≤ δ on random samples."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_evidence.lemma42 import Z_LB, delta_lemma42
from shared_evidence.store_bench import _publish_c

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "store"
N_RAND = 4000
N_EDGE = 200
SEED = 0


def _one(mu0, tau, sig2, mu0s, taus, sig2s, n, xbar):
    s1 = float(n) * float(xbar)
    _m, _v, lb = _publish_c(int(n), s1, float(sig2), float(mu0), float(tau))
    _ms, _vs, lbs = _publish_c(int(n), s1, float(sig2s), float(mu0s), float(taus))
    d = delta_lemma42(mu0, tau, sig2, mu0s, taus, sig2s, z=Z_LB)
    gap = abs(lb - lbs)
    ratio = gap / d if d > 0 else (0.0 if gap <= 1e-15 else float("inf"))
    ok = gap <= d + 1e-12
    return ok, gap, d, ratio


def run_test(seed: int = SEED):
    rng = np.random.default_rng(seed)
    recs = []
    n_ok = 0
    n_tot = 0
    max_ratio = 0.0
    max_gap = 0.0
    worst = None

    def add(mu0, tau, sig2, mu0s, taus, sig2s, n, xbar, tag):
        nonlocal n_ok, n_tot, max_ratio, max_gap, worst
        ok, gap, d, ratio = _one(mu0, tau, sig2, mu0s, taus, sig2s, n, xbar)
        n_tot += 1
        n_ok += int(ok)
        if ratio > max_ratio:
            max_ratio = float(ratio)
            worst = dict(
                tag=tag, n=int(n), xbar=float(xbar),
                mu0=float(mu0), tau0_2=float(tau), sigma2=float(sig2),
                mu0_s=float(mu0s), tau0_2_s=float(taus), sigma2_s=float(sig2s),
                gap=float(gap), delta=float(d), ratio=float(ratio), ok=bool(ok),
            )
        max_gap = max(max_gap, float(gap))
        recs.append(ok)

    for i in range(N_RAND):
        add(
            rng.uniform(0, 1), rng.uniform(0.0025, 0.1), rng.uniform(0.001, 0.1),
            rng.uniform(0, 1), rng.uniform(0.0025, 0.1), rng.uniform(0.001, 0.1),
            int(rng.integers(0, 201)), rng.uniform(0, 1), "rand",
        )
    for n in (0, 1, 2, 10, 50, 200):
        for xbar in (0.0, 0.5, 1.0):
            for _ in range(N_EDGE // 18 + 1):
                add(
                    rng.uniform(0, 1), rng.uniform(0.0025, 0.1), rng.uniform(0.001, 0.1),
                    rng.uniform(0, 1), rng.uniform(0.0025, 0.1), rng.uniform(0.001, 0.1),
                    n, xbar, f"edge_n{n}",
                )
    out = dict(
        n_ok=n_ok, n_tot=n_tot, n_fail=n_tot - n_ok,
        max_ratio=max_ratio, max_gap=max_gap, worst=worst,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = run_test()
    (OUT_DIR / "lemma_test.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(
        f"lemma42 n_ok={rec['n_ok']}/{rec['n_tot']} fail={rec['n_fail']} "
        f"max |Δlb|/δ={rec['max_ratio']:.6g} max|Δlb|={rec['max_gap']:.6g}",
        flush=True,
    )
    if rec["worst"]:
        print("worst", rec["worst"], flush=True)


if __name__ == "__main__":
    main()
