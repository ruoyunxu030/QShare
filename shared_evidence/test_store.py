#!/usr/bin/env python3
"""Incremental HyperAcc vs full recompute, and Q2 filter vs full answers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from shared_evidence.lemma42 import Z_LB, delta_lemma42
from shared_evidence.store_bench import SEED, TAU0_FLOOR, HyperAcc, hyper_full, make_stream
from shared_evidence.store_refresh import PRIOR, _moments

RS = (100, 1_000, 10_000)
ALPHAS = (0.8, 0.9, 0.95)
CS = (0.7, 0.8, 0.9)
ALPHA0 = 0.8


def z_alpha(a: float) -> float:
    if abs(float(a) - 0.8) < 1e-12:
        return float(Z_LB)
    return float(norm.ppf(float(a)))


def publish_mv(n, s1, sigma2, mu0, tau0_2):
    n = np.asarray(n, float)
    s1 = np.asarray(s1, float)
    m = np.full(n.shape, float(mu0))
    v = np.full(n.shape, float(tau0_2) ** 0.5)
    ok = n > 0
    if not np.any(ok):
        return m, v
    prec = 1.0 / max(float(tau0_2), 1e-15) + n[ok] / max(float(sigma2), 1e-15)
    m[ok] = (float(mu0) / max(float(tau0_2), 1e-15) + s1[ok] / max(float(sigma2), 1e-15)) / prec
    v[ok] = prec ** -0.5
    return m, v


def hyper_full_vec(n, s1, s2):
    n = np.asarray(n, float)
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    js = n >= 1
    js2 = n >= 2
    if not np.any(js2):
        sigma2 = 1.0 / 18.0
    else:
        sse = float(np.sum(s2[js2] - s1[js2] ** 2 / np.maximum(n[js2], 1e-15)))
        df = float(np.sum(n[js2] - 1.0))
        sigma2 = max(sse / max(df, 1.0), 1e-8)
    n_ev = int(js.sum())
    if n_ev < 2:
        return float(sigma2), 2.0 / 3.0, 1.0 / 18.0
    means = s1[js] / np.maximum(n[js], 1e-15)
    mu0 = float(means.mean())
    var_m = float(np.sum((means - mu0) ** 2) / max(n_ev - 1, 1))
    mean_se = float(np.sum(sigma2 / np.maximum(n[js], 1e-15)) / n_ev)
    tau0_2 = max(var_m - mean_se, TAU0_FLOOR)
    return float(sigma2), float(mu0), float(tau0_2)


def _remat(n_arr, s1_arr, idx, snap, mstar, vstar):
    if len(idx) == 0:
        return
    s2s, m0s, t2s = snap
    m, v = publish_mv(n_arr[idx], s1_arr[idx], s2s, m0s, t2s)
    mstar[idx] = m
    vstar[idx] = v


def test_hyper_acc_matches_full(seed=0, n_sellers=40, n_updates=200):
    rng = np.random.default_rng(seed)
    acc = HyperAcc()
    n, s1, s2 = {}, {}, {}
    for _ in range(n_updates):
        s = int(rng.integers(0, n_sellers))
        ph = float(rng.uniform(0.2, 0.99))
        acc.update(s, ph)
        n[s] = n.get(s, 0) + 1
        s1[s] = s1.get(s, 0.0) + ph
        s2[s] = s2.get(s, 0.0) + ph * ph
        got = acc.hyper()
        exp = hyper_full(n, s1, s2)
        for a, b in zip(got, exp):
            if abs(a - b) > 1e-9:
                raise AssertionError(f"mismatch {got} vs {exp} after update seller={s}")
    return True


def test_q2_vs_full(ns=(1000, 10_000, 100_000)):
    """4,932 query points: Q2 filter vs full recompute. Expect no misses."""
    z0 = z_alpha(ALPHA0)
    combos = [(a, c, z_alpha(a)) for a in ALPHAS for c in CS]
    n_query = 0
    n_miss = 0
    n_eq = 0
    n_rows = 0
    max_dm = 0.0
    for n in ns:
        n = int(n)
        rng = np.random.default_rng(SEED + [n])
        _pis, events = make_stream(n, rng)
        for ev in events:
            ev.pop("rows", None)
        n_writes = len(events)
        moments = {R: set(_moments(n_writes, R)) for R in RS}
        hyp = HyperAcc()
        n_arr = np.zeros(n, np.int32)
        s1_arr = np.zeros(n, float)
        s2_arr = np.zeros(n, float)
        m0, v0 = publish_mv(n_arr, s1_arr, *PRIOR)
        states = {R: dict(snap=PRIOR, mstar=m0.copy(), vstar=v0.copy()) for R in RS}
        theta = PRIOR
        for i, ev in enumerate(events, start=1):
            touched = []
            for ph, p2, s in ev["stats"]:
                s = int(s)
                hyp.update(s, float(ph))
                n_arr[s] += 1
                s1_arr[s] += float(ph)
                s2_arr[s] += float(p2)
                touched.append(s)
            theta = hyp.hyper()
            idx = np.asarray(touched, dtype=int)
            for R, st in states.items():
                _remat(n_arr, s1_arr, idx, st["snap"], st["mstar"], st["vstar"])
                if i % int(R) == 0:
                    st["snap"] = theta
                    m, v = publish_mv(n_arr, s1_arr, *theta)
                    st["mstar"] = m
                    st["vstar"] = v
            need = [R for R in RS if i in moments[R]]
            if not need:
                continue
            n_query += len(need)
            m, v = publish_mv(n_arr, s1_arr, *theta)
            fs2, fmu, ftau = hyper_full_vec(n_arr, s1_arr, s2_arr)
            mf, vf = publish_mv(n_arr, s1_arr, fs2, fmu, ftau)
            max_dm = max(max_dm, float(np.max(np.abs(m - mf))) if n else 0.0)
            for R in need:
                st = states[R]
                sigma2, mu0, tau0_2 = theta
                s2s, m0s, t2s = st["snap"]
                delta = delta_lemma42(mu0, tau0_2, sigma2, m0s, t2s, s2s, z=z0)
                lb_star = st["mstar"] - z0 * st["vstar"]
                for a, c, za in combos:
                    exact = (m - za * v) > c
                    cands = lb_star > (c - delta)
                    exact_f = (mf - za * vf) > c
                    n_miss += int(np.sum(exact & ~cands))
                    n_eq += int(bool(np.array_equal(exact, exact_f)))
                    n_rows += 1
        print(f"test_q2 N={n} points so far {n_query}", flush=True)
    if n_query != 4932:
        raise AssertionError(f"query points {n_query} != 4932")
    if n_miss:
        raise AssertionError(f"n_miss={n_miss}")
    if n_eq != n_rows:
        raise AssertionError(f"Q2 vs full {n_eq}/{n_rows}")
    print(f"test_q2_vs_full ok points={n_query} max|Δm|={max_dm:.3e}")
    return True


def main():
    test_hyper_acc_matches_full()
    test_hyper_acc_matches_full(seed=7, n_sellers=8, n_updates=50)
    test_q2_vs_full()
    print("test_store ok")


if __name__ == "__main__":
    main()
