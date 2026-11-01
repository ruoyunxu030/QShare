"""Query helpers for the four stores and the scan-all baseline."""

from __future__ import annotations

import time

import numpy as np

from shared_evidence.lemma42 import Z_LB, delta_lemma42
from shared_evidence.store_bench import (
    N_Q, N_Q_SLOW, SEG, SLOW_S, TOP_K,
    _pct, _publish_c, _q1_a, _q1_b, _q2_ab, _q3_ab, hyper_full,
)
from shared_evidence.store_bench_query_time import _q1_ap, _q2_ap, _q3_ap
from shared_evidence.store_refresh import _theta

Q2_TH = 0.9


def _nq(sample_dt):
    return N_Q_SLOW if sample_dt > SLOW_S else N_Q


def _q1_sql(con, lo, hi):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = con.execute(
        "SELECT sigma2, mu0, tau0_2 FROM hyper WHERE id=1"
    ).fetchone()
    rows = con.execute(
        "SELECT seller, n, s1 FROM seller_stats WHERE seller BETWEEN ? AND ?",
        (int(lo), int(hi)),
    ).fetchall()
    scored = []
    for s, n, s1 in rows:
        m, _v, _lb = _publish_c(int(n), float(s1), float(sigma2), float(mu0), float(tau0_2))
        scored.append((m, int(s)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    hit = [s for _m, s in scored[:TOP_K]]
    return time.perf_counter() - t0, hit


def _q2_sql(con):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = _theta(con, "hyper")
    s2s, m0s, t2s = _theta(con, "hyper_snap")
    delta = delta_lemma42(mu0, tau0_2, sigma2, m0s, t2s, s2s, z=Z_LB)
    cands = con.execute(
        "SELECT seller, n, s1 FROM seller_stats WHERE lb > ?",
        (Q2_TH - delta,),
    ).fetchall()
    hit = []
    for s, n, s1 in cands:
        _m, _v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        if lb > Q2_TH:
            hit.append(int(s))
    return time.perf_counter() - t0, hit, int(len(cands)), float(delta)


def _q3_sql(con, n_sellers):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = con.execute(
        "SELECT sigma2, mu0, tau0_2 FROM hyper WHERE id=1"
    ).fetchone()
    rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
    by = {}
    for s, n, s1 in rows:
        m, _v, _lb = _publish_c(int(n), float(s1), float(sigma2), float(mu0), float(tau0_2))
        by[int(s)] = m
    out = []
    for lo in range(0, n_sellers, SEG):
        vals = [by[s] for s in range(lo, min(lo + SEG, n_sellers)) if s in by]
        if vals:
            out.append((lo, float(np.mean(vals)), float(np.max(vals))))
    return time.perf_counter() - t0, out


def _full_answers(con, n_sellers):
    rows = con.execute("SELECT seller, n, s1, s2 FROM seller_stats").fetchall()
    n, s1, s2 = {}, {}, {}
    for s, nj, sm, sm2 in rows:
        n[int(s)] = int(nj)
        s1[int(s)] = float(sm)
        s2[int(s)] = float(sm2)
    fs2, fmu, ftau = hyper_full(n, s1, s2)
    sigma2, mu0, tau0_2 = _theta(con, "hyper")
    full_m, acc_m = {}, {}
    q2_full, q2_acc = [], []
    for s, nj in n.items():
        m_f, v_f, lb_f = _publish_c(nj, s1[s], fs2, fmu, ftau)
        m_a, v_a, lb_a = _publish_c(nj, s1[s], sigma2, mu0, tau0_2)
        full_m[s] = m_f
        acc_m[s] = m_a
        if lb_f > Q2_TH:
            q2_full.append(s)
        if lb_a > Q2_TH:
            q2_acc.append(s)
    dm = [abs(acc_m[s] - full_m[s]) for s in acc_m]
    return dict(
        theta_acc=dict(sigma2=sigma2, mu0=mu0, tau0_2=tau0_2),
        theta_full=dict(sigma2=fs2, mu0=fmu, tau0_2=ftau),
        maxabs_d_m=float(np.max(dm) if dm else 0.0),
        q2_full=sorted(q2_full),
        q2_acc=sorted(q2_acc),
        q2_set_equal=sorted(q2_full) == sorted(q2_acc),
        q2_symdiff=int(len(set(q2_full) ^ set(q2_acc))),
        full_m=full_m,
        n_map=n,
        s1_map=s1,
    )


def _q2_scan(con, c=Q2_TH):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = _theta(con, "hyper")
    rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
    hit, lbs = [], []
    for s, n, s1 in rows:
        _m, _v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        lbs.append((int(s), float(lb)))
        if lb > c:
            hit.append(int(s))
    return time.perf_counter() - t0, hit, lbs


def _q2_filter_c(con, c):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = _theta(con, "hyper")
    s2s, m0s, t2s = _theta(con, "hyper_snap")
    delta = delta_lemma42(mu0, tau0_2, sigma2, m0s, t2s, s2s, z=Z_LB)
    cands = con.execute(
        "SELECT seller, n, s1 FROM seller_stats WHERE lb > ?",
        (float(c) - delta,),
    ).fetchall()
    hit = []
    for s, n, s1 in cands:
        _m, _v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        if lb > c:
            hit.append(int(s))
    return time.perf_counter() - t0, hit, int(len(cands)), float(delta)


def _interleave_c_ap(con_c, con_ap, n_sellers):
    n_seg = max(n_sellers // SEG, 1)
    dtc0, _ = _q1_sql(con_c, 0, SEG - 1)
    dta0, _ = _q1_ap(con_ap, 0, SEG - 1)
    nq1 = max(_nq(dtc0), _nq(dta0))
    q1_c, q1_ap, sidc0, sida0 = [], [], None, None
    for j in range(nq1):
        loj = (j % n_seg) * SEG
        hij = loj + SEG - 1
        dtc, sidc = _q1_sql(con_c, loj, hij)
        dta, sida = _q1_ap(con_ap, loj, hij)
        q1_c.append(dtc)
        q1_ap.append(dta)
        if j == 0:
            sidc0, sida0 = sidc, sida
    dtc0, hit_c, n_cand, delta = _q2_sql(con_c)
    dta0, hit_ap = _q2_ap(con_ap)
    nq2 = max(_nq(dtc0), _nq(dta0))
    q2_c, q2_ap = [], []
    for j in range(nq2):
        dtc, hc, nc, dlt = _q2_sql(con_c)
        dta, ha = _q2_ap(con_ap)
        q2_c.append(dtc)
        q2_ap.append(dta)
        if j == 0:
            hit_c, hit_ap, n_cand, delta = hc, ha, nc, dlt
    dtc0, _ = _q3_sql(con_c, n_sellers)
    dta0, _ = _q3_ap(con_ap, n_sellers)
    nq3 = max(_nq(dtc0), _nq(dta0))
    q3_c, q3_ap, q3_out_c = [], [], None
    for j in range(nq3):
        dtc, outc = _q3_sql(con_c, n_sellers)
        dta, outa = _q3_ap(con_ap, n_sellers)
        q3_c.append(dtc)
        q3_ap.append(dta)
        if j == 0:
            q3_out_c = outc
    return dict(
        q1_c=_pct(q1_c), q1_ap=_pct(q1_ap), q1_n=len(q1_c),
        q2_c=_pct(q2_c), q2_ap=_pct(q2_ap), q2_n=len(q2_c),
        q3_c=_pct(q3_c), q3_ap=_pct(q3_ap), q3_n=len(q3_c),
        q2_hit_c=sorted(hit_c), q2_hit_ap=sorted(hit_ap),
        q2_cand=int(n_cand), q2_delta=float(delta),
        q1_sid_c=sidc0, q1_sid_ap=sida0,
        q3_out_c=q3_out_c,
    )


def _time_ab(con, kind, n_sellers):
    n_seg = max(n_sellers // SEG, 1)
    q1 = _q1_a if kind == "a" else _q1_b
    dtc0, sid0 = q1(con, 0, SEG - 1)
    nq1 = _nq(dtc0)
    q1s = []
    for j in range(nq1):
        loj = (j % n_seg) * SEG
        hij = loj + SEG - 1
        dt, sid = q1(con, loj, hij)
        q1s.append(dt)
        if j == 0:
            sid0 = sid
    dt2, hit = _q2_ab(con, kind)
    nq2 = _nq(dt2)
    q2s, hit0 = [], hit
    for j in range(nq2):
        dt, hit = _q2_ab(con, kind)
        q2s.append(dt)
        if j == 0:
            hit0 = hit
    dt3, out = _q3_ab(con, kind, n_sellers)
    nq3 = _nq(dt3)
    q3s, out0 = [], out
    for j in range(nq3):
        dt, out = _q3_ab(con, kind, n_sellers)
        q3s.append(dt)
        if j == 0:
            out0 = out
    return dict(
        q1=_pct(q1s), q1_n=len(q1s), q1_sid=sid0,
        q2=_pct(q2s), q2_n=len(q2s), q2_hit=sorted(hit0),
        q3=_pct(q3s), q3_n=len(q3s), q3_out=out0,
    )


def _time_d(con, n_sellers):
    n_seg = max(n_sellers // SEG, 1)
    dtc0, sid0 = _q1_sql(con, 0, SEG - 1)
    nq1 = _nq(dtc0)
    q1s = []
    for j in range(nq1):
        loj = (j % n_seg) * SEG
        hij = loj + SEG - 1
        dt, sid = _q1_sql(con, loj, hij)
        q1s.append(dt)
        if j == 0:
            sid0 = sid
    dt2, hit, _lbs = _q2_scan(con, Q2_TH)
    nq2 = _nq(dt2)
    q2s, hit0 = [], hit
    for j in range(nq2):
        dt, hit, _ = _q2_scan(con, Q2_TH)
        q2s.append(dt)
        if j == 0:
            hit0 = hit
    dt3, out = _q3_sql(con, n_sellers)
    nq3 = _nq(dt3)
    q3s, out0 = [], out
    for j in range(nq3):
        dt, out = _q3_sql(con, n_sellers)
        q3s.append(dt)
        if j == 0:
            out0 = out
    return dict(
        q1=_pct(q1s), q1_n=len(q1s), q1_sid=sid0,
        q2=_pct(q2s), q2_n=len(q2s), q2_hit=sorted(hit0),
        q3=_pct(q3s), q3_n=len(q3s), q3_out=out0,
    )


def _selectivity(con, n_sellers):
    _dt, _hit, lbs = _q2_scan(con, -1e9)
    vals = np.array([lb for _s, lb in lbs], float)
    n1 = max(int(round(0.01 * n_sellers)), 1)
    order = np.argsort(-vals)
    c1 = float(vals[order[n1 - 1]]) if n1 <= len(vals) else Q2_TH
    rows = []
    for workload, c in (("threshold_0.9", Q2_TH), ("top1pct", c1)):
        dt0, hit_f, cand, delta = _q2_filter_c(con, c)
        nq = _nq(dt0)
        fts, sts = [], []
        hit_s = None
        for j in range(nq):
            dtf, hf, nc, dlt = _q2_filter_c(con, c)
            dts, hs, _ = _q2_scan(con, c)
            fts.append(dtf)
            sts.append(dts)
            if j == 0:
                hit_f, hit_s, cand, delta = hf, hs, nc, dlt
        rows.append(dict(
            name=workload, c=float(c),
            n_ans_filter=int(len(hit_f)), n_ans_scan=int(len(hit_s or [])),
            cand=int(cand), delta=float(delta),
            nrep=int(nq),
            q2_filter_median_s=_pct(fts)["median"], q2_filter_p95_s=_pct(fts)["p95"],
            q2_scan_median_s=_pct(sts)["median"], q2_scan_p95_s=_pct(sts)["p95"],
            sets_equal=sorted(hit_f) == sorted(hit_s or []),
            answer_frac=float(len(hit_f) / max(n_sellers, 1)),
        ))
    return rows, c1
