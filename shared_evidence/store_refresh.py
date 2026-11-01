#!/usr/bin/env python3
"""QShare store: snapshot θ*, Lemma 4.2 bound table, queries between refreshes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shared_evidence.lemma42 import Z_LB, delta_lemma42
from shared_evidence.store_bench import (
    HyperAcc,
    SEED,
    _connect,
    _publish_c,
    make_stream,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "store"
Q2_TH = 0.9
N_Q2 = 20
NS = (1_000, 10_000, 100_000)
RS = (100, 1_000, 10_000)
PRIOR = (1.0 / 18.0, 2.0 / 3.0, 1.0 / 18.0)  # σ², μ0, τ0²


def _init_c(con, n_sellers: int):
    sigma2, mu0, tau0_2 = PRIOR
    m, v, lb = _publish_c(0, 0.0, sigma2, mu0, tau0_2)
    con.execute(
        "CREATE TABLE buyer_evidence ("
        "seller INTEGER, buyer INTEGER, t INTEGER, pi_hat REAL, n_rows INTEGER)"
    )
    con.execute(
        "CREATE TABLE seller_stats ("
        "seller INTEGER PRIMARY KEY, n INTEGER, s1 REAL, s2 REAL, "
        "m REAL, v REAL, lb REAL)"
    )
    con.execute("CREATE TABLE hyper (id INTEGER PRIMARY KEY, sigma2 REAL, mu0 REAL, tau0_2 REAL)")
    con.execute("CREATE TABLE hyper_snap (id INTEGER PRIMARY KEY, sigma2 REAL, mu0 REAL, tau0_2 REAL)")
    con.executemany(
        "INSERT INTO seller_stats VALUES (?,?,?,?,?,?,?)",
        [(i, 0, 0.0, 0.0, m, v, lb) for i in range(n_sellers)],
    )
    con.execute("INSERT INTO hyper VALUES (1, ?, ?, ?)", (sigma2, mu0, tau0_2))
    con.execute("INSERT INTO hyper_snap VALUES (1, ?, ?, ?)", (sigma2, mu0, tau0_2))
    con.execute("CREATE INDEX ix_c_lb ON seller_stats(lb)")
    con.commit()


def _theta(con, table="hyper"):
    return tuple(map(float, con.execute(f"SELECT sigma2, mu0, tau0_2 FROM {table} WHERE id=1").fetchone()))


def _write_one(con, hyp: HyperAcc, ev_rows, stats_upd, touched, snap):
    """Insert + update acc + rematerialize touched with snapshot θ*. No full refresh."""
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO buyer_evidence VALUES (?,?,?,?,?)", ev_rows)
    con.executemany(
        "UPDATE seller_stats SET n=n+1, s1=s1+?, s2=s2+? WHERE seller=?",
        stats_upd,
    )
    for ph, _p2, s in stats_upd:
        hyp.update(int(s), float(ph))
    sigma2, mu0, tau0_2 = hyp.hyper()
    con.execute("UPDATE hyper SET sigma2=?, mu0=?, tau0_2=? WHERE id=1", (sigma2, mu0, tau0_2))
    s2s, m0s, t2s = snap
    ts = [int(s) for s in touched]
    qmarks = ",".join("?" * len(ts)) if ts else ""
    if ts:
        fetched = con.execute(
            f"SELECT seller, n, s1 FROM seller_stats WHERE seller IN ({qmarks})",
            ts,
        ).fetchall()
        upd = []
        for s, n, s1 in fetched:
            m, v, lb = _publish_c(int(n), float(s1), s2s, m0s, t2s)
            upd.append((m, v, lb, int(s)))
        con.executemany("UPDATE seller_stats SET m=?, v=?, lb=? WHERE seller=?", upd)
    con.commit()
    return time.perf_counter() - t0, (sigma2, mu0, tau0_2)


def _refresh_all(con, theta):
    """θ* ← current θ; rematerialize every seller with θ*."""
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = theta
    con.execute("BEGIN")
    con.execute("UPDATE hyper_snap SET sigma2=?, mu0=?, tau0_2=? WHERE id=1", (sigma2, mu0, tau0_2))
    rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
    upd = []
    for s, n, s1 in rows:
        m, v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        upd.append((m, v, lb, int(s)))
    con.executemany("UPDATE seller_stats SET m=?, v=?, lb=? WHERE seller=?", upd)
    con.commit()
    return time.perf_counter() - t0


def _q2_filter(con, theta, snap):
    t0 = time.perf_counter()
    sigma2, mu0, tau0_2 = theta
    s2s, m0s, t2s = snap
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
    dt = time.perf_counter() - t0
    return dt, hit, int(len(cands)), float(delta)


def _q2_exact(con, theta):
    sigma2, mu0, tau0_2 = theta
    rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
    hit = []
    for s, n, s1 in rows:
        _m, _v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        if lb > Q2_TH:
            hit.append(int(s))
    return hit


def _moments(n_writes: int, R: int) -> list[int]:
    step = max(int(R) // 4, 1)
    if step > n_writes:
        raw = [n_writes // 4, n_writes // 2, (3 * n_writes) // 4, n_writes]
        return sorted({max(int(x), 1) for x in raw})
    out = list(range(step, n_writes + 1, step))
    if out[-1] != n_writes:
        out.append(n_writes)
    return out


def run_combo(n: int, R: int, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + [int(n)])
    print(f"refresh stream N={n} R={R} ...", flush=True)
    _pis, events = make_stream(n, rng)
    for ev in events:
        ev.pop("rows", None)
    moments = set(_moments(len(events), R))
    print(f"refresh N={n} R={R} events={len(events)} query_moments={sorted(moments)[:8]}...", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"refresh_{n}_{R}_"))
    db = tmp / "store.sqlite"
    con = _connect(db)
    _init_c(con, n)
    hyp = HyperAcc()
    snap = PRIOR
    writes, refresh_times = [], []
    qrows = []
    t_wall0 = time.perf_counter()
    for i, ev in enumerate(events, start=1):
        dt_w, theta = _write_one(con, hyp, ev["ev"], ev["stats"], ev["touched"], snap)
        writes.append(dt_w)
        if i % int(R) == 0:
            dt_r = _refresh_all(con, theta)
            refresh_times.append(dt_r)
            snap = theta
        if i not in moments:
            continue
        snap_now = _theta(con, "hyper_snap")
        theta_now = _theta(con, "hyper")
        exact = set(_q2_exact(con, theta_now))
        dts = []
        hit, cand_n, delta = [], 0, 0.0
        for _ in range(N_Q2):
            dti, hit, cand_n, delta = _q2_filter(con, theta_now, snap_now)
            dts.append(dti)
        got = set(hit)
        miss = len(exact - got)
        extra = len(got - exact)
        if int(R) > len(events):
            frac = float(i) / float(len(events))
            refreshed = 0
        elif i % int(R) == 0:
            frac = 1.0
            refreshed = 1
        else:
            frac = (i % int(R)) / float(R)
            refreshed = 0
        qrows.append(dict(
            n_sellers=int(n), R=int(R), writes=int(i),
            frac=float(frac),
            refreshed=int(refreshed),
            delta=float(delta),
            cand_n=int(cand_n), ans_n=int(len(got)),
            miss=int(miss), extra=int(extra),
            q2_median_s=float(np.median(dts)),
            q2_p95_s=float(np.quantile(dts, 0.95)),
            q2_n=int(len(dts)),
        ))
        print(
            f"  N={n} R={R} t={i} frac={frac:.2f} δ={delta:.4g} "
            f"|C|={cand_n} |A|={len(got)} miss={miss} extra={extra} "
            f"q2={np.median(dts)*1e3:.2f}ms",
            flush=True,
        )
    size = db.stat().st_size if db.exists() else 0
    con.close()
    try:
        os.remove(db)
        os.rmdir(tmp)
    except OSError:
        pass
    amort = float(np.mean(writes)) if writes else np.nan
    if refresh_times:
        amort = amort + float(np.sum(refresh_times)) / max(len(writes), 1)
    wrow = dict(
        n_sellers=int(n), R=int(R), n_writes=int(len(writes)),
        write_median_s=float(np.median(writes)) if writes else np.nan,
        write_p95_s=float(np.quantile(writes, 0.95)) if writes else np.nan,
        write_amort_mean_s=amort,
        refresh_med_s=float(np.median(refresh_times)) if refresh_times else 0.0,
        refresh_p95_s=float(np.quantile(refresh_times, 0.95)) if refresh_times else 0.0,
        refresh_n=int(len(refresh_times)),
        bytes=int(size),
        wall_s=time.perf_counter() - t_wall0,
        q_moments=int(len(qrows)),
    )
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(out_dir / f"store_refresh_n{n}_R{R}.csv", index=False)
    pd.DataFrame([wrow]).to_csv(out_dir / f"store_write_n{n}_R{R}.csv", index=False)
    (out_dir / f"store_n{n}_R{R}.json").write_text(json.dumps(
        dict(write=wrow, n_query=len(qrows),
             timestamp=datetime.now().isoformat(timespec="seconds")),
        indent=2, default=str,
    ) + "\n")
    print(
        f"refresh done N={n} R={R} wall={wrow['wall_s']:.1f}s "
        f"write_med={wrow['write_median_s']*1e3:.3f}ms "
        f"amort={wrow['write_amort_mean_s']*1e3:.3f}ms "
        f"refresh_n={wrow['refresh_n']} size={size/1e6:.2f}MB",
        flush=True,
    )
    return qdf, wrow


def small_concurrency(out_dir: Path, n: int = 1000, R: int = 100, n_proc: int = 3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"small solo N={n} R={R} ...", flush=True)
    _q, solo = run_combo(n, R, out_dir)
    shutil.copy2(out_dir / f"store_refresh_n{n}_R{R}.csv", out_dir / f"store_refresh_n{n}_R{R}_solo.csv")
    shutil.copy2(out_dir / f"store_write_n{n}_R{R}.csv", out_dir / f"store_write_n{n}_R{R}_solo.csv")
    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONUNBUFFERED",
    ):
        env.setdefault(k, "1")
    procs = []
    t0 = time.perf_counter()
    for i in range(n_proc):
        dest = out_dir / f"small_p{i}"
        dest.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "shared_evidence.store_refresh",
            "--n", str(n), "--R", str(R), "--out", str(dest),
        ]
        print(f"small parallel p{i}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    if any(c != 0 for c in codes):
        raise SystemExit(f"small workers failed {codes}")
    par_w = []
    par_q = []
    for i in range(n_proc):
        dest = out_dir / f"small_p{i}"
        par_w.append(pd.read_csv(dest / f"store_write_n{n}_R{R}.csv").iloc[0])
        q = pd.read_csv(dest / f"store_refresh_n{n}_R{R}.csv")
        par_q.append(float(q.q2_median_s.median()))
    write_ratio = float(np.mean([float(r.write_median_s) for r in par_w]) / max(solo["write_median_s"], 1e-12))
    q2_ratio = float(np.mean(par_q) / max(float(np.median(pd.read_csv(out_dir / f"store_refresh_n{n}_R{R}.csv").q2_median_s)), 1e-12))
    ok = write_ratio < 1.5 and q2_ratio < 1.5
    rec = dict(
        n=n, R=R, n_proc=n_proc, wall_s=wall,
        solo_write_ms=1e3 * solo["write_median_s"],
        par_write_ms=1e3 * float(np.mean([float(r.write_median_s) for r in par_w])),
        write_ratio=write_ratio,
        solo_q2_ms=1e3 * float(np.median(pd.read_csv(out_dir / f"store_refresh_n{n}_R{R}.csv").q2_median_s)),
        par_q2_ms=1e3 * float(np.mean(par_q)),
        q2_ratio=q2_ratio,
        concurrency=n_proc if ok else 1,
    )
    (out_dir / "small_concurrency.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("small", json.dumps(rec), flush=True)
    return rec


def combine(out_dir: Path):
    out_dir = Path(out_dir)
    qparts, wparts = [], []
    for n in NS:
        for R in RS:
            q = out_dir / f"store_refresh_n{n}_R{R}.csv"
            w = out_dir / f"store_write_n{n}_R{R}.csv"
            if q.exists():
                qparts.append(pd.read_csv(q))
            if w.exists():
                wparts.append(pd.read_csv(w))
    if qparts:
        pd.concat(qparts, ignore_index=True).to_csv(out_dir / "store_refresh.csv", index=False)
        print("wrote", out_dir / "store_refresh.csv", flush=True)
    if wparts:
        pd.concat(wparts, ignore_index=True).to_csv(out_dir / "store_write.csv", index=False)
        print("wrote", out_dir / "store_write.csv", flush=True)
