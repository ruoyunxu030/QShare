#!/usr/bin/env python3
"""Record-log helpers: aggregate buyer records at query time."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shared_evidence.store_bench import (
    HyperAcc,
    N_Q,
    N_Q_SLOW,
    NS,
    SEED,
    SEG,
    SLOW_S,
    TOP_K,
    WALL_CAP_S,
    _connect,
    _jaccard,
    _pct,
    _publish_c,
    hyper_full,
    make_stream,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "store"
STORE_DIR = ROOT / "results" / "store"
Q2_TH = 0.9


def _init_ap(con):
    con.execute(
        "CREATE TABLE buyer_evidence ("
        "seller INTEGER, buyer INTEGER, t INTEGER, pi_hat REAL, n_rows INTEGER)"
    )
    con.commit()


def _write_ap(con, ev_rows):
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO buyer_evidence VALUES (?,?,?,?,?)", ev_rows)
    con.commit()
    return time.perf_counter() - t0


def _agg_ap(con):
    """Full-table GROUP BY to (n, s1, s2) and (σ², μ0, τ0²). Same formulas as (c)."""
    rows = con.execute(
        "SELECT seller, COUNT(*), SUM(pi_hat), SUM(pi_hat * pi_hat) "
        "FROM buyer_evidence GROUP BY seller"
    ).fetchall()
    n, s1, s2 = {}, {}, {}
    for s, nj, sm, sm2 in rows:
        n[int(s)] = int(nj)
        s1[int(s)] = float(sm)
        s2[int(s)] = float(sm2)
    sigma2, mu0, tau0_2 = hyper_full(n, s1, s2)
    return n, s1, s2, float(sigma2), float(mu0), float(tau0_2)


def _q1_ap(con, lo, hi):
    t0 = time.perf_counter()
    n, s1, _s2, sigma2, mu0, tau0_2 = _agg_ap(con)
    scored = []
    for s in range(int(lo), int(hi) + 1):
        m, _v, _lb = _publish_c(n.get(s, 0), s1.get(s, 0.0), sigma2, mu0, tau0_2)
        scored.append((m, int(s)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return time.perf_counter() - t0, [s for _m, s in scored[:TOP_K]]


def _q2_ap(con):
    t0 = time.perf_counter()
    n, s1, _s2, sigma2, mu0, tau0_2 = _agg_ap(con)
    hit = []
    for s, nj in n.items():
        _m, _v, lb = _publish_c(nj, s1[s], sigma2, mu0, tau0_2)
        if lb > Q2_TH:
            hit.append(int(s))
    return time.perf_counter() - t0, hit


def _q3_ap(con, n_sellers):
    t0 = time.perf_counter()
    n, s1, _s2, sigma2, mu0, tau0_2 = _agg_ap(con)
    out = []
    for lo in range(0, n_sellers, SEG):
        vals = []
        for s in range(lo, min(lo + SEG, n_sellers)):
            m, _v, _lb = _publish_c(n.get(s, 0), s1.get(s, 0.0), sigma2, mu0, tau0_2)
            vals.append(m)
        if vals:
            out.append((lo, float(np.mean(vals)), float(np.max(vals))))
    return time.perf_counter() - t0, out


def _c_answers(events, n_sellers):
    """What exact (c) returns after the same stream (HyperAcc + publish)."""
    hyp = HyperAcc()
    n, s1 = {}, {}
    for ev in events:
        for ph, _p2, s in ev["stats"]:
            s = int(s)
            hyp.update(s, float(ph))
            n[s] = n.get(s, 0) + 1
            s1[s] = s1.get(s, 0.0) + float(ph)
    sigma2, mu0, tau0_2 = hyp.hyper()
    n_seg = max(n_sellers // SEG, 1)

    def q1(lo, hi):
        scored = []
        for s in range(int(lo), int(hi) + 1):
            m, _v, _lb = _publish_c(n.get(s, 0), s1.get(s, 0.0), sigma2, mu0, tau0_2)
            scored.append((m, int(s)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _m, s in scored[:TOP_K]]

    hit = []
    for s, nj in n.items():
        _m, _v, lb = _publish_c(nj, s1[s], sigma2, mu0, tau0_2)
        if lb > Q2_TH:
            hit.append(int(s))
    q3 = []
    for lo in range(0, n_sellers, SEG):
        vals = []
        for s in range(lo, min(lo + SEG, n_sellers)):
            m, _v, _lb = _publish_c(n.get(s, 0), s1.get(s, 0.0), sigma2, mu0, tau0_2)
            vals.append(m)
        if vals:
            q3.append((lo, float(np.mean(vals)), float(np.max(vals))))
    return dict(q1=q1, q2=hit, q3=q3, n_seg=n_seg, hyper=(sigma2, mu0, tau0_2))


def _run_ap(n_sellers, events):
    tmp = Path(tempfile.mkdtemp(prefix=f"ap_{n_sellers}_"))
    db = tmp / "store.sqlite"
    con = _connect(db)
    _init_ap(con)
    writes = []
    t_wall0 = time.perf_counter()
    stopped = False
    n_done = 0
    for ev in events:
        n_done += 1
        writes.append(_write_ap(con, ev["ev"]))
        if time.perf_counter() - t_wall0 > WALL_CAP_S:
            stopped = True
            break
    n_seg = max(n_sellers // SEG, 1)
    q1s, q2s, q3s = [], [], []
    q1_sets = []
    dt, _sids = _q1_ap(con, 0, SEG - 1)
    nq = N_Q_SLOW if dt > SLOW_S else N_Q
    for j in range(nq):
        loj = (j % n_seg) * SEG
        hij = loj + SEG - 1
        dti, sidj = _q1_ap(con, loj, hij)
        q1s.append(dti)
        if j < 20:
            q1_sets.append((loj, sidj))
        if time.perf_counter() - t_wall0 > WALL_CAP_S:
            stopped = True
            break
    dt, hit = _q2_ap(con)
    nq = N_Q_SLOW if dt > SLOW_S else N_Q
    q2s.append(dt)
    q2_set = hit
    for _ in range(max(nq - 1, 0)):
        dti, _ = _q2_ap(con)
        q2s.append(dti)
        if time.perf_counter() - t_wall0 > WALL_CAP_S:
            stopped = True
            break
    dt, q3_out = _q3_ap(con, n_sellers)
    nq = N_Q_SLOW if dt > SLOW_S else N_Q
    q3s.append(dt)
    for _ in range(max(nq - 1, 0)):
        dti, _ = _q3_ap(con, n_sellers)
        q3s.append(dti)
        if time.perf_counter() - t_wall0 > WALL_CAP_S:
            stopped = True
            break
    size = db.stat().st_size if db.exists() else 0
    con.close()
    try:
        os.remove(db)
        os.rmdir(tmp)
    except OSError:
        pass
    return dict(
        kind="ap", n_sellers=int(n_sellers), n_writes=int(n_done),
        stopped=bool(stopped), wall_s=time.perf_counter() - t_wall0,
        write=_pct(writes), q1=_pct(q1s), q2=_pct(q2s), q3=_pct(q3s),
        bytes=int(size), q2_n=int(len(q2_set)),
        q2_set=q2_set, q1_sets=q1_sets, q3_out=q3_out,
        write_median_amort_s=float(np.median(writes)) if writes else np.nan,
    )


def _flat(rec, extra=None):
    row = dict(
        kind=rec["kind"], n_sellers=rec["n_sellers"], n_writes=rec["n_writes"],
        stopped=rec["stopped"], wall_s=rec["wall_s"], bytes=rec["bytes"],
        write_median_s=rec.get("write_median_amort_s", rec["write"]["median"]),
        write_p95_s=rec["write"]["p95"],
        q1_median_s=rec["q1"]["median"], q1_p95_s=rec["q1"]["p95"],
        q2_median_s=rec["q2"]["median"], q2_p95_s=rec["q2"]["p95"],
        q3_median_s=rec["q3"]["median"], q3_p95_s=rec["q3"]["p95"],
        q2_n=rec["q2_n"],
        q1_n=rec["q1"]["n"], q2_qn=rec["q2"]["n"], q3_qn=rec["q3"]["n"],
    )
    if extra:
        row.update(extra)
    return row


def run_one(n: int, out_dir: Path, tag: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + [int(n)])
    print(f"(a') stream N={n} ...", flush=True)
    _pis, events = make_stream(n, rng)
    for ev in events:
        ev.pop("rows", None)
    print(f"(a') N={n} events={len(events)} writes ...", flush=True)
    rec = _run_ap(n, events)
    cref = _c_answers(events, n) if not rec["stopped"] or rec["n_writes"] == len(events) else None
    extra = {}
    overlap = None
    if cref is not None:
        ja = [
            _jaccard(cref["q1"](lo, lo + SEG - 1), sids)
            for lo, sids in rec["q1_sets"]
        ]
        q3_c = cref["q3"]
        by_c = {lo: (mn, mx) for lo, mn, mx in q3_c}
        mae = []
        for lo, mn, mx in rec["q3_out"]:
            if lo in by_c:
                mae.append(abs(mn - by_c[lo][0]))
                mae.append(abs(mx - by_c[lo][1]))
        q3_mae = float(max(mae)) if mae else np.nan
        extra = dict(
            q1_jaccard=float(np.mean(ja)) if ja else np.nan,
            q2_jaccard=_jaccard(cref["q2"], rec["q2_set"]),
            q3_mae=q3_mae,
        )
        overlap = dict(
            n_sellers=n, pair="ap-c",
            q1_jaccard=extra["q1_jaccard"],
            q2_jaccard=extra["q2_jaccard"],
            q3_mae=q3_mae,
        )
    flat = _flat(rec, extra)
    suffix = f"_{tag}" if tag else ""
    pd.DataFrame([flat]).to_csv(out_dir / f"store_bench_n{n}{suffix}.csv", index=False)
    if overlap:
        pd.DataFrame([overlap]).to_csv(out_dir / f"store_bench_overlap_n{n}{suffix}.csv", index=False)
    dump = {k: v for k, v in rec.items() if k not in ("q2_set", "q1_sets", "q3_out")}
    (out_dir / f"store_bench_n{n}{suffix}.json").write_text(json.dumps(
        dict(row=dump, overlap=overlap, tag=tag,
             timestamp=datetime.now().isoformat(timespec="seconds")),
        indent=2, default=str,
    ) + "\n")
    print(
        f"  writes={rec['n_writes']} wall={rec['wall_s']:.1f}s "
        f"write_med={rec['write']['median']*1e3:.2f}ms "
        f"q1={rec['q1']['median']*1e3:.2f}ms q2={rec['q2']['median']*1e3:.2f}ms "
        f"q3={rec['q3']['median']*1e3:.2f}ms q2_n={rec['q2_n']} "
        f"q1J={extra.get('q1_jaccard', float('nan')):.6f} "
        f"q2J={extra.get('q2_jaccard', float('nan')):.6f} "
        f"q3mae={extra.get('q3_mae', float('nan')):.4g} "
        f"size={rec['bytes']/1e6:.2f}MB stopped={rec['stopped']}",
        flush=True,
    )
    return flat, overlap


def small_concurrency(out_dir: Path, n: int = 1000, n_proc: int = 3):
    """One solo N=1e3, then n_proc copies together. Decide if they slow each other."""
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"small solo N={n} ...", flush=True)
    solo, _ = run_one(n, out_dir, tag="solo")
    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONUNBUFFERED",
    ):
        env.setdefault(k, "1")
    procs = []
    t0 = time.perf_counter()
    for i in range(n_proc):
        cmd = [
            sys.executable, "-m", "shared_evidence.store_bench_query_time",
            "--n", str(n), "--tag", f"p{i}", "--out", str(out_dir),
        ]
        print(f"small parallel start p{i}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    if any(c != 0 for c in codes):
        raise SystemExit(f"small workers failed {codes}")
    parts = []
    for i in range(n_proc):
        p = out_dir / f"store_bench_n{n}_p{i}.csv"
        parts.append(pd.read_csv(p).iloc[0])
    par = pd.DataFrame(parts)
    ratio = float(par.q1_median_s.mean() / max(solo["q1_median_s"], 1e-12))
    write_ratio = float(par.write_median_s.mean() / max(solo["write_median_s"], 1e-12))
    ok = ratio < 1.5 and write_ratio < 1.5
    rec = dict(
        n=n, n_proc=n_proc, wall_s=wall,
        solo_q1_ms=1e3 * solo["q1_median_s"],
        par_q1_ms=1e3 * float(par.q1_median_s.mean()),
        q1_ratio=ratio,
        solo_write_ms=1e3 * solo["write_median_s"],
        par_write_ms=1e3 * float(par.write_median_s.mean()),
        write_ratio=write_ratio,
        concurrency=n_proc if ok else 1,
        note="parallel ok" if ok else "parallel slowed queries; run scales one at a time",
    )
    (out_dir / "small_concurrency.json").write_text(json.dumps(rec, indent=2) + "\n")
    import shutil
    for name in (f"store_bench_n{n}", f"store_bench_overlap_n{n}"):
        for ext in (".csv", ".json"):
            src = out_dir / f"{name}_solo{ext}"
            dst = out_dir / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, dst)
    print("small", json.dumps(rec), flush=True)
    return rec


def combine(out_dir: Path):
    out_dir = Path(out_dir)
    parts = []
    ovs = []
    for n in NS:
        p = out_dir / f"store_bench_n{n}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
        elif (out_dir / f"store_bench_n{n}_solo.csv").exists() and n == 1000:
            parts.append(pd.read_csv(out_dir / f"store_bench_n{n}_solo.csv"))
        o = out_dir / f"store_bench_overlap_n{n}.csv"
        if o.exists():
            ovs.append(pd.read_csv(o))
        elif (out_dir / f"store_bench_overlap_n{n}_solo.csv").exists() and n == 1000:
            ovs.append(pd.read_csv(out_dir / f"store_bench_overlap_n{n}_solo.csv"))
    if not parts:
        print("no (a′) shards yet", flush=True)
        return
    ap = pd.concat(parts, ignore_index=True)
    if (STORE_DIR / "store_bench.csv").exists():
        ab = pd.read_csv(STORE_DIR / "store_bench.csv")
        ab = ab[ab.kind.isin(("a", "b"))].copy()
    else:
        ab = pd.DataFrame()
    if (STORE_DIR / "store_bench.csv").exists():
        c = pd.read_csv(STORE_DIR / "store_bench.csv")
        c = c[c.kind == "c"].copy()
    else:
        c = pd.DataFrame()
    cols = [
        "kind", "n_sellers", "n_writes", "stopped", "wall_s", "bytes",
        "write_median_s", "write_p95_s",
        "q1_median_s", "q1_p95_s", "q2_median_s", "q2_p95_s",
        "q3_median_s", "q3_p95_s", "q2_n",
    ]
    frames = []
    for df in (ab, ap, c):
        if df is None or df.empty:
            continue
        keep = [c for c in cols if c in df.columns]
        frames.append(df[keep])
    out = pd.concat(frames, ignore_index=True)
    kind_ord = {"a": 0, "b": 1, "ap": 2, "c": 3}
    out["_k"] = out.kind.map(kind_ord)
    out = out.sort_values(["n_sellers", "_k"]).drop(columns=["_k"])
    out.to_csv(out_dir / "store_bench.csv", index=False)
    if ovs:
        ov = pd.concat(ovs, ignore_index=True)
        ov.to_csv(out_dir / "store_bench_overlap.csv", index=False)
    print("wrote", out_dir / "store_bench.csv", flush=True)
