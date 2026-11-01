#!/usr/bin/env python3
"""Row-log and row-aggregate helpers, stream generator, HyperAcc."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import betainc

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "store"
N_EV_BUYERS = 20
N_ROWS = 40
SEG = 100
N_WRITE_SELLERS = 20
TOP_K = 10
N_Q = 200
N_Q_SLOW = 20
SLOW_S = 1.0
WALL_CAP_S = 3 * 3600.0
REFRESH_EVERY = 1000
NS = (1_000, 10_000, 100_000)
PI_LEVELS = (0.5, 0.7, 0.9, 1.0)
SEED = [20261019, 4, 1008]
Q2_TH = 0.9
TAU0_FLOOR = 0.0025
Z_LB = 0.8416


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def _pct(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return dict(n=0, median=np.nan, p95=np.nan)
    return dict(n=int(len(xs)), median=float(np.median(xs)), p95=float(np.quantile(xs, 0.95)))


class HyperAcc:
    """Incremental σ̂² / EB sufficient stats. Matches BuyerEvidenceStore._sigma2_tau."""

    def __init__(self):
        self.sse = 0.0
        self.df = 0.0
        self.n_ev = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_invn = 0.0
        self.n = {}
        self.s1 = {}
        self.s2 = {}

    def _sub(self, s):
        n = self.n.get(s, 0)
        if n <= 0:
            return
        x = self.s1[s] / n
        self.n_ev -= 1
        self.sum_x -= x
        self.sum_x2 -= x * x
        self.sum_invn -= 1.0 / n
        if n >= 2:
            self.sse -= self.s2[s] - self.s1[s] ** 2 / n
            self.df -= n - 1

    def _add(self, s):
        n = self.n.get(s, 0)
        if n <= 0:
            return
        x = self.s1[s] / n
        self.n_ev += 1
        self.sum_x += x
        self.sum_x2 += x * x
        self.sum_invn += 1.0 / n
        if n >= 2:
            self.sse += self.s2[s] - self.s1[s] ** 2 / n
            self.df += n - 1

    def update(self, s, pi_hat):
        s = int(s)
        self._sub(s)
        self.n[s] = self.n.get(s, 0) + 1
        self.s1[s] = self.s1.get(s, 0.0) + float(pi_hat)
        self.s2[s] = self.s2.get(s, 0.0) + float(pi_hat) ** 2
        self._add(s)

    def hyper(self):
        if self.df <= 0:
            sigma2 = 1.0 / 18.0
        else:
            sigma2 = max(self.sse / self.df, 1e-8)
        if self.n_ev < 2:
            return sigma2, 2.0 / 3.0, 1.0 / 18.0
        mu0 = self.sum_x / self.n_ev
        var_m = (self.sum_x2 - self.sum_x ** 2 / self.n_ev) / max(self.n_ev - 1, 1)
        mean_se = sigma2 * self.sum_invn / self.n_ev
        tau0_2 = max(var_m - mean_se, TAU0_FLOOR)
        return float(sigma2), float(mu0), float(tau0_2)


def hyper_full(n, s1, s2):
    js = [s for s, nj in n.items() if nj >= 1]
    js2 = [s for s in js if n[s] >= 2]
    if not js2:
        sigma2 = 1.0 / 18.0
    else:
        sse = sum(s2[s] - s1[s] ** 2 / n[s] for s in js2)
        df = sum(n[s] - 1 for s in js2)
        sigma2 = max(sse / max(df, 1), 1e-8)
    if len(js) < 2:
        return sigma2, 2.0 / 3.0, 1.0 / 18.0
    means = [s1[s] / n[s] for s in js]
    mu0 = float(sum(means) / len(means))
    var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
    mean_se = float(sum(sigma2 / n[s] for s in js) / len(js))
    return float(sigma2), float(mu0), float(max(var_m - mean_se, TAU0_FLOOR))


def _init_a(con):
    con.execute("CREATE TABLE row_evidence (seller INTEGER, row_id INTEGER, buyer INTEGER, gamma REAL)")
    con.execute("CREATE INDEX ix_a_sr ON row_evidence(seller, row_id)")
    con.commit()


def _init_b(con, n_sellers: int):
    _init_a(con)
    con.execute("CREATE TABLE row_stats (seller INTEGER, row_id INTEGER, n INTEGER, sum_gamma REAL, PRIMARY KEY (seller, row_id))")
    con.execute("CREATE TABLE seller_beta (seller INTEGER PRIMARY KEY, a REAL, b REAL)")
    con.executemany("INSERT INTO seller_beta VALUES (?,?,?)", [(i, 2.0, 1.0) for i in range(n_sellers)])
    con.commit()


def _init_c(con, n_sellers: int):
    con.execute("CREATE TABLE buyer_evidence (seller INTEGER, buyer INTEGER, t INTEGER, pi_hat REAL, n_rows INTEGER)")
    con.execute("CREATE TABLE seller_stats (seller INTEGER PRIMARY KEY, n INTEGER, s1 REAL, s2 REAL, m REAL, v REAL, lb REAL)")
    con.execute("CREATE TABLE hyper (id INTEGER PRIMARY KEY, sigma2 REAL, mu0 REAL, tau0_2 REAL)")
    con.executemany(
        "INSERT INTO seller_stats VALUES (?,?,?,?,?,?,?)",
        [(i, 0, 0.0, 0.0, 2.0 / 3.0, (1.0 / 18.0) ** 0.5, 2.0 / 3.0 - Z_LB * (1.0 / 18.0) ** 0.5)
         for i in range(n_sellers)],
    )
    con.execute("INSERT INTO hyper VALUES (1, ?, ?, ?)", (1.0 / 18.0, 2.0 / 3.0, 1.0 / 18.0))
    con.execute("CREATE INDEX ix_c_m ON seller_stats(m)")
    con.execute("CREATE INDEX ix_c_lb ON seller_stats(lb)")
    con.commit()


def _gamma_rows(rng, pi, n=N_ROWS):
    clean = rng.random(n) < pi
    g = np.empty(n, float)
    n_c = int(clean.sum())
    if n_c:
        g[clean] = rng.beta(19.0, 1.0, size=n_c)
    if n_c < n:
        g[~clean] = rng.beta(1.0, 4.0, size=n - n_c)
    return g


def make_stream(n_sellers: int, rng):
    """One shared evidence stream: list of (rows, ev, stats, touched)."""
    pis = rng.choice(np.asarray(PI_LEVELS), size=n_sellers)
    seen = {s: [] for s in range(n_sellers)}
    events = []
    buyer_id = 0
    n_chunks = n_sellers // N_WRITE_SELLERS
    for wave in range(N_EV_BUYERS):
        for ch in range(n_chunks):
            sids = list(range(ch * N_WRITE_SELLERS, (ch + 1) * N_WRITE_SELLERS))
            buyer_id += 1
            rows, ev, stats, touched = [], [], [], []
            for s in sids:
                n_new = N_ROWS // 2 if seen[s] else N_ROWS
                n_old = N_ROWS - n_new
                g_new = _gamma_rows(rng, pis[s], n_new)
                start = len(seen[s])
                for i, gi in enumerate(g_new):
                    rid = start + i
                    seen[s].append(rid)
                    rows.append((s, rid, buyer_id, float(gi)))
                if n_old and seen[s]:
                    old = rng.choice(np.asarray(seen[s][:start] if start else seen[s]), size=n_old, replace=True)
                    g_old = _gamma_rows(rng, pis[s], n_old)
                    for rid, gi in zip(old, g_old):
                        rows.append((s, int(rid), buyer_id, float(gi)))
                ph = float(np.mean([r[3] for r in rows if r[0] == s]))
                ev.append((s, buyer_id, wave + 1, ph, N_ROWS))
                stats.append((ph, ph * ph, s))
                touched.append(s)
            events.append(dict(rows=rows, ev=ev, stats=stats, touched=touched))
    return pis, events


def _write_a(con, rows):
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO row_evidence VALUES (?,?,?,?)", rows)
    con.commit()
    return time.perf_counter() - t0


def _write_b(con, rows):
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO row_evidence VALUES (?,?,?,?)", rows)
    by = {}
    for s, rid, _b, g in rows:
        by.setdefault((int(s), int(rid)), []).append(float(g))
    for (s, rid), gs in by.items():
        cur = con.execute("SELECT n, sum_gamma FROM row_stats WHERE seller=? AND row_id=?", (s, rid)).fetchone()
        beta = con.execute("SELECT a, b FROM seller_beta WHERE seller=?", (s,)).fetchone()
        a, b = (float(beta[0]), float(beta[1])) if beta else (2.0, 1.0)
        if cur:
            n0, s0 = int(cur[0]), float(cur[1])
            g0 = s0 / n0
            a -= g0
            b -= (1.0 - g0)
            n1, s1 = n0 + len(gs), s0 + sum(gs)
        else:
            n1, s1 = len(gs), sum(gs)
        g1 = s1 / n1
        a += g1
        b += (1.0 - g1)
        con.execute(
            "INSERT INTO row_stats(seller, row_id, n, sum_gamma) VALUES (?,?,?,?) "
            "ON CONFLICT(seller, row_id) DO UPDATE SET n=?, sum_gamma=?",
            (s, rid, n1, s1, n1, s1),
        )
        con.execute("UPDATE seller_beta SET a=?, b=? WHERE seller=?", (a, b, s))
    con.commit()
    return time.perf_counter() - t0


def _publish_c(n, s1, sigma2, mu0, tau0_2):
    if n <= 0:
        m, v = mu0, tau0_2 ** 0.5
    else:
        prec = 1.0 / tau0_2 + n / sigma2
        m = (mu0 / tau0_2 + s1 / sigma2) / prec
        v = prec ** -0.5
    return float(m), float(v), float(m - Z_LB * v)


def _refresh_all_c(con, sigma2, mu0, tau0_2):
    t0 = time.perf_counter()
    rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
    upd = []
    for s, n, s1 in rows:
        m, v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        upd.append((m, v, lb, int(s)))
    con.executemany("UPDATE seller_stats SET m=?, v=?, lb=? WHERE seller=?", upd)
    return time.perf_counter() - t0


def _write_c(con, hyp: HyperAcc, ev_rows, stats_upd, touched, n_writes, refresh_times):
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO buyer_evidence VALUES (?,?,?,?,?)", ev_rows)
    con.executemany(
        "UPDATE seller_stats SET n=n+1, s1=s1+?, s2=s2+? WHERE seller=?",
        stats_upd,
    )
    for s, ph, *_ in ((s, ph) for ph, _p2, s in stats_upd):
        hyp.update(int(s), float(ph))
    sigma2, mu0, tau0_2 = hyp.hyper()
    con.execute("UPDATE hyper SET sigma2=?, mu0=?, tau0_2=? WHERE id=1", (sigma2, mu0, tau0_2))
    for s in touched:
        n, s1, _s2 = con.execute("SELECT n, s1, s2 FROM seller_stats WHERE seller=?", (s,)).fetchone()
        m, v, lb = _publish_c(int(n), float(s1), sigma2, mu0, tau0_2)
        con.execute("UPDATE seller_stats SET m=?, v=?, lb=? WHERE seller=?", (m, v, lb, s))
    extra = 0.0
    if n_writes % REFRESH_EVERY == 0:
        extra = _refresh_all_c(con, sigma2, mu0, tau0_2)
        refresh_times.append(extra)
    con.commit()
    return time.perf_counter() - t0


def _q1_a(con, lo, hi):
    q = (
        "WITH rowm AS ("
        " SELECT seller, row_id, AVG(gamma) g FROM row_evidence "
        " WHERE seller BETWEEN ? AND ? GROUP BY seller, row_id)"
        " SELECT seller, (2.0+SUM(g))/(3.0+COUNT(*)) m FROM rowm "
        " GROUP BY seller ORDER BY m DESC LIMIT ?"
    )
    t0 = time.perf_counter()
    rows = con.execute(q, (lo, hi, TOP_K)).fetchall()
    return time.perf_counter() - t0, [int(r[0]) for r in rows]


def _q1_b(con, lo, hi):
    q = "SELECT seller FROM seller_beta WHERE seller BETWEEN ? AND ? ORDER BY a/(a+b) DESC LIMIT ?"
    t0 = time.perf_counter()
    rows = con.execute(q, (lo, hi, TOP_K)).fetchall()
    return time.perf_counter() - t0, [int(r[0]) for r in rows]


def _q1_c(con, lo, hi):
    q = "SELECT seller FROM seller_stats WHERE seller BETWEEN ? AND ? ORDER BY m DESC LIMIT ?"
    t0 = time.perf_counter()
    rows = con.execute(q, (lo, hi, TOP_K)).fetchall()
    return time.perf_counter() - t0, [int(r[0]) for r in rows]


def _q2_ab(con, kind):
    t0 = time.perf_counter()
    if kind == "a":
        rows = con.execute(
            "WITH rowm AS (SELECT seller, row_id, AVG(gamma) g FROM row_evidence GROUP BY seller, row_id) "
            "SELECT seller, 2.0+SUM(g), 1.0+COUNT(*)-SUM(g) FROM rowm GROUP BY seller"
        ).fetchall()
    else:
        rows = con.execute("SELECT seller, a, b FROM seller_beta").fetchall()
    hit = [int(s) for s, a, b in rows if 1.0 - float(betainc(a, b, Q2_TH)) > 0.8]
    return time.perf_counter() - t0, hit


def _q2_c(con, fresh=False):
    t0 = time.perf_counter()
    if fresh:
        sigma2, mu0, tau0_2 = con.execute("SELECT sigma2, mu0, tau0_2 FROM hyper WHERE id=1").fetchone()
        rows = con.execute("SELECT seller, n, s1 FROM seller_stats").fetchall()
        hit = []
        for s, n, s1 in rows:
            m, v, lb = _publish_c(int(n), float(s1), float(sigma2), float(mu0), float(tau0_2))
            if lb > Q2_TH:
                hit.append(int(s))
    else:
        hit = [int(r[0]) for r in con.execute("SELECT seller FROM seller_stats WHERE lb > ?", (Q2_TH,)).fetchall()]
    return time.perf_counter() - t0, hit


def _q3_ab(con, kind, n_sellers):
    t0 = time.perf_counter()
    if kind == "a":
        rows = con.execute(
            "WITH rowm AS (SELECT seller, row_id, AVG(gamma) g FROM row_evidence GROUP BY seller, row_id) "
            "SELECT seller, (2.0+SUM(g))/(3.0+COUNT(*)) FROM rowm GROUP BY seller"
        ).fetchall()
    else:
        rows = con.execute("SELECT seller, a/(a+b) FROM seller_beta").fetchall()
    by = {int(s): float(m) for s, m in rows}
    out = []
    for lo in range(0, n_sellers, SEG):
        vals = [by[s] for s in range(lo, min(lo + SEG, n_sellers)) if s in by]
        if vals:
            out.append((lo, float(np.mean(vals)), float(np.max(vals))))
    return time.perf_counter() - t0, out


def _q3_c(con, n_sellers):
    t0 = time.perf_counter()
    rows = con.execute("SELECT seller, m FROM seller_stats").fetchall()
    by = {int(s): float(m) for s, m in rows}
    out = []
    for lo in range(0, n_sellers, SEG):
        vals = [by[s] for s in range(lo, min(lo + SEG, n_sellers)) if s in by]
        if vals:
            out.append((lo, float(np.mean(vals)), float(np.max(vals))))
    return time.perf_counter() - t0, out


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _nq(sample_dt):
    return N_Q_SLOW if sample_dt > SLOW_S else N_Q


def _run_kind(kind, n_sellers, events):
    tmp = Path(tempfile.mkdtemp(prefix=f"store_{kind}_{n_sellers}_"))
    db = tmp / "store.sqlite"
    con = _connect(db)
    if kind == "a":
        _init_a(con)
    elif kind == "b":
        _init_b(con, n_sellers)
    else:
        _init_c(con, n_sellers)
    hyp = HyperAcc()
    writes, refresh_times = [], []
    t_wall0 = time.perf_counter()
    stopped = False
    n_done = 0
    for ev in events:
        n_done += 1
        if kind == "a":
            writes.append(_write_a(con, ev["rows"]))
        elif kind == "b":
            writes.append(_write_b(con, ev["rows"]))
        else:
            writes.append(_write_c(con, hyp, ev["ev"], ev["stats"], ev["touched"], n_done, refresh_times))
        if time.perf_counter() - t_wall0 > WALL_CAP_S:
            stopped = True
            break
    n_seg = max(n_sellers // SEG, 1)
    q1s, q2s, q3s, q2_lag = [], [], [], []
    q1_sets, q2_set = [], []
    for i in range(min(5, n_seg)):
        lo = i * SEG
        hi = lo + SEG - 1
        if kind == "a":
            dt, sids = _q1_a(con, lo, hi)
        elif kind == "b":
            dt, sids = _q1_b(con, lo, hi)
        else:
            dt, sids = _q1_c(con, lo, hi)
        nq = _nq(dt)
        if i == 0:
            for j in range(nq):
                loj = ((j) % n_seg) * SEG
                hij = loj + SEG - 1
                if kind == "a":
                    dti, sidj = _q1_a(con, loj, hij)
                elif kind == "b":
                    dti, sidj = _q1_b(con, loj, hij)
                else:
                    dti, sidj = _q1_c(con, loj, hij)
                q1s.append(dti)
                if j < 20:
                    q1_sets.append((loj, sidj))
            break
    if kind == "c":
        dt, hit = _q2_c(con, fresh=False)
        _, hit_f = _q2_c(con, fresh=True)
        q2_lag.append(_jaccard(hit, hit_f))
    else:
        dt, hit = _q2_ab(con, kind)
    nq = _nq(dt)
    q2s.append(dt)
    q2_set = hit
    for _ in range(nq - 1):
        if kind == "c":
            dti, h0 = _q2_c(con, fresh=False)
            _, hf = _q2_c(con, fresh=True)
            q2_lag.append(_jaccard(h0, hf))
        else:
            dti, _ = _q2_ab(con, kind)
        q2s.append(dti)
    if kind == "c":
        dt, _ = _q3_c(con, n_sellers)
    else:
        dt, _ = _q3_ab(con, kind, n_sellers)
    nq = _nq(dt)
    q3s.append(dt)
    for _ in range(nq - 1):
        if kind == "c":
            dti, _ = _q3_c(con, n_sellers)
        else:
            dti, _ = _q3_ab(con, kind, n_sellers)
        q3s.append(dti)
    size = db.stat().st_size if db.exists() else 0
    con.close()
    try:
        os.remove(db)
        os.rmdir(tmp)
    except OSError:
        pass
    write_med = float(np.median(writes)) if writes else np.nan
    if refresh_times:
        amort = float(np.sum(refresh_times)) / max(n_done, 1)
        write_med = write_med + amort
    return dict(
        kind=kind, n_sellers=int(n_sellers), n_writes=int(n_done),
        stopped=bool(stopped), wall_s=time.perf_counter() - t_wall0,
        write=_pct(writes), q1=_pct(q1s), q2=_pct(q2s), q3=_pct(q3s),
        bytes=int(size), q2_n=int(len(q2_set)),
        q2_set=q2_set, q1_sets=q1_sets,
        refresh_n=int(len(refresh_times)),
        refresh_med=float(np.median(refresh_times)) if refresh_times else 0.0,
        q2_lag=float(np.mean(q2_lag)) if q2_lag else np.nan,
        write_median_amort_s=write_med,
    )


def run_bench(out_dir: Path = OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, overlap = [], []
    for n in NS:
        rng = np.random.default_rng(SEED + [int(n)])
        print(f"store_bench stream N={n} ...", flush=True)
        _pis, events = make_stream(n, rng)
        recs = {}
        for kind in ("a", "b", "c"):
            print(f"store_bench {kind} N={n} events={len(events)} ...", flush=True)
            rec = _run_kind(kind, n, events)
            recs[kind] = rec
            rows.append(rec)
            print(
                f"  writes={rec['n_writes']} wall={rec['wall_s']:.1f}s "
                f"write_med={rec['write']['median']*1e3:.2f}ms "
                f"q1={rec['q1']['median']*1e3:.2f}ms q2_n={rec['q2_n']} "
                f"size={rec['bytes']/1e6:.1f}MB stopped={rec['stopped']}",
                flush=True,
            )
        if "a" in recs and "b" in recs:
            ja = [_jaccard(sa, sb) for (lo_a, sa), (lo_b, sb) in zip(recs["a"]["q1_sets"], recs["b"]["q1_sets"]) if lo_a == lo_b]
            overlap.append(dict(
                n_sellers=n, pair="a-b",
                q1_jaccard=float(np.mean(ja)) if ja else np.nan,
                q2_jaccard=_jaccard(recs["a"]["q2_set"], recs["b"]["q2_set"]),
            ))
        if "a" in recs and "c" in recs:
            ja = [_jaccard(sa, sc) for (lo_a, sa), (lo_c, sc) in zip(recs["a"]["q1_sets"], recs["c"]["q1_sets"]) if lo_a == lo_c]
            overlap.append(dict(
                n_sellers=n, pair="a-c",
                q1_jaccard=float(np.mean(ja)) if ja else np.nan,
                q2_jaccard=_jaccard(recs["a"]["q2_set"], recs["c"]["q2_set"]),
                q2_lag_c=recs["c"].get("q2_lag"),
            ))
        if any(recs[k]["stopped"] for k in recs):
            print(f"store_bench stop after N={n}", flush=True)
            break
    import pandas as pd
    flat = []
    for r in rows:
        flat.append(dict(
            kind=r["kind"], n_sellers=r["n_sellers"], n_writes=r["n_writes"],
            stopped=r["stopped"], wall_s=r["wall_s"], bytes=r["bytes"],
            write_median_s=r.get("write_median_amort_s", r["write"]["median"]),
            write_p95_s=r["write"]["p95"],
            q1_median_s=r["q1"]["median"], q1_p95_s=r["q1"]["p95"],
            q2_median_s=r["q2"]["median"], q2_p95_s=r["q2"]["p95"],
            q3_median_s=r["q3"]["median"], q3_p95_s=r["q3"]["p95"],
            q2_n=r["q2_n"], refresh_n=r.get("refresh_n", 0),
            refresh_med_s=r.get("refresh_med", 0.0),
            q2_lag=r.get("q2_lag"),
        ))
    pd.DataFrame(flat).to_csv(out_dir / "store_bench.csv", index=False)
    if overlap:
        pd.DataFrame(overlap).to_csv(out_dir / "store_bench_overlap.csv", index=False)
    dump = [{k: v for k, v in r.items() if k not in ("q2_set", "q1_sets")} for r in rows]
    (out_dir / "store_bench.json").write_text(json.dumps(
        dict(rows=dump, overlap=overlap, timestamp=datetime.now().isoformat(timespec="seconds")),
        indent=2, default=str,
    ) + "\n")
    print("wrote", out_dir / "store_bench.csv", flush=True)
    return rows
