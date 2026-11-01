#!/usr/bin/env python3
"""Paper store benches: four stores, refresh interval, scan-all. Single process, one BLAS thread."""

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

from shared_evidence.lemma42 import Z_LB, delta_lemma42
from shared_evidence.store_bench import (
    REFRESH_EVERY, SEED, SEG, HyperAcc, _connect, _init_a, _init_b, _pct,
    _q1_a, _q1_b, _q2_ab, _q3_ab, _write_a, _write_b, make_stream,
)
from shared_evidence.store_bench_query_time import (
    _init_ap, _q1_ap, _q2_ap, _q3_ap, _write_ap,
)
from shared_evidence.store_ops import (
    Q2_TH, _full_answers, _interleave_c_ap, _nq, _q1_sql, _q2_filter_c,
    _q2_sql, _q3_sql, _selectivity, _time_ab, _time_d,
)
from shared_evidence.store_refresh import (
    PRIOR, _init_c, _moments, _refresh_all, _theta, _write_one,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "store"
NAMES = {
    "a": "Row log",
    "b": "Row aggregates",
    "ap": "Record log",
    "c": "QShare",
    "d": "scan-all",
}


def mid_stop(n, R=1000):
    n = int(n)
    s = int(round(0.9975 * n))
    s = min(max(s, 1), n - 1)
    if int(R) > 0 and s % int(R) == 0:
        s -= 1
    return int(s)


def loadavg():
    if hasattr(os, "getloadavg"):
        return tuple(float(x) for x in os.getloadavg())
    return (float("nan"),) * 3


def wait_idle(load_max=1.0, poll_s=600):
    while True:
        la = loadavg()
        print(f"idle-check load={la} max={load_max} {datetime.now().isoformat(timespec='seconds')}", flush=True)
        if la[0] < float(load_max):
            return la
        time.sleep(int(poll_s))


def _db_bytes(con, path: Path) -> int:
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    n = 0
    for p in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if p.exists():
            n += p.stat().st_size
    return int(n)


def _cleanup(con, tmp):
    if con is None or tmp is None:
        return
    try:
        con.close()
    except Exception:
        pass
    try:
        db = Path(tmp) / "store.sqlite"
        if db.exists():
            os.remove(db)
        for extra in (str(db) + "-wal", str(db) + "-shm"):
            if os.path.exists(extra):
                os.remove(extra)
        os.rmdir(tmp)
    except OSError:
        pass


def _amort(writes, refresh_times):
    if not writes:
        return float("nan")
    extra = float(np.sum(refresh_times)) / max(len(writes), 1) if refresh_times else 0.0
    return float(np.mean(writes)) + extra


def run_one(
    n_sellers: int,
    n_mid: int,
    n_end: int,
    R: int,
    out_dir: Path,
    do_ap: bool,
    do_ab: bool,
    do_d: bool,
    tag: str,
):
    n_sellers, n_mid, n_end, R = int(n_sellers), int(n_mid), int(n_end), int(R)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tag
    done = out_dir / f"{stem}.json"
    if done.exists():
        print(f"skip existing {done}", flush=True)
        return json.loads(done.read_text())

    la0 = loadavg()
    print(
        f"{stem} N={n_sellers} mid={n_mid} end={n_end} R={R} "
        f"ap={do_ap} ab={do_ab} d={do_d} load={la0}",
        flush=True,
    )
    rng = np.random.default_rng(SEED + [int(n_sellers)])
    _pis, events = make_stream(n_sellers, rng)
    events = events[: int(n_end)]
    if not do_ab:
        for ev in events:
            ev.pop("rows", None)

    tmp_c = Path(tempfile.mkdtemp(prefix=f"qs_c_{stem}_"))
    db_c = tmp_c / "store.sqlite"
    con_c = _connect(db_c)
    _init_c(con_c, n_sellers)
    con_ap = db_ap = tmp_ap = None
    if do_ap:
        tmp_ap = Path(tempfile.mkdtemp(prefix=f"qs_ap_{stem}_"))
        db_ap = tmp_ap / "store.sqlite"
        con_ap = _connect(db_ap)
        _init_ap(con_ap)
    con_a = con_b = db_a = db_b = tmp_a = tmp_b = None
    if do_ab:
        tmp_a = Path(tempfile.mkdtemp(prefix=f"qs_a_{stem}_"))
        tmp_b = Path(tempfile.mkdtemp(prefix=f"qs_b_{stem}_"))
        db_a, db_b = tmp_a / "store.sqlite", tmp_b / "store.sqlite"
        con_a, con_b = _connect(db_a), _connect(db_b)
        _init_a(con_a)
        _init_b(con_b, n_sellers)

    hyp = HyperAcc()
    snap = PRIOR
    writes_c, writes_ap, writes_a, writes_b, refresh_times = [], [], [], [], []
    delta_rows = []
    moments = set(_moments(n_mid, R)) | {n_mid}
    t0 = time.perf_counter()
    n_done = 0
    theta = PRIOR
    bytes_c = bytes_ap = 0
    cap = rec_d = rec_ab = sel_rows = None
    c1 = np.nan
    chk = None
    q2_at_mid = None

    for ev in events:
        n_done += 1
        if n_done <= n_mid:
            dt_w, theta = _write_one(con_c, hyp, ev["ev"], ev["stats"], ev["touched"], snap)
            writes_c.append(dt_w)
            if do_ap:
                writes_ap.append(_write_ap(con_ap, ev["ev"]))
            if n_done % int(R) == 0:
                refresh_times.append(_refresh_all(con_c, theta))
                snap = theta
            if n_done in moments:
                s2s, m0s, t2s = _theta(con_c, "hyper_snap")
                sig, mu, t2 = theta
                dlt = delta_lemma42(mu, t2, sig, m0s, t2s, s2s, z=Z_LB)
                delta_rows.append(dict(
                    n_sellers=n_sellers, R=int(R), writes=int(n_done),
                    delta=float(dlt),
                    frac=(n_done % int(R)) / float(R) if int(R) else np.nan,
                    refreshed=int(n_done % int(R) == 0),
                ))
        if do_ab:
            writes_a.append(_write_a(con_a, ev["rows"]))
            writes_b.append(_write_b(con_b, ev["rows"]))
        if n_done % 10000 == 0:
            print(f"  {stem} wrote {n_done}/{n_end}", flush=True)
        if n_done == n_mid:
            wall_mid = time.perf_counter() - t0
            last_refresh = int(n_done // int(R) * int(R)) if R else 0
            print(f"  {stem} at mid {n_mid} last_refresh={last_refresh} wall={wall_mid:.1f}s", flush=True)
            chk = _full_answers(con_c, n_sellers)
            if do_ap:
                print("  interleave QShare / Record log", flush=True)
                cap = _interleave_c_ap(con_c, con_ap, n_sellers)
            else:
                print("  time Q2 QShare only", flush=True)
                dt0, hit, nc, dlt = _q2_filter_c(con_c, Q2_TH)
                nq = _nq(dt0)
                dts = []
                for j in range(nq):
                    dt, h, nci, dlti = _q2_filter_c(con_c, Q2_TH)
                    dts.append(dt)
                    if j == 0:
                        hit, nc, dlt = h, nci, dlti
                cap = dict(
                    q1_c=_pct([]), q1_ap=_pct([]), q1_n=0,
                    q2_c=_pct(dts), q2_ap=_pct([]), q2_n=len(dts),
                    q3_c=_pct([]), q3_ap=_pct([]), q3_n=0,
                    q2_hit_c=sorted(hit), q2_hit_ap=[],
                    q2_cand=int(nc), q2_delta=float(dlt),
                    q1_sid_c=None, q1_sid_ap=None, q3_out_c=None,
                )
            q2_at_mid = cap
            if do_d:
                print("  time scan-all + selectivity", flush=True)
                rec_d = _time_d(con_c, n_sellers)
                sel_rows, c1 = _selectivity(con_c, n_sellers)
            bytes_c = _db_bytes(con_c, db_c)
            if do_ap:
                bytes_ap = _db_bytes(con_ap, db_ap)

    wall = time.perf_counter() - t0
    rec_ab = {}
    bytes_a = bytes_b = 0
    if do_ab:
        print("  time Row log", flush=True)
        rec_ab["a"] = _time_ab(con_a, "a", n_sellers)
        bytes_a = _db_bytes(con_a, db_a)
        print("  time Row aggregates", flush=True)
        rec_ab["b"] = _time_ab(con_b, "b", n_sellers)
        bytes_b = _db_bytes(con_b, db_b)

    la1 = loadavg()
    last_refresh = int(n_mid // int(R) * int(R)) if R else 0
    q2_c_set = sorted(cap["q2_hit_c"]) if cap else []
    rec = dict(
        tag=stem, n_sellers=n_sellers, n_mid=n_mid, n_end=n_end, R=int(R),
        n_writes_c=int(len(writes_c)), n_writes_ab=int(len(writes_a)),
        last_refresh=last_refresh,
        just_refreshed=int(n_mid % int(R) == 0) if R else 0,
        loadavg_start=list(la0), loadavg_end=list(la1),
        wall_s=wall,
        write_c=_pct(writes_c), write_ap=_pct(writes_ap),
        write_a=_pct(writes_a), write_b=_pct(writes_b),
        write_c_amort_s=_amort(writes_c, refresh_times),
        refresh=_pct(refresh_times),
        refresh_n=int(len(refresh_times)),
        bytes_c=int(bytes_c), bytes_ap=int(bytes_ap),
        bytes_a=int(bytes_a), bytes_b=int(bytes_b),
        sqlite_pragma="WAL,synchronous=OFF,temp_store=MEMORY",
        names=NAMES,
        protocol="QShare/Record log between refreshes; Row log/Row aggregates at stream end; QShare write is amortised",
        q2_hit_n=len(q2_c_set),
        maxabs_d_m=float(chk["maxabs_d_m"]) if chk else np.nan,
        q2_c_vs_full=bool(q2_c_set == chk["q2_full"]) if chk else None,
        q2_c_vs_acc=bool(q2_c_set == chk["q2_acc"]) if chk else None,
        q2_acc_vs_full=bool(chk["q2_set_equal"]) if chk else None,
        c1pct=float(c1) if c1 == c1 else None,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    if cap:
        rec.update(
            q1_c=cap["q1_c"], q1_ap=cap["q1_ap"],
            q2_c=cap["q2_c"], q2_ap=cap["q2_ap"],
            q3_c=cap["q3_c"], q3_ap=cap["q3_ap"],
            q1_n=cap["q1_n"], q2_n=cap["q2_n"], q3_n=cap["q3_n"],
            q2_cand=cap["q2_cand"], q2_delta=cap["q2_delta"],
            q2_c_vs_ap=bool(q2_c_set == cap.get("q2_hit_ap")),
        )
    if rec_d is not None:
        rec["d_q1"] = rec_d["q1"]
        rec["d_q2"] = rec_d["q2"]
        rec["d_q3"] = rec_d["q3"]
        rec["d_q2_n"] = rec_d["q2_n"]
        rec["q2_c_vs_d"] = bool(q2_c_set == rec_d["q2_hit"])
        rec["d_q2_hit_n"] = len(rec_d["q2_hit"])
    for k in ("a", "b"):
        if k in rec_ab:
            rec[f"{k}_q1"] = rec_ab[k]["q1"]
            rec[f"{k}_q2"] = rec_ab[k]["q2"]
            rec[f"{k}_q3"] = rec_ab[k]["q3"]
            rec[f"{k}_q1_n"] = rec_ab[k]["q1_n"]
            rec[f"{k}_q2_n"] = rec_ab[k]["q2_n"]
            rec[f"{k}_q3_n"] = rec_ab[k]["q3_n"]
            rec[f"{k}_q2_hit_n"] = len(rec_ab[k]["q2_hit"])
    rec["selectivity"] = sel_rows
    rec["delta_points"] = delta_rows
    dump = {k: v for k, v in rec.items() if k not in ("q3_out_c",)}
    (out_dir / f"{stem}.json").write_text(json.dumps(dump, indent=2, default=str) + "\n")
    pd.DataFrame(delta_rows).to_csv(out_dir / f"{stem}_delta.csv", index=False)

    def ms(p):
        if not p or p.get("n", 0) == 0:
            return "NA"
        return f"{1e3*p['median']:.4f} / {1e3*p['p95']:.4f}"

    md = [
        f"# {stem}",
        "",
        f"N={n_sellers} mid={n_mid} end={n_end} R={R} load_start={la0} load_end={la1}",
        f"writes_c={len(writes_c)} last_refresh={last_refresh} wall={wall:.1f}s",
        f"Write QShare amort={1e3*rec['write_c_amort_s']:.4f} ms  "
        f"median={ms(rec['write_c'])}  refresh_n={len(refresh_times)} "
        f"refresh_med={rec['refresh']['median'] if rec['refresh']['n'] else float('nan'):.4f}s",
        f"Size MB QShare/Record log/Row log/Row aggregates = "
        f"{bytes_c/1e6:.3f} / {bytes_ap/1e6:.3f} / {bytes_a/1e6:.3f} / {bytes_b/1e6:.3f}",
    ]
    if cap:
        md += [
            f"Q1 QShare/Record log ms = {ms(cap['q1_c'])} / {ms(cap['q1_ap'])} n={cap['q1_n']}",
            f"Q2 QShare/Record log ms = {ms(cap['q2_c'])} / {ms(cap['q2_ap'])} n={cap['q2_n']} "
            f"|hit|={len(q2_c_set)} |C|={cap['q2_cand']} δ={cap['q2_delta']:.3e}",
            f"Q3 QShare/Record log ms = {ms(cap['q3_c'])} / {ms(cap['q3_ap'])} n={cap['q3_n']}",
        ]
    if rec_d is not None:
        md.append(
            f"Q1/Q2/Q3 scan-all ms = {ms(rec_d['q1'])} / {ms(rec_d['q2'])} / {ms(rec_d['q3'])} "
            f"|hit_d|={len(rec_d['q2_hit'])} equal_c={q2_c_set == rec_d['q2_hit']}"
        )
    for k in ("a", "b"):
        if k in rec_ab:
            md.append(
                f"Q1/Q2/Q3 ({NAMES[k]}) ms = {ms(rec_ab[k]['q1'])} / {ms(rec_ab[k]['q2'])} / "
                f"{ms(rec_ab[k]['q3'])} n={rec_ab[k]['q1_n']}/{rec_ab[k]['q2_n']}/{rec_ab[k]['q3_n']} "
                f"|hit|={len(rec_ab[k]['q2_hit'])}"
            )
    if chk:
        md += [
            f"max|Δm| acc vs hyper_full = {chk['maxabs_d_m']:.3e}",
            f"Q2 QShare vs acc = {q2_c_set == chk['q2_acc']}; QShare vs full = {q2_c_set == chk['q2_full']}; "
            f"acc vs full = {chk['q2_set_equal']}",
        ]
    if sel_rows:
        md.append("selectivity:")
        for r in sel_rows:
            ratio = (r["q2_scan_median_s"] / r["q2_filter_median_s"]) if r["q2_filter_median_s"] else np.nan
            wl = {"c09": "threshold_0.9", "frac1": "top1pct"}
            md.append(
                f"- {wl.get(r['name'], r['name'])} c={r['c']:.4f} ans={r['n_ans_filter']} ({r['answer_frac']:.4f}) "
                f"|C|={r['cand']} δ={r['delta']:.3e}  "
                f"QShare {1e3*r['q2_filter_median_s']:.4f} ms / scan-all {1e3*r['q2_scan_median_s']:.4f} ms  "
                f"scan/filter={ratio:.2f}× equal={r['sets_equal']}"
            )
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("wrote", out_dir / f"{stem}.md", flush=True)

    for con, tmp in ((con_c, tmp_c), (con_ap, tmp_ap), (con_a, tmp_a), (con_b, tmp_b)):
        _cleanup(con, tmp)
    return rec


def _delta95(delta_rows):
    if not delta_rows:
        return float("nan"), 0
    d = np.array([r["delta"] for r in delta_rows], float)
    use = d[(d > 0) & (d < 0.5)]
    if len(use) == 0:
        return float("nan"), 0
    return float(np.quantile(use, 0.95)), int(len(use))


def assemble(out_dir: Path, recs: dict):
    out_dir = Path(out_dir)
    fig_rows = []
    for n in (1000, 10000, 100000):
        key = f"n{n}"
        r = recs.get(key)
        if not r:
            continue
        fig_rows.append(dict(
            n_sellers=n, name=NAMES["c"],
            write_s=r["write_c_amort_s"], write_is_amort=1,
            q2_median_s=r["q2_c"]["median"], q2_p95_s=r["q2_c"]["p95"],
            bytes=r["bytes_c"], n_stop=r["n_mid"], R=r["R"],
        ))
        fig_rows.append(dict(
            n_sellers=n, name=NAMES["ap"],
            write_s=r["write_ap"]["median"], write_is_amort=0,
            q2_median_s=r["q2_ap"]["median"], q2_p95_s=r["q2_ap"]["p95"],
            bytes=r["bytes_ap"], n_stop=r["n_mid"], R=r["R"],
        ))
        fig_rows.append(dict(
            n_sellers=n, name=NAMES["a"],
            write_s=r["write_a"]["median"], write_is_amort=0,
            q2_median_s=r["a_q2"]["median"], q2_p95_s=r["a_q2"]["p95"],
            bytes=r["bytes_a"], n_stop=r["n_end"], R=r["R"],
        ))
        fig_rows.append(dict(
            n_sellers=n, name=NAMES["b"],
            write_s=r["write_b"]["median"], write_is_amort=0,
            q2_median_s=r["b_q2"]["median"], q2_p95_s=r["b_q2"]["p95"],
            bytes=r["bytes_b"], n_stop=r["n_end"], R=r["R"],
        ))
    fig_cols = [
        "n_sellers", "name", "write_s", "write_is_amort",
        "q2_median_s", "q2_p95_s", "bytes", "n_stop", "R",
    ]
    fig = pd.DataFrame(fig_rows, columns=fig_cols)
    fig.to_csv(out_dir / "stores_by_n.csv", index=False)

    t4 = recs.get("n100000")
    t4_rows = []
    if t4:
        t4_rows = [
            dict(name=NAMES["a"], write_s=t4["write_a"]["median"], write_kind="median",
                 q1_s=t4["a_q1"]["median"], q1_p95=t4["a_q1"]["p95"],
                 q2_s=t4["a_q2"]["median"], q2_p95=t4["a_q2"]["p95"],
                 q3_s=t4["a_q3"]["median"], q3_p95=t4["a_q3"]["p95"],
                 bytes=t4["bytes_a"], n_stop=t4["n_end"],
                 nrep_q1=t4["a_q1_n"], nrep_q2=t4["a_q2_n"], nrep_q3=t4["a_q3_n"]),
            dict(name=NAMES["b"], write_s=t4["write_b"]["median"], write_kind="median",
                 q1_s=t4["b_q1"]["median"], q1_p95=t4["b_q1"]["p95"],
                 q2_s=t4["b_q2"]["median"], q2_p95=t4["b_q2"]["p95"],
                 q3_s=t4["b_q3"]["median"], q3_p95=t4["b_q3"]["p95"],
                 bytes=t4["bytes_b"], n_stop=t4["n_end"],
                 nrep_q1=t4["b_q1_n"], nrep_q2=t4["b_q2_n"], nrep_q3=t4["b_q3_n"]),
            dict(name=NAMES["ap"], write_s=t4["write_ap"]["median"], write_kind="median",
                 q1_s=t4["q1_ap"]["median"], q1_p95=t4["q1_ap"]["p95"],
                 q2_s=t4["q2_ap"]["median"], q2_p95=t4["q2_ap"]["p95"],
                 q3_s=t4["q3_ap"]["median"], q3_p95=t4["q3_ap"]["p95"],
                 bytes=t4["bytes_ap"], n_stop=t4["n_mid"],
                 nrep_q1=t4["q1_n"], nrep_q2=t4["q2_n"], nrep_q3=t4["q3_n"]),
            dict(name=NAMES["c"], write_s=t4["write_c_amort_s"], write_kind="amort_mean",
                 q1_s=t4["q1_c"]["median"], q1_p95=t4["q1_c"]["p95"],
                 q2_s=t4["q2_c"]["median"], q2_p95=t4["q2_c"]["p95"],
                 q3_s=t4["q3_c"]["median"], q3_p95=t4["q3_c"]["p95"],
                 bytes=t4["bytes_c"], n_stop=t4["n_mid"],
                 nrep_q1=t4["q1_n"], nrep_q2=t4["q2_n"], nrep_q3=t4["q3_n"]),
        ]
    t4_cols = [
        "name", "write_s", "write_kind", "q1_s", "q1_p95", "q2_s", "q2_p95",
        "q3_s", "q3_p95", "bytes", "n_stop", "nrep_q1", "nrep_q2", "nrep_q3",
    ]
    t4df = pd.DataFrame(t4_rows, columns=t4_cols)
    t4df.to_csv(out_dir / "stores_n1e5.csv", index=False)

    t5_rows = []
    for R, key in ((100, "refresh_R100"), (1000, "n100000"), (10000, "refresh_R10000")):
        r = recs.get(key)
        if not r:
            continue
        d95, n_d = _delta95(r.get("delta_points") or [])
        t5_rows.append(dict(
            R=int(R),
            write_amort_s=r["write_c_amort_s"],
            refresh_med_s=r["refresh"]["median"] if r["refresh"]["n"] else np.nan,
            refresh_p95_s=r["refresh"]["p95"] if r["refresh"]["n"] else np.nan,
            refresh_n=r["refresh_n"],
            q2_median_s=r["q2_c"]["median"],
            q2_p95_s=r["q2_c"]["p95"],
            q2_n=r.get("q2_n", 0),
            delta95=d95, n_delta_pts=n_d,
            n_mid=r["n_mid"], bytes=r["bytes_c"],
        ))
    t5_cols = [
        "R", "write_amort_s", "refresh_med_s", "refresh_p95_s", "refresh_n",
        "q2_median_s", "q2_p95_s", "q2_n", "delta95", "n_delta_pts", "n_mid", "bytes",
    ]
    t5df = pd.DataFrame(t5_rows, columns=t5_cols)
    t5df.to_csv(out_dir / "refresh_interval.csv", index=False)

    wl = {"c09": "threshold_0.9", "frac1": "top1pct"}
    sel = []
    if t4 and t4.get("selectivity"):
        for row in t4["selectivity"]:
            ratio = (row["q2_scan_median_s"] / row["q2_filter_median_s"]) if row["q2_filter_median_s"] else np.nan
            sel.append(dict(
                workload=wl.get(row["name"], row["name"]),
                c=row["c"], n_ans=row["n_ans_filter"],
                answer_frac=row["answer_frac"], cand=row["cand"], delta=row["delta"],
                qshare_s=row["q2_filter_median_s"], scan_s=row["q2_scan_median_s"],
                scan_over_qshare=ratio, sets_equal=row["sets_equal"],
            ))
    sel_cols = [
        "workload", "c", "n_ans", "answer_frac", "cand", "delta",
        "qshare_s", "scan_s", "scan_over_qshare", "sets_equal",
    ]
    pd.DataFrame(sel, columns=sel_cols).to_csv(out_dir / "scan_all.csv", index=False)
    print("wrote", out_dir / "stores_n1e5.csv", out_dir / "refresh_interval.csv",
          out_dir / "stores_by_n.csv", out_dir / "scan_all.csv", flush=True)
    return fig


def plot_fig4(fig: pd.DataFrame, out_dir: Path):
    if fig is None or len(fig) == 0:
        p = Path(out_dir) / "stores_by_n.csv"
        fig = pd.read_csv(p) if p.exists() else None
    if fig is None or len(fig) == 0:
        print("no fig data", flush=True)
        return
    from paper.figures import _use_libertine, fig4
    _use_libertine()
    fig4()
    print("wrote fig4_*.pdf", flush=True)


def load_existing(out_dir: Path) -> dict:
    out = {}
    mapping = {
        "n1000": "n1000.json",
        "n10000": "n10000.json",
        "n100000": "n100000.json",
        "refresh_R100": "refresh_R100.json",
        "refresh_R10000": "refresh_R10000.json",
    }
    for k, fn in mapping.items():
        p = out_dir / fn
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def main():
    ap = argparse.ArgumentParser(description="Four-store bench. Paper numbers need an idle machine.")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, nargs="+", default=None,
                    help="seller counts to run (default: 1000 10000 100000 plus R grid at 1e5)")
    ap.add_argument("--load-max", type=float, default=1.0)
    ap.add_argument("--poll-s", type=int, default=600)
    ap.add_argument("--idle", action="store_true",
                    help="wait until load average < --load-max before each job")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(k, "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        p = out / "stores_by_n.csv"
        plot_fig4(pd.read_csv(p) if p.exists() else None, out)
        return
    do_wait = bool(args.idle) and not args.skip_wait
    if do_wait:
        wait_idle(args.load_max, args.poll_s)
    recs = load_existing(out)
    ns = [int(x) for x in args.n] if args.n else [1000, 10000, 100000]
    jobs = []
    for n in ns:
        jobs.append((n, mid_stop(n), n, 1000, True, True, n >= 100000, f"n{n}"))
    if 100000 in ns:
        jobs.append((100000, 99750, 99750, 100, False, False, False, "refresh_R100"))
        jobs.append((100000, 99750, 99750, 10000, False, False, False, "refresh_R10000"))
    for n, mid, end, R, ap_, ab, d, tag in jobs:
        if tag in recs:
            print(f"have {tag}", flush=True)
            continue
        if do_wait:
            wait_idle(args.load_max, args.poll_s)
        recs[tag] = run_one(n, mid, end, R, out, do_ap=ap_, do_ab=ab, do_d=d, tag=tag)
    fig = assemble(out, recs)
    if args.n is None or 100000 in ns:
        plot_fig4(fig, out)
    print("store done", datetime.now().isoformat(timespec="seconds"), "load", loadavg(), flush=True)


if __name__ == "__main__":
    main()
