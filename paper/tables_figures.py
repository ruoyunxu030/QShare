#!/usr/bin/env python3
"""Rebuild paper Tables 2 to 7 and Figures 3 to 5 from results/."""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MKT = ROOT / "results" / "market"
STO = ROOT / "results" / "store"
OUT = ROOT / "paper" / "out"

TAB2_POL = [
    "indep", "share_naive", "share_agg", "share_agg_ts",
    "share_agg_ucb", "fund80_agg", "floor80_agg",
]
TAB2_LABEL = {
    "indep": "Independent",
    "share_naive": "Row-level sharing",
    "share_agg": "QShare without floor",
    "share_agg_ts": "QShare without floor, with TS",
    "share_agg_ucb": "QShare without floor, with UCB",
    "fund80_agg": "Entry subsidy",
    "floor80_agg": "QShare",
}
TAB2_MKTS = [
    ("amazon", "base"),
    ("amazon", "entry"),
    ("domainnet", "base"),
    ("domainnet", "entry"),
]
FLOOR_ROWS = (
    ("share_agg", 0, "QShare"),
    ("floor40_agg", 40, "QShare"),
    ("floor80_agg", 80, "QShare"),
    ("floor40_row", 40, "Row-level"),
    ("floor80_row", 80, "Row-level"),
)
STORE_ORDER = ("Row log", "Row aggregates", "Record log", "QShare")


def rnd(x, nd: int = 1, times: int | float = 1):
    """Round-half-up. `times` is applied in decimal space (use 100 for rates)."""
    if x is None or x == "":
        return ""
    try:
        if isinstance(x, Decimal):
            d = x
        else:
            v = float(x)
            if not np.isfinite(v):
                return ""
            d = Decimal(str(v))
    except (TypeError, ValueError):
        return ""
    d = d * Decimal(str(times))
    q = Decimal("1").scaleb(-int(nd))
    d = d.quantize(q, rounding=ROUND_HALF_UP)
    if nd <= 0:
        return int(d)
    return format(d, "f")


def _md(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")
    df.to_csv(path.with_suffix(".csv"), index=False)


def _spend_pct(buyers: pd.DataFrame, ds: str, market: str, pol: str) -> float:
    g = buyers[(buyers.dataset == ds) & (buyers.market == market) & (buyers.policy == pol)]
    if g.empty:
        return float("nan")
    ev = g["small_purchase_spend"].fillna(0) + g["platform_paid"].fillna(0)
    pkt = g["packet_spend"].fillna(0)
    tot = ev + pkt
    rec = []
    for _, gi in g.assign(ev=ev, tot=tot).groupby("instance"):
        t = float(gi["tot"].sum())
        rec.append(100.0 * float(gi["ev"].sum()) / t if t > 0 else np.nan)
    return float(np.nanmean(rec)) if rec else float("nan")


def tab2():
    fr = pd.read_csv(MKT / "frontier.csv")
    buyers = pd.read_csv(MKT / "per_buyer.csv")
    rows = []
    for pol in TAB2_POL:
        rec = dict(method=TAB2_LABEL[pol], policy=pol)
        for ds, market in TAB2_MKTS:
            sub = fr[(fr.dataset == ds) & (fr.market == market) & (fr.policy == pol)]
            key = f"{ds}_{market}"
            if sub.empty:
                rec[f"{key}_loss"] = ""
                rec[f"{key}_lock"] = ""
                rec[f"{key}_fail"] = ""
                rec[f"{key}_spend"] = ""
                continue
            r = sub.iloc[0]
            rec[f"{key}_loss"] = rnd(r["red"], 1)
            rec[f"{key}_lock"] = rnd(r["lock"], 1, times=100)
            fail = r["unsold"]
            rec[f"{key}_fail"] = "" if pd.isna(fail) else rnd(fail, 0, times=100)
            rec[f"{key}_spend"] = rnd(_spend_pct(buyers, ds, market, pol), 1)
        rows.append(rec)
    df = pd.DataFrame(rows)
    _md(df, OUT / "tab2_main.md")
    return "tab2_main Table 2 (7 methods x 4 markets Loss/Lock/Fail/Spend)"


def tab3():
    src = pd.read_csv(MKT / "seller_ranking.csv")
    rows = []
    for _, r in src.iterrows():
        seen = float(r["seen"])
        rows.append(dict(
            dataset=r["dataset"],
            Seen=rnd(seen, 1 if seen < 1 else 0, times=100 if seen <= 1 else 1),
            auc_seen_indep=rnd(r["auc_seen_indep"], 3),
            auc_seen_qshare=rnd(r["auc_seen_qshare"], 3),
            auc_all_indep=rnd(r["auc_all_indep"], 3),
            auc_all_qshare=rnd(r["auc_all_qshare"], 3),
            top2_indep=rnd(r["top2_indep"], 3),
            top2_qshare=rnd(r["top2_qshare"], 3),
        ))
    _md(pd.DataFrame(rows), OUT / "tab3_alone.md")
    return "tab3_alone Table 3 (Seen / AUC seen / AUC all / Top-2 at t=30)"


def _align_inst(a, b):
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    common = pd.Index(sorted(a.index.intersection(b.index)))
    return a.loc[common], b.loc[common]


def _diff_boot(a, b, n_boot=5000, seed=0):
    a, b = _align_inst(a, b)
    d = (a - b).to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _ratio_pct_boot(q, i, n_boot=5000, seed=0):
    q, i = _align_inst(q, i)
    q, i = q.to_numpy(float), i.to_numpy(float)
    if len(q) == 0:
        return np.nan, np.nan, np.nan
    pt = 100.0 * (1.0 - float(q.mean()) / float(i.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(q), size=(n_boot, len(q)))
    m = 100.0 * (1.0 - q[idx].mean(1) / i[idx].mean(1))
    return pt, float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _ci_cell(pt, lo, hi, nd: int, times: float = 1):
    if not np.isfinite(pt):
        return ""
    return f"{rnd(times * pt, nd)} [{rnd(times * lo, nd)}, {rnd(times * hi, nd)}]"


def _inst_excess(buyers, ds, market, pol):
    g = buyers[
        (buyers.dataset == ds) & (buyers.market == market)
        & (buyers.policy == pol) & (buyers.buyer_t >= 16) & (buyers.buyer_t <= 30)
    ]
    return g.groupby("instance").excess.mean().sort_index()


def _inst_spend_frac(buyers, ds, market, pol):
    g = buyers[(buyers.dataset == ds) & (buyers.market == market) & (buyers.policy == pol)]
    rec = {}
    for inst, gi in g.groupby("instance"):
        ev = float((gi["small_purchase_spend"].fillna(0) + gi["platform_paid"].fillna(0)).sum())
        pkt = float(gi["packet_spend"].fillna(0).sum())
        tot = ev + pkt
        rec[int(inst)] = ev / tot if tot > 0 else np.nan
    return pd.Series(rec).sort_index()


def tab4():
    df = pd.read_csv(STO / "stores_n1e5.csv")
    rows = []
    for name in STORE_ORDER:
        sub = df[df.name == name]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(dict(
            Store=name,
            Write=_fmt_ms(1e3 * float(r["write_s"])),
            Q1=_fmt_ms(1e3 * float(r["q1_s"])),
            Q2=_fmt_ms(1e3 * float(r["q2_s"])),
            Q3=_fmt_ms(1e3 * float(r["q3_s"])),
            Size=_fmt_int_comma(float(r["bytes"]) / 1e6),
        ))
    _md(pd.DataFrame(rows), OUT / "tab4_store.md")
    return "tab4_store Table 4 (four stores at N=1e5)"


def _fmt_ms(ms):
    v = float(ms)
    if v >= 100:
        return f"{int(rnd(v, 0)):,}"
    if v >= 10:
        return rnd(v, 1)
    return rnd(v, 2)


def _fmt_int_comma(x):
    return f"{int(rnd(x, 0)):,}"


def _fmt_sci(x, sig=2):
    s = f"{float(x):.{sig - 1}e}"
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


def tab5():
    df = pd.read_csv(STO / "refresh_interval.csv")
    rows = []
    for R in (100, 1000, 10000):
        sub = df[df.R == R]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(dict(
            R=_fmt_int_comma(R),
            Write=_fmt_ms(1e3 * float(r["write_amort_s"])),
            Q2=_fmt_ms(1e3 * float(r["q2_median_s"])),
            delta95=_fmt_sci(r["delta95"], 2),
        ))
    _md(pd.DataFrame(rows), OUT / "tab5_refresh.md")
    return "tab5_refresh Table 5 (R grid, N=1e5)"


def tab6():
    buyers = pd.read_csv(MKT / "per_buyer.csv")
    lockin = pd.read_csv(MKT / "lockin_pi1.csv")
    rows = []
    for ds, market in TAB2_MKTS:
        rec = dict(market=f"{ds} {market}")
        q = _inst_excess(buyers, ds, market, "floor80_agg")
        i = _inst_excess(buyers, ds, market, "indep")
        rec["Loss"] = _ci_cell(*_ratio_pct_boot(q, i), nd=1)
        lq = lockin[
            (lockin.dataset == ds) & (lockin.market == market) & (lockin.policy == "floor80_agg")
        ].set_index("instance")["frac"]
        li = lockin[
            (lockin.dataset == ds) & (lockin.market == market) & (lockin.policy == "indep")
        ].set_index("instance")["frac"]
        rec["Lock"] = _ci_cell(*_diff_boot(lq, li), nd=1, times=100)
        sq = _inst_spend_frac(buyers, ds, market, "floor80_agg")
        si = _inst_spend_frac(buyers, ds, market, "indep")
        d_sp, *_ = _diff_boot(sq, si)
        rec["Spend"] = rnd(100.0 * d_sp, 1)
        rows.append(rec)
    _md(pd.DataFrame(rows), OUT / "tab6_paired.md")
    return "tab6_paired Table 6 (QShare vs independent)"


def tab7():
    fr = pd.read_csv(MKT / "frontier.csv")
    buyers = pd.read_csv(MKT / "per_buyer.csv")
    rows = []
    for pol, F, store in FLOOR_ROWS:
        rec = dict(Store=store, F=F)
        for ds in ("amazon", "domainnet"):
            sub = fr[(fr.dataset == ds) & (fr.market == "base") & (fr.policy == pol)]
            if sub.empty:
                rec[f"{ds}_loss"] = rec[f"{ds}_lock"] = rec[f"{ds}_spend"] = ""
                continue
            r = sub.iloc[0]
            rec[f"{ds}_loss"] = rnd(r["red"], 1)
            rec[f"{ds}_lock"] = rnd(r["lock"], 1, times=100)
            rec[f"{ds}_spend"] = rnd(_spend_pct(buyers, ds, "base", pol), 1)
        rows.append(rec)
    _md(pd.DataFrame(rows), OUT / "tab7_floor_size.md")
    return "tab7_floor_size Table 7 (F=0/40/80, QShare and Row-level)"


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild paper Tables 2–7 and Figures 3–5 from results/."
    )
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    notes = [tab2(), tab3(), tab4(), tab5(), tab6(), tab7()]
    print("\n".join(notes))
    from paper.figures import _use_libertine, fig3, fig4, fig5
    _use_libertine()
    fig3()
    fig4()
    fig5()
    print("wrote Figures 3–5 to paper/out/")


if __name__ == "__main__":
    main()
