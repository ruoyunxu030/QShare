"""Write the six market CSVs used by paper.tables_figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = [
    "indep", "share_naive", "share_ts", "share_agg", "share_agg_ts",
    "share_agg_ucb",
    "floor40_row", "floor80_row", "floor40_agg", "floor80_agg",
    "fund80_agg",
]
PAIRED_VS_INDEP = ("floor80_agg", "floor40_agg", "share_agg")
N_BOOT = 5000
BUYER_COLS = [
    "dataset", "market", "instance", "buyer_t", "policy",
    "excess", "test_loss", "small_purchase_spend", "packet_spend",
    "n_checked", "platform_paid",
]


def paper_policies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "policy" not in df.columns:
        return df
    return df[df.policy.isin(POLICIES)].copy()


def slim_buyers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = paper_policies(df.copy())
    cols = [c for c in BUYER_COLS if c in out.columns]
    return out[cols]


def _ratio_pct_boot(e_m, e_ref, n_boot=N_BOOT, seed=0):
    e_m = np.asarray(e_m, float)
    e_ref = np.asarray(e_ref, float)
    if len(e_m) == 0 or len(e_m) != len(e_ref):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(e_m)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = 100.0 * (1.0 - e_m[idx].mean(1) / np.maximum(np.abs(e_ref[idx].mean(1)), 1e-15))
    point = 100.0 * (1.0 - float(e_m.mean()) / max(abs(float(e_ref.mean())), 1e-15))
    return point, float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def _mean_boot(x, n_boot=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _diff_boot(a, b, n_boot=N_BOOT, seed=0):
    """Paired a-b. Sort by instance id before resampling (seed depends on order)."""
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    common = pd.Index(sorted(a.index.intersection(b.index)))
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _diff_boot_ordered(a, b, n_boot=N_BOOT, seed=0):
    """Paired difference in the given instance order (no re-sort)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    common = np.isfinite(a) & np.isfinite(b)
    a, b = a[common], b[common]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def _inst_means(df, dataset, market, policy, t0, t1, col):
    sub = df[
        (df.dataset == dataset) & (df.market == market)
        & (df.policy == policy) & (df.buyer_t >= t0) & (df.buyer_t <= t1)
    ]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("instance")[col].mean().sort_index()


def _loss_col(dataset):
    return "test_loss" if dataset == "cifar10n" else "excess"


def _good_sellers(frame, ds):
    """π=1 sellers in this instance (CIFAR-10N: top accuracy quartile).

    `frame` must be the market-config universe (posterior), not payments.
    Sellers who never appear in payments are still in the denominator.
    """
    if frame is None or len(frame) == 0:
        return pd.Index([])
    if ds == "cifar10n":
        if "qbin" in frame.columns and frame["qbin"].notna().any():
            qb = frame.groupby("seller").qbin.first()
            return qb.index[qb.astype(str).isin(("3", "3.0", "Q3"))]
        pis = frame.groupby("seller").true_pi.first()
        return pis.index[pis >= pis.quantile(0.75)]
    pis = frame.groupby("seller").true_pi.first()
    return pis.index[np.asarray(pis, float) >= 0.999]


def _universe(posts, pays, ds, market, inst, pol):
    """Seller list for one instance: posterior if present, else payments."""
    if posts is not None and len(posts):
        g = posts[
            (posts.dataset == ds) & (posts.market == market)
            & (posts.instance == inst) & (posts.policy == pol)
        ]
        if len(g):
            return g
    if pays is not None and len(pays):
        return pays[
            (pays.dataset == ds) & (pays.market == market)
            & (pays.instance == inst) & (pays.policy == pol)
        ]
    return pd.DataFrame()


def _lock_frac(gpay, ds, good=None):
    if good is None:
        good = _good_sellers(gpay, ds)
    n = len(good)
    if n == 0:
        return np.nan
    if gpay is None or len(gpay) == 0:
        return 1.0
    after = gpay[(gpay.buyer_t >= 11) & (gpay.buyer_t <= 30)]
    bought = set()
    if len(after) and "bought_packet" in after.columns:
        bought = set(after.loc[after.bought_packet == 1, "seller"].tolist())
    return sum(1 for s in good if s not in bought) / n


def _unsold(gpay, eid):
    if eid is None:
        return np.nan
    ge = gpay[gpay.seller == eid]
    return 0.0 if len(ge) and (ge["bought_packet"] == 1).any() else 1.0


def _ev_frac(gbuy, gpay):
    ev = float(gbuy["small_purchase_spend"].sum()) if len(gbuy) and "small_purchase_spend" in gbuy.columns else 0.0
    plat = float(gpay["platform_paid"].sum()) if len(gpay) and "platform_paid" in gpay else 0.0
    buyer_paid = float(gpay["paid"].sum()) if len(gpay) else 0.0
    tot = buyer_paid + plat
    if tot <= 0:
        return np.nan
    return (ev + plat) / tot


def write_report(out_dir: Path):
    out_dir = Path(out_dir)
    buyers = pd.read_csv(out_dir / "per_buyer.csv") if (out_dir / "per_buyer.csv").exists() else pd.DataFrame()
    pays = pd.read_csv(out_dir / "payments.csv") if (out_dir / "payments.csv").exists() else pd.DataFrame()
    posts = pd.read_csv(out_dir / "posterior.csv") if (out_dir / "posterior.csv").exists() else pd.DataFrame()
    metas = json.loads((out_dir / "run_meta.json").read_text()) if (out_dir / "run_meta.json").exists() else []
    buyers = slim_buyers(buyers) if len(buyers) else buyers
    pays = paper_policies(pays) if len(pays) else pays
    posts = paper_policies(posts) if len(posts) else posts
    if len(buyers):
        buyers.to_csv(out_dir / "per_buyer.csv", index=False)
    if buyers.empty:
        print(f"wrote no market tables (missing per_buyer.csv) under {out_dir}", flush=True)
        return

    entry_id = {}
    for m in metas:
        if m.get("entry_seller_id") is not None:
            entry_id[(m["dataset"], int(m["instance"]))] = int(m["entry_seller_id"])

    _write_frontier(out_dir, buyers, pays, posts, entry_id)
    _write_lockin(out_dir, pays, posts)
    _write_entry(out_dir, pays, posts, entry_id)
    _write_first_order(out_dir, pays, posts)
    _write_paired_vs_indep(out_dir, buyers, pays, posts, entry_id)
    print(f"wrote market tables under {out_dir}", flush=True)


def _write_frontier(out_dir, buyers, pays, posts, entry_id):
    main_rows = []
    if pays.empty:
        return
    last = pays[(pays.buyer_t >= 16) & (pays.buyer_t <= 30)]
    for ds in sorted(buyers.dataset.unique()):
        col = _loss_col(ds)
        for market in sorted(buyers[buyers.dataset == ds].market.unique()):
            e_ind = _inst_means(buyers, ds, market, "indep", 16, 30, col)
            for pol in POLICIES:
                em = _inst_means(buyers, ds, market, pol, 16, 30, col)
                if em.empty:
                    continue
                common = em.index.intersection(e_ind.index)
                red, rlo, rhi = _ratio_pct_boot(em.loc[common].to_numpy(), e_ind.loc[common].to_numpy())
                locks, unsolds, ev_tot, plat, plat_frac, hhis = [], [], [], [], [], []
                gpay = pays[(pays.dataset == ds) & (pays.market == market) & (pays.policy == pol)]
                gbuy = buyers[(buyers.dataset == ds) & (buyers.market == market) & (buyers.policy == pol)]
                for inst in common:
                    gp = gpay[gpay.instance == inst]
                    good = _good_sellers(_universe(posts, pays, ds, market, inst, pol), ds)
                    locks.append(_lock_frac(gp, ds, good=good))
                    unsolds.append(_unsold(gp, entry_id.get((ds, int(inst)))) if market == "entry" else np.nan)
                    gb = gbuy[gbuy.instance == inst]
                    ev = float(gb["small_purchase_spend"].sum()) if len(gb) else 0.0
                    plat_i = float(gp["platform_paid"].sum()) if "platform_paid" in gp else 0.0
                    buyer_paid = float(gp["paid"].sum()) if len(gp) else 0.0
                    tot = buyer_paid + plat_i
                    ev_tot.append(ev + plat_i)
                    plat.append(plat_i)
                    plat_frac.append((ev + plat_i) / tot if tot > 0 else np.nan)
                    gl = last[(last.dataset == ds) & (last.market == market) & (last.policy == pol) & (last.instance == inst)]
                    if len(gl):
                        by_s = gl.groupby("seller").paid.sum()
                        tpaid = float(by_s.sum())
                        hhis.append(float(((by_s / tpaid) ** 2).sum()) if tpaid > 0 else np.nan)
                lm, *_ = _mean_boot(np.asarray(locks, float))
                um, *_ = _mean_boot(np.asarray(unsolds, float))
                ev, *_ = _mean_boot(np.asarray(ev_tot, float))
                pm, *_ = _mean_boot(np.asarray(plat, float))
                hm, *_ = _mean_boot(np.asarray(hhis, float))
                main_rows.append(dict(
                    dataset=ds, market=market, policy=pol, red=red, lock=lm, unsold=um,
                    ev_spend=ev, plat=pm, hhi=hm,
                ))
    if main_rows:
        pd.DataFrame(main_rows).to_csv(out_dir / "frontier.csv", index=False)


def _write_lockin(out_dir, pays, posts):
    if pays.empty and posts.empty:
        return
    lock_rows = []
    if not posts.empty:
        key_df = posts[["dataset", "market", "instance", "policy"]].drop_duplicates()
    else:
        key_df = pays[["dataset", "market", "instance", "policy"]].drop_duplicates()
    for _, key in key_df.iterrows():
        ds, market, inst, pol = key.dataset, key.market, key.instance, key.policy
        good = _good_sellers(_universe(posts, pays, ds, market, inst, pol), ds)
        if len(good) == 0:
            continue
        gpay = pays[
            (pays.dataset == ds) & (pays.market == market)
            & (pays.instance == inst) & (pays.policy == pol)
        ] if not pays.empty else pd.DataFrame()
        frac = _lock_frac(gpay, ds, good=good)
        n_lock = int(round(frac * len(good))) if np.isfinite(frac) else 0
        lock_rows.append(dict(
            dataset=ds, market=market, instance=int(inst), policy=pol,
            n_good=int(len(good)), n_locked=int(n_lock),
            frac=float(frac) if np.isfinite(frac) else np.nan,
        ))
    if lock_rows:
        pd.DataFrame(lock_rows).to_csv(out_dir / "lockin_pi1.csv", index=False)


def _write_entry(out_dir, pays, posts, entry_id):
    if not entry_id or pays.empty:
        return
    rows_out = []
    ep = pays[pays.market == "entry"]
    for (ds, inst, pol), g in ep.groupby(["dataset", "instance", "policy"]):
        eid = entry_id.get((ds, int(inst)))
        if eid is None:
            continue
        ge = g[g.seller == eid]
        first = ge[ge.bought_packet == 1]
        sold = not first.empty
        first_t = int(first.buyer_t.min()) if sold else np.nan
        plat = float(ge["platform_paid"].sum()) if "platform_paid" in ge else 0.0
        pkt = float(ge["packet_paid"].sum()) if len(ge) else 0.0
        small_inc = float(ge["paid"].sum()) - pkt if len(ge) else 0.0
        ev = {}
        if len(posts):
            gpst = posts[
                (posts.dataset == ds) & (posts.market == "entry")
                & (posts.instance == inst) & (posts.policy == pol) & (posts.seller == eid)
            ]
            col_ev = "n_rows_ev" if "n_rows_ev" in gpst.columns else "n_distinct"
            for tcut in (15, 20, 30):
                sl = gpst[gpst.buyer_t == tcut]
                ev[tcut] = float(sl[col_ev].iloc[0]) if len(sl) else np.nan
        rows_out.append(dict(
            dataset=ds, instance=int(inst), policy=pol, entry_seller=int(eid),
            sold=int(sold), first_packet_t=first_t,
            wait_buyers=(first_t - 15) if sold else np.nan,
            income_small=small_inc, income_packet=pkt, income_plat=plat,
            ev15=ev.get(15), ev20=ev.get(20), ev30=ev.get(30),
        ))
    ndf = pd.DataFrame(rows_out)
    if len(ndf):
        ndf.to_csv(out_dir / "entry_newcomer.csv", index=False)


def _write_first_order(out_dir, pays, posts):
    if posts.empty:
        return
    keys = ["dataset", "market", "instance", "policy", "seller"]
    uni = posts.groupby(keys, sort=True).true_pi.first().reset_index()
    if pays.empty or "packets_bought" not in pays.columns:
        uni["first_order_t"] = np.nan
    else:
        bought = pays[pays.packets_bought > 0]
        first = bought.groupby(keys, sort=True).buyer_t.min().rename("first_order_t")
        uni = uni.merge(first, on=keys, how="left")
    uni = uni.sort_values(keys)
    uni.to_csv(out_dir / "first_order.csv", index=False)


def _write_paired_vs_indep(out_dir, buyers, pays, posts, entry_id):
    if buyers.empty or pays.empty:
        return
    recs = []
    markets = (
        ("amazon", "base"),
        ("amazon", "entry"),
        ("domainnet", "base"),
        ("domainnet", "entry"),
        ("cifar10n", "annotator_strat"),
    )
    for ds, market in markets:
        col = _loss_col(ds)
        e_ind = _inst_means(buyers, ds, market, "indep", 16, 30, col)
        if e_ind.empty:
            continue
        gpay_i = pays[(pays.dataset == ds) & (pays.market == market) & (pays.policy == "indep")]
        gbuy_i = buyers[(buyers.dataset == ds) & (buyers.market == market) & (buyers.policy == "indep")]
        lock_i, unsold_i, ev_i = [], [], []
        for inst in e_ind.index:
            gp = gpay_i[gpay_i.instance == inst]
            good = _good_sellers(_universe(posts, pays, ds, market, inst, "indep"), ds)
            lock_i.append(_lock_frac(gp, ds, good=good))
            unsold_i.append(_unsold(gp, entry_id.get((ds, int(inst)))) if market == "entry" else np.nan)
            ev_i.append(_ev_frac(gbuy_i[gbuy_i.instance == inst], gp))
        lock_i = pd.Series(lock_i, index=e_ind.index, dtype=float)
        unsold_i = pd.Series(unsold_i, index=e_ind.index, dtype=float)
        ev_i = pd.Series(ev_i, index=e_ind.index, dtype=float)
        for pol in PAIRED_VS_INDEP:
            em = _inst_means(buyers, ds, market, pol, 16, 30, col)
            if em.empty:
                continue
            common = em.index.intersection(e_ind.index)
            if len(common) == 0:
                continue
            gpay_p = pays[(pays.dataset == ds) & (pays.market == market) & (pays.policy == pol)]
            gbuy_p = buyers[(buyers.dataset == ds) & (buyers.market == market) & (buyers.policy == pol)]
            lock_p, unsold_p, ev_p = [], [], []
            for inst in common:
                gp = gpay_p[gpay_p.instance == inst]
                good = _good_sellers(_universe(posts, pays, ds, market, inst, pol), ds)
                lock_p.append(_lock_frac(gp, ds, good=good))
                unsold_p.append(_unsold(gp, entry_id.get((ds, int(inst)))) if market == "entry" else np.nan)
                ev_p.append(_ev_frac(gbuy_p[gbuy_p.instance == inst], gp))
            lock_p = pd.Series(lock_p, index=common, dtype=float)
            unsold_p = pd.Series(unsold_p, index=common, dtype=float)
            ev_p = pd.Series(ev_p, index=common, dtype=float)
            metrics = [
                ("excess_t16_30", em.loc[common], e_ind.loc[common], "arr"),
                ("lock_pi1", lock_p, lock_i.loc[common], "inst"),
                ("newcomer_unsold", unsold_p, unsold_i.loc[common], "arr"),
                ("evidence_frac", ev_p, ev_i.loc[common], "arr"),
            ]
            for name, a, b, kind in metrics:
                if name == "newcomer_unsold" and market != "entry":
                    mean, lo, hi = np.nan, np.nan, np.nan
                elif kind == "inst":
                    mean, lo, hi = _diff_boot(a, b)
                else:
                    mean, lo, hi = _diff_boot_ordered(np.asarray(a, float), np.asarray(b, float))
                recs.append(dict(
                    dataset=ds, market=market,
                    contrast=f"{pol} \u2212 indep",
                    policy=pol, metric=name,
                    mean=mean, ci_lo=lo, ci_hi=hi, n=int(len(common)),
                ))
    df = pd.DataFrame(recs)
    if len(df):
        df.to_csv(out_dir / "paired_vs_indep.csv", index=False)
