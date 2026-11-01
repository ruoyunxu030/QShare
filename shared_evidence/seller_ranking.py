#!/usr/bin/env python3
"""Seller ranking at t=30 and body-text CSVs from a finished market run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared_evidence.evidence_checks import write_all

ROOT = Path(__file__).resolve().parents[1]
MKT = ROOT / "results" / "market"
ORDER = ("amazon", "domainnet", "cifar10n")
UNK = -1e9


def _auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def _visible(posts):
    if "appear_at" not in posts.columns:
        return posts.copy()
    aa = pd.to_numeric(posts.appear_at, errors="coerce")
    bt = pd.to_numeric(posts.buyer_t, errors="coerce")
    return posts.loc[aa.isna() | (aa <= bt)].copy()


def _reconstruct_small(pi, pays):
    pi = pi.copy()
    if "did_small" in pi.columns:
        pi["did_small"] = pd.to_numeric(pi.did_small, errors="coerce").fillna(0).astype(int)
        return pi
    keys = ["dataset", "market", "instance", "policy", "buyer_t", "seller"]
    p = pays.copy()
    p["did_small"] = (p.paid > p.packet_paid + 1e-12).astype(int)
    p = p[keys + ["did_small"]].drop_duplicates(keys)
    out = pi.merge(p, on=keys, how="left")
    out["did_small"] = out.did_small.fillna(0).astype(int)
    return out


def _metrics(score, pi, src):
    score, pi = np.asarray(score, float), np.asarray(pi, float)
    src = np.asarray(src).astype(str)
    out = dict(auc=np.nan, top2=np.nan)
    ok = np.isfinite(score) & np.isfinite(pi)
    if ok.sum() >= 3:
        best = (pi[ok] == np.nanmax(pi[ok])).astype(int)
        out["auc"] = _auc(best, score[ok])
    top2, nseg, nok = [], 0, 0
    for seg in np.unique(src):
        mg = src == seg
        nseg += 1
        if np.isfinite(score[mg]).sum() < 2:
            continue
        nok += 1
        order = np.argsort(-np.where(np.isfinite(score[mg]), score[mg], UNK))[:2]
        bestv = np.nanmax(pi[mg])
        top2.append(float(np.mean(pi[mg][order] == bestv)))
    out["top2"] = float(np.mean(top2)) if top2 else np.nan
    return out


def compare_instances(pi_hats, pays, posts, t=30, phase="decision", policies=("floor80_agg",)):
    ind = _reconstruct_small(pi_hats[pi_hats.policy == "indep"].copy(), pays)
    share = _visible(posts)
    share = share[share.policy.isin(policies)].copy()
    keys = ["dataset", "market", "instance", "seller"]
    a_t = ind[ind.buyer_t == t].copy()
    pt = t - 1 if phase == "decision" else t
    skip = pt < 1
    b = share[share.buyer_t == (t if skip else pt)][
        keys + ["policy", "post_mean", "true_pi", "source"]
    ].copy()
    if skip:
        b["post_mean"] = np.nan
    a_join = a_t[keys + ["pi_small", "pi_final", "did_small", "true_pi"]].copy()
    vis = b.merge(a_join, on=keys, how="left", suffixes=("_s", "_i"))
    if "true_pi_s" in vis.columns:
        vis["true_pi"] = vis["true_pi_s"]
    vis["did_small"] = vis.did_small.fillna(0).astype(int)
    vis["hat"] = vis.pi_small.astype(float) if phase == "decision" else vis.pi_final.astype(float)
    rows = []
    for (ds, mk, inst, pol), g0 in vis.groupby(["dataset", "market", "instance", "policy"]):
        n_vis = int(len(g0))
        n_small = int((g0.did_small == 1).sum())
        frac = n_small / n_vis if n_vis else np.nan
        common = g0[g0.did_small == 1]
        if phase == "after":
            common = common[np.isfinite(common.hat)]
        scopes = [("common", common)]
        if phase == "decision":
            scopes.append(("all_filled", g0))
        for scope, g in scopes:
            if g.empty:
                continue
            src = g.source.astype(str).to_numpy() if "source" in g.columns else np.array([""] * len(g))
            mh = _metrics(g.hat.to_numpy(float), g.true_pi.to_numpy(float), src)
            mp = _metrics(g.post_mean.to_numpy(float), g.true_pi.to_numpy(float), src) if not skip else dict(auc=np.nan, top2=np.nan)
            rows.append(dict(
                dataset=ds, market=mk, instance=int(inst), policy=pol,
                scope=scope, frac_indep=frac,
                hat_auc=mh["auc"], post_auc=mp["auc"],
                hat_top2=mh["top2"], post_top2=mp["top2"],
            ))
    return pd.DataFrame(rows)


def assemble(inst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in ORDER:
        g = inst[inst.dataset == ds]
        if g.empty:
            continue
        seen = float(g.frac_indep.mean())
        com, allf = g[g.scope == "common"], g[g.scope == "all_filled"]
        rows.append(dict(
            dataset=ds, seen=seen,
            auc_seen_indep=float(com.hat_auc.mean()),
            auc_seen_qshare=float(com.post_auc.mean()),
            auc_all_indep=float(allf.hat_auc.mean()),
            auc_all_qshare=float(allf.post_auc.mean()),
            top2_indep=float(allf.hat_top2.mean()),
            top2_qshare=float(allf.post_top2.mean()),
        ))
    return pd.DataFrame(rows)


def first_buyer_auc(pi_hats, pays, posts) -> pd.DataFrame:
    """Amazon, after the first buyer: AUC of the published mean on every seller indep scored."""
    ind = pi_hats[(pi_hats.policy == "indep") & (pi_hats.buyer_t == 1)].copy()
    keys = ["dataset", "market", "instance", "seller"]
    a = ind[keys + ["pi_final", "true_pi"]].rename(columns={"pi_final": "hat"})
    recs = []
    lab = {"floor80_agg": "QShare", "share_agg": "QShare without floor"}
    for pol in lab:
        b = posts[(posts.policy == pol) & (posts.buyer_t == 1)][
            keys + ["post_mean", "true_pi"]
        ]
        m = a.merge(b, on=keys, suffixes=("_i", "_s"), how="inner")
        if "true_pi_i" in m.columns:
            m["true_pi"] = m["true_pi_i"]
        m = m[m.dataset == "amazon"]
        aucs = []
        for _, g in m.groupby(["market", "instance"]):
            tru = g.true_pi.to_numpy(float)
            pm = g.post_mean.to_numpy(float)
            aucs.append(_auc((tru == np.nanmax(tru)).astype(int), pm))
        recs.append(dict(method=lab[pol], auc=float(np.mean(aucs)) if aucs else np.nan))
    return pd.DataFrame(recs)


def from_run(run_dir: Path, indep_dir: Path | None, out_dir: Path):
    run_dir = Path(run_dir)
    indep_dir = Path(indep_dir) if indep_dir else run_dir
    out_dir = Path(out_dir)
    posts = pd.read_csv(run_dir / "posterior.csv")
    pi = pd.read_csv(indep_dir / "pi_hats.csv")
    pays = pd.read_csv(indep_dir / "payments.csv")
    has_t30 = (("buyer_t" in pi.columns) and (pi["buyer_t"] == 30).any()) or (
        ("buyer_t" in posts.columns) and (posts["buyer_t"] == 30).any()
    )
    if not has_t30:
        print(
            f"Table 3 needs a full market run with 30 buyers including the indep policy; "
            f"no buyer t=30 found in {run_dir}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    inst = compare_instances(pi, pays, posts, t=30, phase="decision", policies=("floor80_agg",))
    keep = (
        ((inst.dataset == "amazon") & (inst.market == "base"))
        | ((inst.dataset == "domainnet") & (inst.market == "base"))
        | (inst.dataset == "cifar10n")
    )
    inst = inst[keep][["dataset", "market", "instance", "scope", "frac_indep",
                       "hat_auc", "post_auc", "hat_top2", "post_top2"]]
    tab = assemble(inst)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst.to_csv(out_dir / "seller_ranking_instances.csv", index=False)
    tab.to_csv(out_dir / "seller_ranking.csv", index=False)
    print("wrote", out_dir / "seller_ranking.csv", "rows", len(tab), flush=True)
    fb = first_buyer_auc(pi, pays, posts)
    write_all(run_dir, out_dir, first_buyer=fb)


def main():
    ap = argparse.ArgumentParser(
        description="From a market run: Table 3 CSVs and body-text CSVs."
    )
    ap.add_argument("--run-dir", default=str(MKT),
                    help="directory containing the saved market-run outputs")
    ap.add_argument("--indep-dir", default=None,
                    help="directory with pi_hats.csv and payments.csv (default: --run-dir)")
    ap.add_argument("--out", default=str(MKT))
    ap.add_argument("--instances", default=None,
                    help="if set, only assemble seller_ranking.csv from this instance file")
    args = ap.parse_args()
    if args.instances:
        inst = pd.read_csv(args.instances)
        df = assemble(inst)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = Path(args.out)
        if out.suffix != ".csv":
            out = out / "seller_ranking.csv"
        df.to_csv(out, index=False)
        print("wrote", out, "rows", len(df))
        return
    from_run(Path(args.run_dir), args.indep_dir, Path(args.out))


if __name__ == "__main__":
    main()
