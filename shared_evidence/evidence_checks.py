#!/usr/bin/env python3
"""Body-text checks from a finished market run: buyer variance, in vs out, funded gap, first-buyer AUC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MKT = ROOT / "results" / "market"
BOOT_N, BOOT_SEED = 5000, 0
POLS = ("share_naive", "share_agg", "floor80_agg")


def _unit_ids(df):
    ds = df["dataset"].astype(str).to_numpy()
    mk = df["market"].astype(str).to_numpy()
    inst = np.asarray(df["instance"]).astype(str)
    return np.array([f"{a}|{b}|{c}" for a, b, c in zip(ds, mk, inst)], dtype=object)


def _qtile(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else np.nan


def boot_mean(df, cols, n=BOOT_N, seed=BOOT_SEED):
    empty = {"n_inst": 0, "n": 0}
    for c in cols:
        empty[c] = empty[f"{c}_lo"] = empty[f"{c}_hi"] = np.nan
    if df is None or len(df) == 0:
        return empty
    ids = _unit_ids(df)
    uniq, inv = np.unique(ids, return_inverse=True)
    mat = df[list(cols)].to_numpy(float)
    acc = np.zeros((len(uniq), mat.shape[1]))
    cnt = np.zeros((len(uniq), mat.shape[1]))
    ok = np.isfinite(mat)
    np.add.at(acc, inv, np.where(ok, mat, 0.0))
    np.add.at(cnt, inv, ok.astype(float))
    inst_mat = acc / np.maximum(cnt, 1.0)
    inst_mat[cnt == 0] = np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(uniq), size=(n, len(uniq)), replace=True)
    with np.errstate(all="ignore"):
        boots = np.nanmean(inst_mat[draws], axis=1)
    out = {"n_inst": int(len(uniq)), "n": int(len(df))}
    for j, c in enumerate(cols):
        out[c] = float(np.nanmean(inst_mat[:, j]))
        out[f"{c}_lo"] = _qtile(boots[:, j], 0.025)
        out[f"{c}_hi"] = _qtile(boots[:, j], 0.975)
    return out


def rel_bin(pi, qbin):
    if pd.notna(qbin) and str(qbin) != "":
        return f"qbin={qbin}"
    if not np.isfinite(pi):
        return "NA"
    for t in (0.5, 0.7, 0.9, 1.0):
        if abs(float(pi) - t) < 1e-9:
            return f"pi={t:.1f}"
    return "other"


def load_run(run_dir: Path):
    recs = pd.read_csv(run_dir / "records.csv")
    posts = pd.read_csv(run_dir / "posterior.csv")
    cat = run_dir / "record_rows.csv.gz"
    rows = pd.read_csv(cat) if cat.exists() else pd.DataFrame()
    recs["score_err"] = recs.gamma_bar - recs.q_jb
    recs["tot_err"] = recs.gamma_bar - recs.true_pi
    qmap = posts.groupby(["dataset", "market", "instance", "seller"], dropna=False)["qbin"].first()
    recs = recs.merge(qmap.rename("qbin").reset_index(), on=["dataset", "market", "instance", "seller"], how="left")
    recs["rel"] = [rel_bin(p, q) for p, q in zip(recs.true_pi, recs.qbin)]
    return recs, rows, posts


def als_mu(sellers, buyers, y, n_iter=80, tol=1e-12):
    s, b, y = np.asarray(sellers), np.asarray(buyers), np.asarray(y, float)
    if y.size == 0 or len(np.unique(s)) < 2 or len(np.unique(b)) < 2:
        return {}, False
    mu = float(np.mean(y))
    a_map = {int(k): float(v) for k, v in pd.Series(y - mu).groupby(s).mean().items()}
    b_map = {int(k): 0.0 for k in np.unique(b)}
    da = 0.0
    for _ in range(n_iter):
        resid_b = y - mu - np.array([a_map[int(x)] for x in s])
        b_new = pd.Series(resid_b).groupby(b).mean()
        b_new = b_new - b_new.mean()
        b_map = {int(k): float(v) for k, v in b_new.items()}
        resid_a = y - mu - np.array([b_map.get(int(x), 0.0) for x in b])
        a_new = pd.Series(resid_a).groupby(s).mean()
        prev = np.array([a_map.get(int(k), 0.0) for k in a_new.index], float)
        da = float(np.max(np.abs(a_new.to_numpy(float) - prev))) if len(a_new) else 0.0
        a_map = {int(k): float(v) for k, v in a_new.items()}
        if da < tol:
            break
    return b_map, (da < 1e-4)


def buyer_variance(recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs[recs.policy == "floor80_agg"].copy()
    raw = []
    for (ds, mk, inst), g in recs.groupby(["dataset", "market", "instance"]):
        y, s, b = g.score_err.to_numpy(float), g.seller.to_numpy(int), g.buyer_t.to_numpy(int)
        if pd.Series(s).nunique() < 5 or pd.Series(b).nunique() < 5 or len(g) < 50:
            continue
        b_map, ok = als_mu(s, b, y)
        if (not ok) or len(b_map) < 5:
            continue
        raw.append(dict(
            dataset=ds, market=mk, instance=int(inst),
            var_buyer=float(np.var(list(b_map.values()), ddof=1)),
        ))
    raw = pd.DataFrame(raw)
    want = [("amazon", "base"), ("domainnet", "base"), ("cifar10n", "annotator_strat")]
    rows = []
    for ds, mk in want:
        g = raw[(raw.dataset == ds) & (raw.market == mk)]
        if g.empty:
            continue
        rows.append(dict(dataset=ds, var_buyer=float(g.var_buyer.mean())))
    return pd.DataFrame(rows)


def task_inout(recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs[recs.policy == "floor80_agg"].copy()
    raw_rows = []
    for keys, g in recs.groupby(["dataset", "market", "instance", "rel", "buyer_t"]):
        ds, mk, inst, rel, t = keys
        inn, outt = g[g.in_task == 1], g[g.in_task == 0]
        if len(inn) < 5 or len(outt) < 5:
            continue
        raw_rows.append(dict(
            dataset=ds, market=mk, instance=int(inst),
            diff_in_minus_out=float(inn.score_err.mean() - outt.score_err.mean()),
        ))
    raw = pd.DataFrame(raw_rows)
    rows = []
    for ds, mk in (("amazon", "base"), ("domainnet", "base")):
        g = raw[(raw.dataset == ds) & (raw.market == mk)]
        if g.empty:
            continue
        st = boot_mean(g, ["diff_in_minus_out"])
        rows.append(dict(
            dataset=ds, diff=st["diff_in_minus_out"],
            diff_lo=st["diff_in_minus_out_lo"], diff_hi=st["diff_in_minus_out_hi"],
        ))
    return pd.DataFrame(rows)


def funded_gap(recs: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    rr = rows.copy()
    dirty = pd.to_numeric(rr.contaminated, errors="coerce")
    rr["row_err"] = rr.gamma.astype(float) - (1.0 - dirty)
    parts = rr.record_key.astype(str).str.split("|", expand=True)
    rr["dataset"], rr["market"] = parts[0], parts[1]
    rr["instance"] = pd.to_numeric(parts[2], errors="coerce")
    rr["policy"] = parts[3]
    rr["buyer_t"] = pd.to_numeric(parts[4], errors="coerce")
    rr["seller"] = pd.to_numeric(parts[5], errors="coerce")
    rr["funded"] = pd.to_numeric(rr.funded, errors="coerce").fillna(0).astype(int)
    meta = recs[["dataset", "market", "instance", "policy", "buyer_t", "seller", "rel"]].drop_duplicates()
    rr = rr.merge(meta, on=["dataset", "market", "instance", "policy", "buyer_t", "seller"], how="left")
    rr = rr[(rr.policy == "floor80_agg") & (rr.buyer_t == 1)]
    out = []
    for keys, g in rr.groupby(["dataset", "market", "rel"]):
        ds, mk, rel = keys
        f1, f0 = g[g.funded == 1], g[g.funded == 0]
        a = boot_mean(f1, ["row_err"]) if len(f1) else {"row_err": np.nan}
        b = boot_mean(f0, ["row_err"]) if len(f0) else {"row_err": np.nan}
        d = a["row_err"] - b["row_err"]
        out.append(dict(
            dataset=ds, market=mk, rel=rel,
            diff_funded_minus_self=d, abs_diff=abs(d) if np.isfinite(d) else np.nan,
        ))
    return pd.DataFrame(out)


def write_all(run_dir: Path, out_dir: Path, first_buyer: pd.DataFrame | None = None):
    recs, rows, _posts = load_run(run_dir)
    bv = buyer_variance(recs)
    io = task_inout(recs)
    fg = funded_gap(recs, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    bv.to_csv(out_dir / "buyer_variance.csv", index=False)
    io.to_csv(out_dir / "task_inout.csv", index=False)
    fg.to_csv(out_dir / "funded_gap.csv", index=False)
    if first_buyer is not None and len(first_buyer):
        first_buyer.to_csv(out_dir / "first_buyer_auc.csv", index=False)
    print("wrote", out_dir / "buyer_variance.csv", out_dir / "task_inout.csv",
          out_dir / "funded_gap.csv", flush=True)
    return bv, io, fg


def main():
    ap = argparse.ArgumentParser(description="Write body-text CSVs from a market run.")
    ap.add_argument("--run-dir", default=str(MKT))
    ap.add_argument("--out", default=str(MKT))
    args = ap.parse_args()
    write_all(Path(args.run_dir), Path(args.out))


if __name__ == "__main__":
    main()
