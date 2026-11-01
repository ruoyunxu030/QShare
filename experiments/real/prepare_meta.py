#!/usr/bin/env python3
"""Write per-dataset σ² and ridge λ into data_real/meta_<dataset>.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]


def _data_root():
    env = os.environ.get("QSHARE_DATA")
    if env:
        return Path(env)
    return ROOT / "data_real"


OUT = _data_root()
AMZ = OUT / "amazon"
DOM = OUT / "domainnet"
LAM_GRID = np.array([0.1, 1.0, 10.0, 100.0])
SEED = 20261017


def load_amazon_groups(encoder: str | None = None):
    path = AMZ / "amazon_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run prepare_amazon.py first")
    meta = json.loads(path.read_text())
    if meta.get("status") != "ready":
        raise RuntimeError(f"Amazon not ready: {meta.get('status')}")
    enc = encoder or meta.get("primary_encoder", "bge-base-en-v1.5")
    groups = {}
    for cat in meta["demand_categories"]:
        z = np.load(AMZ / f"cat_{enc}_{cat}.npz", allow_pickle=True)
        groups[cat] = dict(
            X_sell=z["X_sell"],
            y_sell=z["y_sell"],
            X_query=z["X_query"],
            X_test=z["X_test"],
            y_test=z["y_test"],
        )
    meta = dict(meta)
    meta["encoder_used"] = enc
    return groups, meta


def load_domainnet_groups(encoder: str = "clip-vit-l14"):
    meta_path = DOM / f"domainnet_meta_{encoder}.json"
    meta = json.loads(meta_path.read_text())
    groups = {}
    for d in meta["demand_categories"]:
        z = np.load(DOM / f"cat_{encoder}_{d}.npz")
        groups[d] = dict(
            X_sell=z["X_sell"],
            y_sell=z["y_sell"],
            X_query=z["X_query"],
            X_test=z["X_test"],
            y_test=z["y_test"],
        )
    meta = dict(meta)
    meta["encoder_used"] = encoder
    return groups, meta


def ols_sigma2(X, y):
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    if resid.ndim == 1:
        dof = max(len(y) - X.shape[1], 1)
        return float(np.sum(resid**2) / dof)
    return float(np.mean(np.sum(resid**2, axis=0) / max(len(y) - X.shape[1], 1)))


def cv_ridge_lambda(X, y):
    model = RidgeCV(alphas=LAM_GRID, cv=5, fit_intercept=False)
    model.fit(X, y)
    a = model.alpha_
    return float(np.mean(a) if np.ndim(a) else a)


def concept_shift(groups: dict, lam: float) -> pd.DataFrame:
    rows = []
    names = list(groups.keys())
    Xp = np.vstack([groups[g]["X_sell"] for g in names])
    yp = np.concatenate([groups[g]["y_sell"] for g in names])
    pooled = Ridge(alpha=lam, fit_intercept=False).fit(Xp, yp)
    w_pool = pooled.coef_
    for g in names:
        Xg, yg = groups[g]["X_sell"], groups[g]["y_sell"]
        Xt, yt = groups[g]["X_test"], groups[g]["y_test"]
        local = Ridge(alpha=lam, fit_intercept=False).fit(Xg, yg)
        w_loc = local.coef_
        rel_l2 = float(np.linalg.norm(w_loc - w_pool) / max(np.linalg.norm(w_pool), 1e-15))
        others = [h for h in names if h != g]
        Xo = np.vstack([groups[h]["X_sell"] for h in others])
        yo = np.concatenate([groups[h]["y_sell"] for h in others])
        other = Ridge(alpha=lam, fit_intercept=False).fit(Xo, yo)
        mse_other = float(mean_squared_error(yt, other.predict(Xt), multioutput="uniform_average"))
        mse_local = float(mean_squared_error(yt, local.predict(Xt), multioutput="uniform_average"))
        rows.append(
            dict(
                group=g,
                n_sell=len(yg),
                n_test=len(yt),
                coef_rel_l2_vs_pooled=rel_l2,
                mse_local=mse_local,
                mse_other_train=mse_other,
                mse_ratio_other_over_local=mse_other / max(mse_local, 1e-15),
            )
        )
    return pd.DataFrame(rows)


def process_dataset(name: str, groups: dict, ds_meta: dict) -> dict:
    Xp = np.vstack([groups[g]["X_sell"] for g in groups])
    yp = np.concatenate([groups[g]["y_sell"] for g in groups])
    notes = []
    sigma2 = ols_sigma2(Xp, yp)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        notes.append(f"WARNING sigma2 abnormal: {sigma2}")
    lam = cv_ridge_lambda(Xp, yp)
    shift = concept_shift(groups, lam)
    shift_path = OUT / f"concept_shift_{name}.csv"
    shift.to_csv(shift_path, index=False)
    info = dict(
        name=name,
        d=int(Xp.shape[1]),
        n_sell_total=int(len(yp)),
        sigma2=sigma2,
        ridge_lambda=lam,
        concept_shift_csv=str(shift_path.relative_to(ROOT)),
        concept_shift_summary=dict(
            coef_rel_l2_median=float(shift.coef_rel_l2_vs_pooled.median()),
            mse_ratio_median=float(shift.mse_ratio_other_over_local.median()),
            mse_ratio_mean=float(shift.mse_ratio_other_over_local.mean()),
        ),
        notes=notes,
        groups=list(groups.keys()),
        seed=SEED,
        lam_grid=LAM_GRID.tolist(),
    )
    for k in ("encoders", "pca", "primary_encoder", "encoder_used", "pca_dim"):
        if k in ds_meta:
            info[k] = ds_meta[k]
    return info


def write_merged_index():
    parts = {}
    for name in ("amazon", "domainnet"):
        p = OUT / f"meta_{name}.json"
        if p.exists():
            parts[name] = json.loads(p.read_text())
    index = dict(seed=SEED, datasets=list(parts.keys()), **parts)
    (OUT / "meta.json").write_text(json.dumps(index, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("amazon", "domainnet", "all"), default="amazon")
    ap.add_argument(
        "--amazon-encoder",
        default=None,
        help="Amazon encoder key (default: primary from amazon_meta.json)",
    )
    ap.add_argument("--domainnet-encoder", default="clip-vit-l14")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.dataset in ("amazon", "all"):
        targets.append(("amazon", lambda: load_amazon_groups(args.amazon_encoder)))
    if args.dataset in ("domainnet", "all"):
        targets.append(
            ("domainnet", lambda: load_domainnet_groups(args.domainnet_encoder))
        )

    for name, loader in targets:
        groups, ds_meta = loader()
        info = process_dataset(name, groups, ds_meta)
        for k in (
            "download_url",
            "download_urls",
            "version",
            "encoder",
            "pca_dim",
            "n_features_raw",
        ):
            if k in ds_meta:
                info[k] = ds_meta[k]
        path = OUT / f"meta_{name}.json"
        path.write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {path}")
        if name == "amazon":
            enc = ds_meta.get("encoder_used") or ds_meta.get("primary_encoder")
            if enc:
                enc_path = OUT / f"meta_amazon_{enc}.json"
                enc_path.write_text(json.dumps(info, indent=2) + "\n")
                print(f"Wrote {enc_path}")
        if name == "domainnet":
            enc = ds_meta.get("encoder_used", "clip-vit-l14")
            enc_path = OUT / f"meta_domainnet_{enc}.json"
            enc_path.write_text(json.dumps(info, indent=2) + "\n")
            print(f"Wrote {enc_path}")
        print(json.dumps(info, indent=2))

    write_merged_index()
    print("Updated", OUT / "meta.json")


if __name__ == "__main__":
    main()
