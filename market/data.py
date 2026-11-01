"""Load prepared Amazon, DomainNet, and CIFAR-10N arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from market.const import PKT_ROWS, data_root

ROOT = Path(__file__).resolve().parents[1]
AMZ = data_root() / "amazon"
DOM = data_root() / "domainnet"
META_DIR = data_root()


def _load_dataset_meta(dataset: str) -> dict:
    """Prefer per-dataset meta_<name>.json; fall back to merged meta.json."""
    per = META_DIR / f"meta_{dataset}.json"
    if per.exists():
        return json.loads(per.read_text())
    merged = META_DIR / "meta.json"
    if merged.exists():
        blob = json.loads(merged.read_text())
        if dataset in blob:
            return blob[dataset]
    raise FileNotFoundError(f"missing meta for {dataset}: tried {per} and {merged}")


def load_groups(
    dataset: str,
    amazon_encoder: str | None = None,
    domainnet_encoder: str = "clip-vit-l14",
):
    if dataset == "amazon":
        meta = _load_dataset_meta(dataset)
        gmeta = json.loads((AMZ / "amazon_meta.json").read_text())
        enc = amazon_encoder or gmeta.get("primary_encoder") or meta.get(
            "primary_encoder", "bge-base-en-v1.5"
        )
        enc_meta_path = META_DIR / f"meta_amazon_{enc}.json"
        if enc_meta_path.exists():
            meta = json.loads(enc_meta_path.read_text())
        groups = {}
        for cat in gmeta["demand_categories"]:
            z = np.load(AMZ / f"cat_{enc}_{cat}.npz", allow_pickle=True)
            groups[cat] = {k: z[k] for k in z.files if k != "category"}
        pkt_rows = PKT_ROWS
    elif dataset == "domainnet":
        mpath = META_DIR / f"meta_domainnet_{domainnet_encoder}.json"
        if mpath.exists():
            meta = json.loads(mpath.read_text())
        else:
            meta = _load_dataset_meta("domainnet")
        gmeta = json.loads((DOM / f"domainnet_meta_{domainnet_encoder}.json").read_text())
        groups = {}
        for cat in gmeta["demand_categories"]:
            z = np.load(DOM / f"cat_{domainnet_encoder}_{cat}.npz")
            groups[cat] = dict(
                X_sell=z["X_sell"],
                y_sell=z["y_sell"],
                X_query=z["X_query"],
                X_test=z["X_test"],
                y_test=z["y_test"],
            )
        pkt_rows = PKT_ROWS
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return groups, float(meta["sigma2"]), float(meta["ridge_lambda"]), int(meta["d"]), pkt_rows


def second_moment(X: np.ndarray) -> np.ndarray:
    return (X.T @ X) / max(len(X), 1)


def load_cifar10n():
    OUT = data_root() / "cifar10n"
    meta = json.loads((OUT / "cifar10n_meta.json").read_text())
    groups = {}
    for g in meta["demand_categories"]:
        z = np.load(OUT / f"demand_{g}.npz")
        groups[g] = {k: z[k] for k in z.files}
    sources = {}
    for s in meta["sources"]:
        z = np.load(OUT / f"source_{s}.npz")
        sources[s] = {k: z[k] for k in z.files}
    seed_pool = dict(np.load(OUT / "seed_pool.npz"))
    return groups, sources, seed_pool, meta


def load_v2(dataset: str):
    enc = "bge-base-en-v1.5" if dataset == "amazon" else "clip-vit-l14"
    groups, sigma2, lam, d, _ = load_groups(
        dataset,
        amazon_encoder=enc if dataset == "amazon" else None,
        domainnet_encoder=enc if dataset == "domainnet" else "clip-vit-l14",
    )
    if dataset == "domainnet":
        for cat in list(groups.keys()):
            z = np.load(DOM / f"cat_{enc}_{cat}.npz")
            if "y_id_sell" in z.files:
                groups[cat]["y_id_sell"] = z["y_id_sell"]
                groups[cat]["y_id_test"] = z["y_id_test"]
    return groups, float(sigma2), float(lam), int(d)


def _get_bundle(cache, dataset):
    if dataset in cache:
        return cache[dataset]
    if dataset == "cifar10n":
        groups, sources, seed_pool, meta = load_cifar10n()
        bundle = dict(
            mode="cifar",
            groups=groups,
            sources=sources,
            seed_pool=seed_pool,
            meta=meta,
            sigma2=float(meta["sigma2"]),
            lam=float(meta["ridge_lambda"]),
            d=int(meta["d"]),
        )
    else:
        groups, sigma2, lam, d = load_v2(dataset)
        bundle = dict(mode="v2", groups=groups, sigma2=sigma2, lam=lam, d=d)
    cache[dataset] = bundle
    return bundle


def draw_seed_cifar(rng, seed_pool, n_seed, d, lam):
    X = seed_pool["X"]
    y = seed_pool["y"]
    idx = rng.choice(len(X), size=n_seed, replace=False)
    X0, y0 = X[idx], y[idx]
    M0 = lam * np.eye(d) + X0.T @ X0
    return M0, X0, y0
