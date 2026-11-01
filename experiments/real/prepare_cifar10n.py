#!/usr/bin/env python3
"""CIFAR-10N: CLIP ViT-L/14 embed, PCA(32)+bias, seller pools.

Sources: aggregate / random1 / random2 / random3 / worst (human noisy labels).
Demand states (K=3 class groups): vehicle, pet, wildlife.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data_real" / "cifar10n"
SEED = 20261018
PKT_ROWS = 40
N_SELLERS = 8
PKTS_PER_SELLER = 5
ROWS_PER_SOURCE = N_SELLERS * PKTS_PER_SELLER * PKT_ROWS  # 1600

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DEMAND_GROUPS = {
    "vehicle": [0, 1, 8, 9],
    "pet": [3, 5],
    "wildlife": [2, 4, 6, 7],
}
LABEL_KEYS = {
    "aggregate": "aggre_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
    "worst": "worse_label",
}
HUMAN_URL = (
    "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
)


def stage_download():
    import tarfile
    import torchvision

    OUT.mkdir(parents=True, exist_ok=True)
    pt = OUT / "CIFAR-10_human.pt"
    tv = OUT / "torchvision"
    tv.mkdir(parents=True, exist_ok=True)
    if pt.exists() and pt.stat().st_size > 1_000_000:
        print("skip CIFAR-10N human labels", flush=True)
    else:
        print(f"download {HUMAN_URL}", flush=True)
        urllib.request.urlretrieve(HUMAN_URL, pt)
        print(f"wrote {pt} size={pt.stat().st_size}", flush=True)
    tar = tv / "cifar-10-python.tar.gz"
    extracted = (tv / "cifar-10-batches-py").exists()
    if extracted:
        print("skip torchvision CIFAR-10", flush=True)
    elif tar.exists() and tar.stat().st_size > 160_000_000:
        print(f"extract {tar}", flush=True)
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(tv)
    else:
        print("download CIFAR-10 via torchvision", flush=True)
        torchvision.datasets.CIFAR10(root=str(tv), train=True, download=True)
        torchvision.datasets.CIFAR10(root=str(tv), train=False, download=True)
    torchvision.datasets.CIFAR10(root=str(tv), train=True, download=False)
    torchvision.datasets.CIFAR10(root=str(tv), train=False, download=False)
    print("download done", flush=True)


def stage_embed(batch_size: int = 128, device: str = "cuda:0"):
    if (OUT / "emb_train.npy").exists() and (OUT / "emb_test.npy").exists():
        print("skip embed", flush=True)
        return
    import torch
    import clip
    import torchvision
    from PIL import Image

    print("embed start", flush=True)
    train = torchvision.datasets.CIFAR10(
        root=str(OUT / "torchvision"), train=True, download=False
    )
    test = torchvision.datasets.CIFAR10(
        root=str(OUT / "torchvision"), train=False, download=False
    )
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    def embed_split(ds, tag):
        n = len(ds)
        out = np.memmap(
            OUT / f"emb_raw_{tag}.npy", dtype=np.float32, mode="w+", shape=(n, 768)
        )
        with torch.no_grad():
            for i in range(0, n, batch_size):
                imgs = []
                for j in range(i, min(i + batch_size, n)):
                    img, _ = ds[j]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    imgs.append(preprocess(img))
                batch = torch.stack(imgs).to(device)
                feats = model.encode_image(batch).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out[i : i + len(imgs)] = feats.cpu().numpy()
                if (i // batch_size) % 20 == 0:
                    print(f"  {tag} {i}/{n}", flush=True)
        out.flush()
        arr = np.array(out)
        del out
        np.save(OUT / f"emb_{tag}.npy", arr)
        return arr

    print("embed train", flush=True)
    Xtr = embed_split(train, "train")
    print("embed test", flush=True)
    Xte = embed_split(test, "test")
    ytr = np.array([train[i][1] for i in range(len(train))], dtype=np.int64)
    yte = np.array([test[i][1] for i in range(len(test))], dtype=np.int64)
    np.save(OUT / "y_clean_train.npy", ytr)
    np.save(OUT / "y_clean_test.npy", yte)
    print(f"embed done n_train={len(Xtr)} n_test={len(Xte)}", flush=True)


def stage_features():
    needed = [OUT / f"source_{name}.npz" for name in LABEL_KEYS]
    needed += [OUT / f"demand_{g}.npz" for g in DEMAND_GROUPS]
    needed.append(OUT / "seed_pool.npz")
    if all(p.exists() for p in needed):
        print("skip features", flush=True)
        return
    import torch

    print("features start", flush=True)
    Xtr = np.load(OUT / "emb_train.npy")
    Xte = np.load(OUT / "emb_test.npy")
    ytr = np.load(OUT / "y_clean_train.npy")
    yte = np.load(OUT / "y_clean_test.npy")
    human = torch.load(
        OUT / "CIFAR-10_human.pt", map_location="cpu", weights_only=False
    )
    clean_h = np.asarray(human["clean_label"]).astype(np.int64)
    assert len(clean_h) == len(ytr)
    match = float(np.mean(clean_h == ytr))
    print(f"clean label match vs torchvision: {match:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    pca = PCA(n_components=32, random_state=SEED)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    sc = StandardScaler()
    Ztr = sc.fit_transform(Ztr)
    Zte = sc.transform(Zte)
    Xtr32 = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Xte32 = np.hstack([Zte, np.ones((len(Zte), 1))])
    d = Xtr32.shape[1]

    C = 10
    Yoh = np.eye(C)[ytr]
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=False, cv=5)
    idx = rng.choice(len(Xtr32), size=min(8000, len(Xtr32)), replace=False)
    ridge.fit(Xtr32[idx], Yoh[idx])
    lam = float(ridge.alpha_)
    pred = ridge.predict(Xtr32[idx])
    sigma2 = float(np.mean((pred - Yoh[idx]) ** 2))

    noise_rates = {}
    for name, key in LABEL_KEYS.items():
        ylab = np.asarray(human[key]).astype(np.int64)
        noise_rates[name] = float(np.mean(ylab != ytr))
        print(f"  noise {name}: {noise_rates[name]*100:.2f}%", flush=True)

    n_need = ROWS_PER_SOURCE * len(LABEL_KEYS)
    assert n_need <= len(Xtr32)
    perm = rng.permutation(len(Xtr32))
    sell_pool = perm[:n_need]
    rest = perm[n_need:]

    n_query_tr = min(2000, len(rest) // 2)
    query_tr_idx = rest[:n_query_tr]
    seed_pool_idx = rest[n_query_tr:]

    cursor = 0
    sources_meta = {}
    for name, key in LABEL_KEYS.items():
        ylab = np.asarray(human[key]).astype(np.int64)
        idx_src = sell_pool[cursor : cursor + ROWS_PER_SOURCE]
        cursor += ROWS_PER_SOURCE
        X_sell = Xtr32[idx_src]
        y_id_sell = ylab[idx_src]
        y_sell = np.eye(C)[y_id_sell]
        y_clean_sell = ytr[idx_src]
        np.savez_compressed(
            OUT / f"source_{name}.npz",
            X_sell=X_sell,
            y_sell=y_sell,
            y_id_sell=y_id_sell,
            y_clean_sell=y_clean_sell,
            sell_idx=idx_src,
        )
        sources_meta[name] = dict(
            n_sell=int(len(idx_src)),
            noise_rate=noise_rates[name],
        )

    for gname, classes in DEMAND_GROUPS.items():
        cls = set(classes)
        q_mask = np.array([ytr[i] in cls for i in query_tr_idx])
        q_idx = query_tr_idx[q_mask]
        t_mask = np.isin(yte, classes)
        t_idx = np.where(t_mask)[0]
        np.savez_compressed(
            OUT / f"demand_{gname}.npz",
            X_query=Xtr32[q_idx] if len(q_idx) else Xtr32[:1],
            X_test=Xte32[t_idx],
            y_test=np.eye(C)[yte[t_idx]],
            y_id_test=yte[t_idx],
            classes=np.array(classes),
        )
        print(
            f"  demand {gname}: query={len(q_idx)} test={len(t_idx)}",
            flush=True,
        )

    np.savez_compressed(
        OUT / "seed_pool.npz",
        X=Xtr32[seed_pool_idx],
        y=np.eye(C)[ytr[seed_pool_idx]],
        y_id=ytr[seed_pool_idx],
        idx=seed_pool_idx,
    )

    meta = dict(
        dataset="cifar10n",
        d=d,
        pca_dim=32,
        ridge_lambda=lam,
        sigma2=sigma2,
        packet_rows=PKT_ROWS,
        n_sellers=N_SELLERS,
        pkts_per_seller=PKTS_PER_SELLER,
        sources=list(LABEL_KEYS.keys()),
        demand_categories=list(DEMAND_GROUPS.keys()),
        noise_rates=noise_rates,
        clean_match_rate=match,
        n_seed_tier="2d",
        seed=SEED,
        encoder="clip-vit-l14",
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
    )
    (OUT / "cifar10n_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ROOT / "data_real" / "meta_cifar10n.json").write_text(
        json.dumps(
            dict(
                name="cifar10n",
                d=d,
                sigma2=sigma2,
                ridge_lambda=lam,
                groups=list(DEMAND_GROUPS.keys()),
                sources=list(LABEL_KEYS.keys()),
                noise_rates=noise_rates,
            ),
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        OUT / "pca_scaler.npz",
        components=pca.components_,
        mean=pca.mean_,
        scaler_mean=sc.mean_,
        scaler_scale=sc.scale_,
    )
    print("features done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=["download", "embed", "features", "all"],
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.stage in ("download", "all"):
        stage_download()
    if args.stage in ("embed", "all"):
        stage_embed(batch_size=args.batch_size, device=args.device)
    if args.stage in ("features", "all"):
        stage_features()


if __name__ == "__main__":
    main()
