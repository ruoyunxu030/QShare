#!/usr/bin/env python3
"""DomainNet (cleaned): download, select classes, CLIP embed, PCA, split."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data_real" / "domainnet"
SEED = 20261017
N_CLASS = 40
MAX_PER_DOMAIN_CLASS = 100
MIN_PER_DOMAIN = 30
PKT_ROWS = 40

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

URLS = {
    "clipart": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
    "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
    "painting": "https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
    "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
    "real": "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
    "sketch": "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
}
TXT_BASE = "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.name}", flush=True)
        return
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    req = urllib.request.Request(url, headers=headers)
    print(f"download {url}", flush=True)
    with urllib.request.urlopen(req, timeout=180) as resp:
        code = getattr(resp, "status", None) or resp.getcode()
        mode = "ab" if existing and code == 206 else "wb"
        if code == 200 and existing:
            part.unlink(missing_ok=True)
            mode = "wb"
        with open(part, mode) as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    part.replace(dest)
    print(f"wrote {dest} size={dest.stat().st_size}", flush=True)


def stage_download():
    raw = OUT / "raw"
    txt = OUT / "txt"
    raw.mkdir(parents=True, exist_ok=True)
    txt.mkdir(parents=True, exist_ok=True)
    print("download DomainNet from csr.bu.edu", flush=True)
    for d in DOMAINS:
        _download(URLS[d], raw / f"{d}.zip")
        for split in ("train", "test"):
            _download(f"{TXT_BASE}/{d}_{split}.txt", txt / f"{d}_{split}.txt")
    print("download done", flush=True)


def stage_extract():
    img_root = OUT / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    print("extract DomainNet zips", flush=True)
    for d in DOMAINS:
        dest = img_root / d
        if dest.exists() and any(dest.iterdir()):
            print(f"skip extract {d}", flush=True)
            continue
        zpath = OUT / "raw" / f"{d}.zip"
        print(f"extract {zpath}", flush=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(img_root)
    print("extract done", flush=True)


def _parse_list(path: Path) -> list[tuple[str, str]]:
    """Return list of (rel_path, class_name)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rel = parts[0]
        bits = Path(rel).parts
        cls = bits[-2] if len(bits) >= 2 else "unknown"
        rows.append((rel, cls))
    return rows


def stage_select():
    if (OUT / "selected_images.csv").exists() and (OUT / "selected_classes.json").exists():
        print("skip select", flush=True)
        return
    print("select DomainNet classes", flush=True)
    rng = np.random.default_rng(SEED)
    per_dom_cls: dict[str, dict[str, list[Path]]] = {d: defaultdict(list) for d in DOMAINS}
    img_root = OUT / "images"
    for d in DOMAINS:
        for split in ("train", "test"):
            for rel, cls in _parse_list(OUT / "txt" / f"{d}_{split}.txt"):
                cands = [
                    img_root / rel,
                    img_root / d / Path(rel).name,
                    img_root / Path(rel),
                ]
                if len(Path(rel).parts) >= 2:
                    cands.append(img_root / Path(*Path(rel).parts[-3:]))
                found = None
                for c in cands:
                    if c.exists():
                        found = c
                        break
                if found is None:
                    c2 = img_root / d / cls / Path(rel).name
                    if c2.exists():
                        found = c2
                if found is not None:
                    per_dom_cls[d][cls].append(found)

    all_classes = set.intersection(*[set(per_dom_cls[d].keys()) for d in DOMAINS])
    eligible = []
    for cls in all_classes:
        counts = {d: len(per_dom_cls[d][cls]) for d in DOMAINS}
        if min(counts.values()) >= MIN_PER_DOMAIN:
            total = sum(counts.values())
            eligible.append((cls, total, counts))
    eligible.sort(key=lambda x: -x[1])
    chosen = eligible[:N_CLASS]
    print(f"eligible classes={len(eligible)}; chosen top {len(chosen)}", flush=True)
    class_names = [c[0] for c in chosen]
    (OUT / "selected_classes.json").write_text(
        json.dumps(
            [dict(class_name=c, total=t, per_domain=cnt) for c, t, cnt in chosen],
            indent=2,
        )
        + "\n"
    )

    rows = []
    for d in DOMAINS:
        for ci, cls in enumerate(class_names):
            paths = list(per_dom_cls[d][cls])
            rng.shuffle(paths)
            paths = paths[:MAX_PER_DOMAIN_CLASS]
            for p in paths:
                rows.append(dict(domain=d, class_name=cls, class_id=ci, path=str(p)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "selected_images.csv", index=False)
    print(df.groupby("domain").size(), flush=True)
    print("select done", flush=True)


def stage_embed(batch_size: int = 64):
    """OpenAI CLIP ViT-L/14 embeddings."""
    key = "clip-vit-l14"
    emb_dir = OUT / f"emb_{key}"
    if (emb_dir / "embeddings.npy").exists():
        print(f"skip embed {key}", flush=True)
        return
    import torch
    from PIL import Image
    import clip

    print(f"embed {key} start", flush=True)
    df = pd.read_csv(OUT / "selected_images.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embed {key} device={device} n={len(df)}", flush=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    def encode_batch(paths):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.inference_mode():
            feats = model.encode_image(imgs).float()
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    (emb_dir / "encoder_meta.json").write_text(
        json.dumps(dict(model_id="openai/CLIP-ViT-L-14", normalize=True), indent=2) + "\n"
    )

    chunk = 512
    embs = []
    for i in range(0, len(df), chunk):
        cpath = emb_dir / f"chunk_{i:06d}.npy"
        if cpath.exists():
            e = np.load(cpath)
        else:
            paths = df["path"].iloc[i : i + chunk].tolist()
            parts = []
            for j in range(0, len(paths), batch_size):
                parts.append(encode_batch(paths[j : j + batch_size]))
            e = np.concatenate(parts, axis=0)
            np.save(cpath, e.astype(np.float32))
            print(f"  chunk {i}/{len(df)}", flush=True)
        embs.append(e)
    E = np.concatenate(embs, axis=0)
    assert len(E) == len(df)
    np.save(emb_dir / "embeddings.npy", E.astype(np.float32))
    print(f"embed done shape={E.shape}", flush=True)


def stage_features_split():
    key = "clip-vit-l14"
    if all((OUT / f"cat_{key}_{d}.npz").exists() for d in DOMAINS):
        print("skip features", flush=True)
        return
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("pca and split start", flush=True)
    df = pd.read_csv(OUT / "selected_images.csv")
    class_names = json.loads((OUT / "selected_classes.json").read_text())
    class_names = [c["class_name"] for c in class_names]
    y_id = df["class_id"].to_numpy()
    Y = np.eye(N_CLASS, dtype=float)[y_id]

    E = np.load(OUT / "emb_clip-vit-l14" / "embeddings.npy")
    pca = PCA(n_components=32, random_state=SEED)
    Z = pca.fit_transform(E)
    Zs = StandardScaler().fit_transform(Z)
    X = np.hstack([Zs, np.ones((len(Zs), 1))])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    W, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
    pred = X[te] @ W
    acc = float((pred.argmax(1) == y_id[te]).mean())
    resid = Y[tr] - X[tr] @ W
    sigma2 = float(np.mean(np.var(resid, axis=0, ddof=1)))
    print(f"[{key}] linear acc={acc:.4f} sigma2={sigma2:.6f} d={X.shape[1]}", flush=True)

    for d in DOMAINS:
        mask = (df["domain"] == d).to_numpy()
        ids = np.where(mask)[0]
        rng_d = np.random.default_rng([SEED, sum(ord(c) for c in d)])
        ids = rng_d.permutation(ids)
        n = len(ids)
        a, b = int(0.6 * n), int(0.2 * n)
        sell, query, test = ids[:a], ids[a : a + b], ids[a + b :]
        np.savez_compressed(
            OUT / f"cat_{key}_{d}.npz",
            X_sell=X[sell],
            y_sell=Y[sell],
            y_id_sell=y_id[sell],
            X_query=X[query],
            y_query=Y[query],
            y_id_query=y_id[query],
            X_test=X[test],
            y_test=Y[test],
            y_id_test=y_id[test],
        )

    enc_meta = json.loads((OUT / f"emb_{key}" / "encoder_meta.json").read_text())
    meta = dict(
        dataset="domainnet",
        version="DomainNet cleaned (ai.bu.edu/DomainNet)",
        download_urls=URLS,
        txt_base=TXT_BASE,
        domains=DOMAINS,
        n_classes=N_CLASS,
        class_names=class_names,
        demand_categories=DOMAINS,
        encoder=enc_meta,
        encoder_key=key,
        pca_dim=32,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
        linear_accuracy=acc,
        sigma2=sigma2,
        packet_rows=PKT_ROWS,
        seed=SEED,
        n_images=int(len(df)),
        d=int(X.shape[1]),
    )
    (OUT / f"domainnet_meta_{key}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("pca and split done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all_cpu",
        choices=[
            "download",
            "extract",
            "select",
            "embed_clip",
            "features",
            "pca",
            "all_cpu",
            "all",
        ],
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("download", "all_cpu", "all"):
        stage_download()
    if args.stage in ("extract", "all_cpu", "all"):
        stage_extract()
    if args.stage in ("select", "all_cpu", "all"):
        stage_select()
    if args.stage in ("embed_clip", "all"):
        stage_embed(batch_size=max(16, args.batch_size // 2))
    if args.stage in ("features", "pca", "all"):
        stage_features_split()


if __name__ == "__main__":
    main()
