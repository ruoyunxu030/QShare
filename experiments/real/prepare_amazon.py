#!/usr/bin/env python3
"""Amazon Reviews 2023 to packet-ready groups: download, sample, embed, PCA, split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_real" / "amazon"
RAW = OUT / "raw"
SAMPLES = OUT / "samples"
SEED = 20261017
N_SAMPLE = 20_000
REPO = "McAuley-Lab/Amazon-Reviews-2023"
MIN_TEXT = 20

ENCODERS = {
    "bge-base-en-v1.5": {
        "hf_id": "BAAI/bge-base-en-v1.5",
        "normalize": True,
        "role": "primary",
    },
}


def load_selected() -> list[dict]:
    path = OUT / "selected_categories.json"
    if not path.exists():
        alt = ROOT / "data" / "amazon_selected_categories.json"
        if alt.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt, path)
        else:
            raise FileNotFoundError(path)
    return json.loads(path.read_text())


def stage_download():
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    selected = load_selected()
    print("download Amazon Reviews 2023 from Hugging Face", flush=True)
    for cat in selected:
        rel = cat["path"]
        dest = RAW / Path(rel).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {dest.name}", flush=True)
            continue
        print(f"download {rel}", flush=True)
        path = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=rel,
            local_dir=str(OUT / "_hf_cache"),
        )
        shutil.copy2(path, dest)
        print(f"wrote {dest} size={dest.stat().st_size}", flush=True)
    print("download done", flush=True)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stage_preprocess():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    selected = load_selected()
    rng = np.random.default_rng(SEED)
    stats = []
    print("preprocess Amazon reviews", flush=True)
    for cat in selected:
        name = cat["name"]
        raw = RAW / f"{name}.jsonl"
        out_pq = SAMPLES / f"{name}.pkl"
        old_parq = SAMPLES / f"{name}.parquet"
        if out_pq.exists():
            print(f"skip sample {name}", flush=True)
            st = json.loads((SAMPLES / f"{name}_stats.json").read_text())
            stats.append(st)
            continue
        print(f"preprocess {name}", flush=True)
        rows = []
        n_raw = 0
        for obj in _iter_jsonl(raw):
            n_raw += 1
            text = (obj.get("text") or "").strip()
            if len(text) < MIN_TEXT:
                continue
            title = (obj.get("title") or "").strip()
            user = obj.get("user_id") or obj.get("reviewerID") or ""
            asin = obj.get("parent_asin") or obj.get("asin") or ""
            rating = obj.get("rating")
            if rating is None:
                rating = obj.get("overall")
            if rating is None:
                continue
            rows.append(
                dict(
                    user_id=str(user),
                    parent_asin=str(asin),
                    text=text,
                    title=title,
                    rating=float(rating),
                    input_text=(title + "\n" + text) if title else text,
                )
            )
        df = pd.DataFrame(rows)
        n_filt = len(df)
        if n_filt:
            df = df.drop_duplicates(subset=["user_id", "parent_asin", "text"])
        n_dedup = len(df)
        if n_dedup > N_SAMPLE:
            idx = rng.choice(n_dedup, size=N_SAMPLE, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        df.to_pickle(out_pq)
        if old_parq.exists():
            old_parq.unlink(missing_ok=True)
        vc = df["rating"].value_counts().sort_index().to_dict()
        st = dict(
            category=name,
            n_raw=n_raw,
            n_after_filter=n_filt,
            n_after_dedup=n_dedup,
            n_sampled=len(df),
            rating_dist={str(k): int(v) for k, v in vc.items()},
            rating_mean=float(df["rating"].mean()) if len(df) else None,
        )
        (SAMPLES / f"{name}_stats.json").write_text(json.dumps(st, indent=2) + "\n")
        stats.append(st)
        print(
            f"  raw={n_raw} filt={n_filt} dedup={n_dedup} sample={len(df)} "
            f"rating_mean={st['rating_mean']}",
            flush=True,
        )
    (OUT / "preprocess_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("preprocess done", flush=True)


def _model_dir(key: str) -> Path:
    d = OUT / f"emb_{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _embed_category(model, key: str, name: str, texts: list[str], batch_size: int, device: str):
    """Chunked embed with resume; each chunk is 2048 rows."""
    mdir = _model_dir(key) / name
    mdir.mkdir(parents=True, exist_ok=True)
    chunk = 2048
    n = len(texts)
    n_chunks = (n + chunk - 1) // chunk
    paths = []
    for ci in range(n_chunks):
        cpath = mdir / f"chunk_{ci:04d}.npy"
        paths.append(cpath)
        if cpath.exists():
            continue
        sl = texts[ci * chunk : (ci + 1) * chunk]
        emb = model.encode(
            sl,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=ENCODERS[key]["normalize"],
        )
        np.save(cpath, emb.astype(np.float32))
        print(f"  [{key}/{name}] chunk {ci+1}/{n_chunks}", flush=True)
    out = mdir / "embeddings.npy"
    if not out.exists():
        arr = np.concatenate([np.load(p) for p in paths], axis=0)
        assert arr.shape[0] == n
        np.save(out, arr)
    return out


def stage_embed(encoder_keys: list[str], batch_size: int = 64):
    selected = load_selected()
    if all(
        (_model_dir(k) / cat["name"] / "embeddings.npy").exists()
        for k in encoder_keys
        for cat in selected
    ):
        print("skip embed", flush=True)
        return
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embed device={device} cuda_count={torch.cuda.device_count()}", flush=True)
    rev_info = {}
    for key in encoder_keys:
        cfg = ENCODERS[key]
        print(f"encoder {key} ({cfg['hf_id']})", flush=True)
        rev_info[key] = {"hf_id": cfg["hf_id"], "normalize": cfg["normalize"]}
        model = SentenceTransformer(cfg["hf_id"], device=device)
        for cat in selected:
            name = cat["name"]
            df = pd.read_pickle(SAMPLES / f"{name}.pkl")
            texts = df["input_text"].astype(str).tolist()
            _embed_category(model, key, name, texts, batch_size, device)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    (OUT / "encoder_revisions.json").write_text(json.dumps(rev_info, indent=2) + "\n")
    print("embed done", flush=True)


def stage_pca(primary_key: str = "bge-base-en-v1.5"):
    selected = load_selected()
    enc_out = OUT / f"features_{primary_key}"
    if all((enc_out / f"{cat['name']}_X.npy").exists() for cat in selected):
        print("skip pca", flush=True)
        return
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("pca start", flush=True)
    embs, ys, cats = [], [], []
    for cat in selected:
        name = cat["name"]
        e = np.load(_model_dir(primary_key) / name / "embeddings.npy")
        df = pd.read_pickle(SAMPLES / f"{name}.pkl")
        assert len(df) == len(e)
        embs.append(e)
        ys.append(df["rating"].to_numpy(np.float64))
        cats.append(np.array([name] * len(df)))
    X_raw = np.vstack(embs)
    y = np.concatenate(ys)
    d_in = X_raw.shape[1]
    print(f"PCA fit on {X_raw.shape} primary={primary_key}", flush=True)
    pca = PCA(n_components=32, random_state=SEED)
    Z = pca.fit_transform(X_raw)
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)
    X = np.hstack([Zs, np.ones((len(Zs), 1))])
    pca_info = dict(
        primary_encoder=primary_key,
        dim_before_pca=int(d_in),
        dim_after_pca=32,
        dim_with_bias=33,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
    )
    (OUT / "pca.json").write_text(json.dumps(pca_info, indent=2) + "\n")
    for key in ENCODERS:
        embs_k = []
        for cat in selected:
            embs_k.append(np.load(_model_dir(key) / cat["name"] / "embeddings.npy"))
        Xk = np.vstack(embs_k)
        pca_k = PCA(n_components=32, random_state=SEED)
        Zk = pca_k.fit_transform(Xk)
        sk = StandardScaler()
        Zks = sk.fit_transform(Zk)
        Xk_full = np.hstack([Zks, np.ones((len(Zks), 1))])
        offset = 0
        enc_k = OUT / f"features_{key}"
        enc_k.mkdir(parents=True, exist_ok=True)
        for cat in selected:
            name = cat["name"]
            n = len(pd.read_pickle(SAMPLES / f"{name}.pkl"))
            np.savez_compressed(
                enc_k / f"{name}_X.npy.npz",
                X=Xk_full[offset : offset + n],
                y=y[offset : offset + n],
            )
            np.save(enc_k / f"{name}_X.npy", Xk_full[offset : offset + n])
            np.save(enc_k / f"{name}_y.npy", y[offset : offset + n])
            offset += n
        meta_k = dict(
            encoder=key,
            dim_before_pca=int(Xk.shape[1]),
            dim_after_pca=32,
            dim_with_bias=33,
            explained_variance_ratio=pca_k.explained_variance_ratio_.tolist(),
            explained_variance_ratio_sum=float(pca_k.explained_variance_ratio_.sum()),
        )
        (enc_k / "pca.json").write_text(json.dumps(meta_k, indent=2) + "\n")
        np.savez_compressed(
            enc_k / "pca_scaler.npz",
            components=pca_k.components_,
            mean=pca_k.mean_,
            scaler_mean=sk.mean_,
            scaler_scale=sk.scale_,
        )
    print("pca done", flush=True)


def _split_pools(n: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    n_sell = int(0.6 * n)
    n_query = int(0.2 * n)
    return idx[:n_sell], idx[n_sell : n_sell + n_query], idx[n_sell + n_query :]


def stage_split(primary_key: str = "bge-base-en-v1.5"):
    selected = load_selected()
    if all((OUT / f"cat_{primary_key}_{cat['name']}.npz").exists() for cat in selected):
        print("skip split", flush=True)
        return
    print("split start", flush=True)
    rev_path = OUT / "encoder_revisions.json"
    if rev_path.exists():
        rev = json.loads(rev_path.read_text())
    else:
        rev = {k: {"hf_id": v["hf_id"], "normalize": v["normalize"]} for k, v in ENCODERS.items()}
    pca = json.loads((OUT / "pca.json").read_text())
    per = {}
    rng_base = {}
    for cat in selected:
        name = cat["name"]
        n = len(np.load(OUT / f"features_{primary_key}" / f"{name}_X.npy"))
        rng_c = np.random.default_rng([SEED, sum(ord(c) for c in name)])
        rng_base[name] = _split_pools(n, rng_c)

    for enc_key in ENCODERS:
        for cat in selected:
            name = cat["name"]
            X = np.load(OUT / f"features_{enc_key}" / f"{name}_X.npy")
            y = np.load(OUT / f"features_{enc_key}" / f"{name}_y.npy")
            sell, query, test = rng_base[name]
            np.savez_compressed(
                OUT / f"cat_{enc_key}_{name}.npz",
                X_sell=X[sell],
                y_sell=y[sell],
                X_query=X[query],
                y_query=y[query],
                X_test=X[test],
                y_test=y[test],
                category=np.array(name),
            )
            if enc_key == primary_key:
                per[name] = dict(
                    n=int(len(X)),
                    n_sell=int(len(sell)),
                    n_query=int(len(query)),
                    n_test=int(len(test)),
                )

    meta = dict(
        dataset="amazon_reviews_2023",
        status="ready",
        d=int(pca["dim_with_bias"]),
        pca_dim=32,
        packet_rows=40,
        n_sample_per_category=N_SAMPLE,
        seed=SEED,
        demand_categories=[c["name"] for c in selected],
        selected=selected,
        per_category=per,
        encoders=rev,
        pca=pca,
        primary_encoder=primary_key,
        split="6:2:2 sell:query:test",
    )
    (OUT / "amazon_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("split done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=["download", "preprocess", "embed", "pca", "split", "all", "all_cpu"],
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("download", "all", "all_cpu"):
        stage_download()
    if args.stage in ("preprocess", "all", "all_cpu"):
        stage_preprocess()
    if args.stage in ("embed", "all"):
        stage_embed(list(ENCODERS.keys()), batch_size=args.batch_size)
    if args.stage in ("pca", "all"):
        stage_pca()
    if args.stage in ("split", "all"):
        stage_split()


if __name__ == "__main__":
    main()
