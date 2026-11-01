"""CIFAR-10N annotator market: one seller = one MTurk worker."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market.const import PKT_ROWS, PKTS_PER_SELLER
from market.features import noise_weight
from market.packets import packet_fishers

ROOT = Path(__file__).resolve().parents[1]

import os as _os
def _data_root():
    env = _os.environ.get('QSHARE_DATA')
    if env:
        return Path(env)
    return ROOT / "data_real"
CIFAR_DIR = _data_root() / "cifar10n"
SIDE_INFO = CIFAR_DIR / "side_info_cifar10N.csv"
HUMAN_PT = CIFAR_DIR / "CIFAR-10_human.pt"
DEMAND_GROUPS = {
    "vehicle": [0, 1, 8, 9],
    "pet": [3, 5],
    "wildlife": [2, 4, 6, 7],
}
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_annotations():
    """Join official side_info (worker1, worker2, worker3 id / time) with random_label1-3."""
    if not SIDE_INFO.exists():
        return None
    csv = pd.read_csv(SIDE_INFO)
    human = torch.load(HUMAN_PT, map_location="cpu", weights_only=False)
    clean = np.asarray(human["clean_label"], int)
    labs = [
        np.asarray(human["random_label1"], int),
        np.asarray(human["random_label2"], int),
        np.asarray(human["random_label3"], int),
    ]
    by_w = defaultdict(list)
    for _, r in csv.iterrows():
        a, b = str(r["Image-batch"]).split("--")
        lo, hi = int(a), int(b)
        wids = [int(r["Worker1-id"]), int(r["Worker2-id"]), int(r["Worker3-id"])]
        secs = [
            int(r["Work-time-in-seconds-1"]),
            int(r["Work-time-in-seconds-2"]),
            int(r["Work-time-in-seconds-3"]),
        ]
        for img in range(lo, hi + 1):
            for k in range(3):
                by_w[wids[k]].append((img, int(labs[k][img]), int(clean[img]), secs[k]))
    X768 = np.load(CIFAR_DIR / "emb_train.npy")
    pca = np.load(CIFAR_DIR / "pca_scaler.npz", allow_pickle=True)
    Z = (X768 - pca["mean"]) @ pca["components"].T
    Z = (Z - pca["scaler_mean"]) / np.maximum(pca["scaler_scale"], 1e-12)
    X = np.hstack([Z, np.ones((len(Z), 1))])
    y_clean = np.asarray(clean, int)
    return dict(by_worker=dict(by_w), X=X, y_clean=y_clean, n_img=len(clean))


def worker_stats(ann=None) -> dict:
    if ann is None:
        ann = load_annotations()
    if ann is None:
        return dict(
            has_side_info=False,
            side_info_file="side_info_cifar10N.csv",
            official_repo="UCSC-REAL/cifar-10-100n",
        )
    W = ann["by_worker"]
    ns, accs, times = [], [], []
    rows = []
    for w, items in W.items():
        n = len(items)
        acc = float(np.mean([lab == cl for _, lab, cl, _ in items]))
        tmean = float(np.mean([sec for *_, sec in items]))
        ns.append(n); accs.append(acc); times.append(tmean)
        conf = Counter((cl, lab) for _, lab, cl, _ in items if lab != cl)
        rows.append(dict(
            worker=int(w), n_ann=n, acc=acc, mean_sec=tmean,
            top_confusions=[(int(a), int(b), int(c)) for (a, b), c in conf.most_common(5)],
        ))
    ns, accs = np.asarray(ns), np.asarray(accs)
    def _pct(a):
        return dict(
            min=float(a.min()), p10=float(np.percentile(a, 10)),
            p50=float(np.median(a)), p90=float(np.percentile(a, 90)),
            max=float(a.max()), mean=float(a.mean()),
        )
    n200 = [r for r in rows if r["n_ann"] >= 200]
    n120 = [r for r in rows if r["n_ann"] >= 120]
    in_seg = {}
    for g, cls in DEMAND_GROUPS.items():
        cs = set(cls)
        n_ok = 0
        for items in W.values():
            if len(items) < 200:
                continue
            if sum(1 for _, _, cl, _ in items if cl in cs) >= 200:
                n_ok += 1
        in_seg[g] = dict(n200_ge200_inseg=n_ok, n200_total=len(n200))
    lo = sorted(n200, key=lambda r: r["acc"])[:5] if n200 else sorted(rows, key=lambda r: r["acc"])[:5]
    hi = sorted(n200, key=lambda r: -r["acc"])[:5] if n200 else sorted(rows, key=lambda r: -r["acc"])[:5]
    return dict(
        has_side_info=True,
        n_workers=len(W),
        n_annotations=int(sum(len(v) for v in W.values())),
        n_ann_dist=_pct(ns),
        acc_dist=_pct(accs),
        n_ge200=len(n200),
        n_ge120=len(n120),
        acc_ge200=_pct(np.array([r["acc"] for r in n200])) if n200 else {},
        in_segment=in_seg,
        lowest_ge200=lo,
        highest_ge200=hi,
        classes=CIFAR10_CLASSES,
        demand_groups=DEMAND_GROUPS,
    )


def _onehot(ids, C=10):
    ids = np.asarray(ids, int)
    y = np.zeros((len(ids), C), float)
    y[np.arange(len(ids)), ids] = 1.0
    return y


CIFAR_MIN_ANN = 120
CIFAR_PKTS = 2


def acc_quartile_cuts(ann, min_ann: int = CIFAR_MIN_ANN):
    """Global accuracy quartile cuts on workers with ≥min_ann labels. Shared by all instances."""
    accs = []
    rows = []
    for w, items in ann["by_worker"].items():
        n = len(items)
        if n < min_ann:
            continue
        acc = float(np.mean([lab == cl for _, lab, cl, _ in items]))
        accs.append(acc)
        rows.append(dict(worker=int(w), n_ann=int(n), acc=acc))
    accs = np.asarray(accs, float)
    cuts = tuple(float(x) for x in np.quantile(accs, [0.25, 0.50, 0.75]))
    return dict(
        min_ann=int(min_ann),
        n_workers=int(len(rows)),
        cuts=cuts,
        acc_min=float(accs.min()) if len(accs) else np.nan,
        acc_max=float(accs.max()) if len(accs) else np.nan,
        acc_mean=float(accs.mean()) if len(accs) else np.nan,
        n_per_bin=_count_bins(accs, cuts),
        workers=rows,
    )


def _count_bins(accs, cuts):
    bins = [0, 0, 0, 0]
    for a in accs:
        bins[_acc_bin(float(a), cuts)] += 1
    return bins


def _acc_bin(acc: float, cuts) -> int:
    if acc <= cuts[0]:
        return 0
    if acc <= cuts[1]:
        return 1
    if acc <= cuts[2]:
        return 2
    return 3


def _adjacent_bins(q: int):
    order = [q]
    for d in range(1, 4):
        if q - d >= 0:
            order.append(q - d)
        if q + d <= 3:
            order.append(q + d)
    return order


def build_stratified_worker_market(
    rng, ann, s_rows: int, cuts,
    n_per_bin: int = 2, min_ann: int = CIFAR_MIN_ANN,
    pkts=CIFAR_PKTS, pkt_rows=PKT_ROWS,
):
    """2 workers × 4 global-acc quartiles per demand segment; one worker one segment."""
    W = ann["by_worker"]
    Xall, yclean = ann["X"], ann["y_clean"]
    C = 10
    cs = float(s_rows) / float(pkt_rows)
    need_pkt = int(pkts) * int(pkt_rows)
    need_small = int(s_rows)
    workers = {}
    for w, items in W.items():
        if len(items) < min_ann:
            continue
        acc_g = float(np.mean([lab == cl for _, lab, cl, _ in items]))
        by_g = {g: [] for g in DEMAND_GROUPS}
        for img, lab, cl, _sec in items:
            for g, cls in DEMAND_GROUPS.items():
                if int(cl) in cls:
                    by_g[g].append((int(img), int(lab), int(cl)))
        workers[int(w)] = dict(acc=acc_g, n=len(items), by_g=by_g, items=items)

    used = set()
    chosen = {}
    backfill = []
    seller_qbin = {}
    worker_of = {}
    Xs, ys, ys_clean, yids, contams, src_idx = [], [], [], [], [], []
    sources, seller_ids, qs, costs, kind_arr, appear, offer_id = [], [], [], [], [], [], []
    worker_ids = []
    sid = 0
    oid = 0
    used_img = set()

    for g, names in DEMAND_GROUPS.items():
        cls = set(names)
        pick = []
        for q in range(4):
            def _eligible(qq):
                out = []
                for w, rec in workers.items():
                    if w in used:
                        continue
                    if _acc_bin(rec["acc"], cuts) != qq:
                        continue
                    if len(rec["by_g"][g]) < need_pkt:
                        continue
                    if rec["n"] < need_pkt + need_small:
                        continue
                    out.append(w)
                return sorted(out)

            pool = _eligible(q)
            took_from = q
            if len(pool) < n_per_bin:
                for qq in _adjacent_bins(q)[1:]:
                    extra = [w for w in _eligible(qq) if w not in pool]
                    if extra:
                        pool = sorted(set(pool) | set(extra))
                        took_from = qq
                    if len(pool) >= n_per_bin:
                        break
                backfill.append(dict(
                    segment=g, quartile=q, need=n_per_bin,
                    had=len(_eligible(q)), took_from=took_from,
                    n_after=len(pool),
                ))
            if len(pool) < n_per_bin:
                raise RuntimeError(
                    f"cifar strat: segment {g} Q{q} has {len(pool)} < {n_per_bin}"
                )
            sel = [int(x) for x in rng.choice(np.asarray(pool), size=n_per_bin, replace=False)]
            for w in sel:
                used.add(int(w))
                pick.append((int(w), q if _acc_bin(workers[int(w)]["acc"], cuts) == q else
                             _acc_bin(workers[int(w)]["acc"], cuts)))
        chosen[g] = [w for w, _ in pick]
        for w, q_want in pick:
            rec = workers[int(w)]
            rows = list(rec["by_g"][g])
            acc_seg = float(np.mean([lab == cl for _, lab, cl in rows]))
            qbin = _acc_bin(rec["acc"], cuts)
            order = rng.permutation(len(rows))
            rows = [rows[i] for i in order]
            pos = 0
            pkt_imgs = set()
            for _ in range(pkts):
                chunk = rows[pos:pos + pkt_rows]
                pos += pkt_rows
                imgs = np.array([t[0] for t in chunk], int)
                labs = np.array([t[1] for t in chunk], int)
                cln = np.array([t[2] for t in chunk], int)
                pkt_imgs.update(int(i) for i in imgs)
                used_img.update(int(i) for i in imgs)
                Xj = Xall[imgs]
                yj = _onehot(labs, C)
                yc = _onehot(cln, C)
                dirty = labs != cln
                Xs.append(Xj); ys.append(yj); ys_clean.append(yc)
                yids.append(labs); contams.append(dirty); src_idx.append(imgs)
                sources.append(g); seller_ids.append(sid)
                qs.append(float(1.0 - acc_seg))
                costs.append(1.0); kind_arr.append("packet")
                appear.append(1); offer_id.append(oid); worker_ids.append(int(w))
                oid += 1
            rest = [
                (img, lab, cl) for img, lab, cl, _ in rec["items"]
                if int(img) not in pkt_imgs
            ]
            if len(rest) < need_small:
                raise RuntimeError(
                    f"cifar strat worker {w} leftover {len(rest)} < small {s_rows}"
                )
            idx = rng.choice(len(rest), size=need_small, replace=False)
            chunk = [rest[i] for i in idx]
            imgs = np.array([t[0] for t in chunk], int)
            labs = np.array([t[1] for t in chunk], int)
            cln = np.array([t[2] for t in chunk], int)
            used_img.update(int(i) for i in imgs)
            Xj = Xall[imgs]
            yj = _onehot(labs, C)
            yc = _onehot(cln, C)
            dirty = labs != cln
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc)
            yids.append(labs); contams.append(dirty); src_idx.append(imgs)
            sources.append(g); seller_ids.append(sid)
            qs.append(float(1.0 - acc_seg))
            costs.append(cs); kind_arr.append("small_s")
            appear.append(1); offer_id.append(oid); worker_ids.append(int(w))
            oid += 1
            seller_qbin[sid] = int(qbin)
            worker_of[sid] = int(w)
            sid += 1

    F = packet_fishers(Xs)
    w = np.array([noise_weight(q) for q in qs], float)
    kind_arr = np.asarray(kind_arr)
    oracle_sources = {}
    for g in DEMAND_GROUPS:
        m = (kind_arr == "packet") & (np.asarray(sources) == g)
        js = np.where(m)[0]
        Xg = np.vstack([Xs[j] for j in js])
        ycg = np.concatenate([np.argmax(ys_clean[j], 1) for j in js])
        oracle_sources[g] = dict(
            X_sell=Xg, y_sell=_onehot(ycg, C), y_clean_sell=ycg,
        )
    leftover = [i for i in range(len(Xall)) if i not in used_img]
    seed_pool = dict(
        X=Xall[leftover],
        y=_onehot(yclean[leftover], C),
        y_id=yclean[leftover],
        idx=np.asarray(leftover, int),
    )
    market = dict(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_idx,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_small=kind_arr != "packet",
        appear_at=np.asarray(appear, int),
        offer_id=np.asarray(offer_id, int),
        worker_id=np.asarray(worker_ids, int),
        used={g: set() for g in DEMAND_GROUPS},
        names=list(DEMAND_GROUPS),
        s_rows=s_rows, small_cost_s=cs,
        entry_seller_id=None, entry_group=None,
        oracle_sources=oracle_sources,
        chosen_workers=chosen,
        seller_qbin=seller_qbin,
        worker_of=worker_of,
        backfill=backfill,
        min_ann=int(min_ann),
        n_sellers_per_segment=n_per_bin * 4,
        pkts=int(pkts),
        quartile_cuts=tuple(cuts),
    )
    return market, seed_pool


def append_floor_purchases_cifar(market: dict, rng, ann, s_rows: int, n_extra: int):
    """Extra small_s per worker-seller from leftover annotations. Separate RNG."""
    n_extra = max(int(n_extra), 0)
    if n_extra <= 0 or ann is None:
        market.setdefault("floor_offer_ids", {})
        market.setdefault("floor_reuse_n", 0)
        return market
    s_rows = int(s_rows)
    C = 10
    Xall, yclean = ann["X"], ann["y_clean"]
    W = ann["by_worker"]
    Xs = list(market["Xs"])
    ys = list(market["ys"])
    ys_clean = list(market["ys_clean"])
    yids = list(market["yids"])
    contams = list(market["contams"])
    src_idx = list(market.get("src_row_idx") or [])
    sources = list(np.asarray(market["sources"]))
    seller_ids = list(np.asarray(market["seller_ids"], int))
    qs = list(np.asarray(market["qs"], float))
    costs = list(np.asarray(market["c"], float))
    kind_arr = list(np.asarray(market["kind"]))
    appear = list(np.asarray(market["appear_at"], int))
    offer_id = list(np.asarray(market["offer_id"], int))
    worker_ids = list(np.asarray(market.get("worker_id"), int)) if market.get("worker_id") is not None else []
    oid0 = int(max(offer_id)) + 1 if offer_id else 0
    cs = float(s_rows) / float(PKT_ROWS)
    worker_of = {int(k): int(v) for k, v in (market.get("worker_of") or {}).items()}
    extras: dict[int, list[int]] = {}
    first = set()
    pkt_imgs: dict[int, set[int]] = {}
    appear_s, src_s, q_s = {}, {}, {}
    for i, (s, k) in enumerate(zip(seller_ids, kind_arr)):
        s = int(s)
        appear_s[s] = int(appear[i])
        src_s[s] = str(sources[i])
        q_s[s] = float(qs[i])
        if k == "packet" and src_idx[i] is not None:
            pkt_imgs.setdefault(s, set()).update(int(x) for x in np.asarray(src_idx[i]).tolist())
        if k != "small_s":
            continue
        if s not in first:
            first.add(s)
        else:
            extras.setdefault(s, []).append(int(offer_id[i]))
    reuse = int(market.get("floor_reuse_n") or 0)
    for s in sorted(appear_s):
        have = list(extras.get(s, []))
        need = n_extra - len(have)
        if need <= 0:
            extras[s] = have
            continue
        wid = worker_of.get(s)
        items = W.get(int(wid), []) if wid is not None else []
        pkt = pkt_imgs.get(s, set())
        rest = [(int(img), int(lab), int(cl)) for img, lab, cl, *_ in items if int(img) not in pkt]
        for _ in range(need):
            if len(rest) >= s_rows:
                pick = rng.choice(len(rest), size=s_rows, replace=False)
            else:
                pool = rest if rest else [(int(img), int(lab), int(cl)) for img, lab, cl, *_ in items]
                if not pool:
                    pool = [(0, 0, 0)]
                pick = rng.choice(len(pool), size=s_rows, replace=True)
                reuse += 1
                chunk = [pool[int(i)] for i in pick]
                imgs = np.array([t[0] for t in chunk], int)
                labs = np.array([t[1] for t in chunk], int)
                cln = np.array([t[2] for t in chunk], int)
                Xj = Xall[imgs]
                yj = _onehot(labs, C)
                yc = _onehot(cln, C)
                dirty = labs != cln
                Xs.append(Xj); ys.append(yj); ys_clean.append(yc)
                yids.append(labs); contams.append(dirty); src_idx.append(imgs)
                sources.append(src_s[s]); seller_ids.append(s); qs.append(q_s[s])
                costs.append(cs); kind_arr.append("small_s"); appear.append(appear_s[s])
                have.append(int(oid0)); offer_id.append(oid0)
                if worker_ids:
                    worker_ids.append(int(wid) if wid is not None else -1)
                oid0 += 1
                continue
            chunk = [rest[int(i)] for i in pick]
            imgs = np.array([t[0] for t in chunk], int)
            labs = np.array([t[1] for t in chunk], int)
            cln = np.array([t[2] for t in chunk], int)
            Xj = Xall[imgs]
            yj = _onehot(labs, C)
            yc = _onehot(cln, C)
            dirty = labs != cln
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc)
            yids.append(labs); contams.append(dirty); src_idx.append(imgs)
            sources.append(src_s[s]); seller_ids.append(s); qs.append(q_s[s])
            costs.append(cs); kind_arr.append("small_s"); appear.append(appear_s[s])
            have.append(int(oid0)); offer_id.append(oid0)
            if worker_ids:
                worker_ids.append(int(wid) if wid is not None else -1)
            oid0 += 1
        extras[s] = have
    F = packet_fishers(Xs)
    w = np.array([noise_weight(q) for q in qs], float)
    kind_arr = np.asarray(kind_arr)
    out = dict(market)
    out.update(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_idx,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_small=kind_arr != "packet",
        appear_at=np.asarray(appear, int),
        offer_id=np.asarray(offer_id, int),
        floor_offer_ids={int(s): [int(x) for x in extras.get(s, [])] for s in appear_s},
        floor_reuse_n=int(reuse),
    )
    if worker_ids:
        out["worker_id"] = np.asarray(worker_ids, int)
    return out


def leftover_rows(groups, s_rows: int, n_sellers: int = 8, extra_entry_rows: int = 0,
                  extra_floor_rows: int = 0) -> pd.DataFrame:
    """Rows remaining after an 8-seller base market (packets + small5 + small_s)."""
    s1 = 5
    need_base = n_sellers * (PKTS_PER_SELLER * PKT_ROWS + s1 + int(s_rows))
    need_entry = PKTS_PER_SELLER * PKT_ROWS + int(s_rows) + int(extra_entry_rows)
    rows = []
    for g, dat in groups.items():
        n = int(len(dat["X_sell"]))
        rem = n - need_base
        rows.append(dict(
            group=g, n_sell=n, used_base=need_base, leftover=rem,
            extra_floor_rows=int(extra_floor_rows),
            leftover_after_floor=int(rem - extra_floor_rows),
            need_entry=need_entry, enough=bool(rem >= need_entry),
        ))
    return pd.DataFrame(rows)
