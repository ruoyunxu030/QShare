"""Build a frozen packet market (packets + small purchases) from group pools."""

from __future__ import annotations

import numpy as np

from market.const import N_SELLERS, PKT_ROWS, PKTS_PER_SELLER, Q_LEVELS, S1_FIXED
from market.features import _risk, noise_weight
from market.packets import packet_fishers


def _inject(dataset, Xsell, ysell, yid_sell, idx, q, rng):
    """Return (Xj, y_noisy, yid_noisy, y_clean, contam_mask).

    contam_mask[i]=True iff the injection Bernoulli flip fired for row i
    (may still equal clean label by chance after redraw).
    Uniform redraw. Use rng.integers so the market draw stays reproducible.
    """
    Xj = Xsell[idx]
    y_clean = np.asarray(ysell[idx], float).copy()
    n = len(idx)
    if q <= 0:
        contam = np.zeros(n, bool)
        if dataset == "amazon":
            return Xj, y_clean.copy(), None, y_clean, contam
        return Xj, y_clean.copy(), np.asarray(yid_sell[idx]).copy(), y_clean, contam
    contam = rng.random(n) < q
    if dataset == "amazon":
        yj = y_clean.copy()
        n_flip = int(contam.sum())
        if n_flip:
            yj[contam] = rng.integers(1, 6, size=n_flip).astype(float)
        return Xj, yj, None, y_clean, contam
    yj = y_clean.copy()
    yidj = np.asarray(yid_sell[idx]).copy()
    C = yj.shape[1]
    n_flip = int(contam.sum())
    if n_flip:
        new_id = rng.integers(0, C, size=n_flip)
        yidj[contam] = new_id
        yj[contam] = 0.0
        yj[contam, new_id] = 1.0
    return Xj, yj, yidj, y_clean, contam

def build_market_adaptive(rng, groups, dataset, s_rows: int,
                          n_sellers: int | None = None):
    """Packets (cost 1) + small5 (cost 5/40) + small_s (cost s/40), disjoint.

    n_sellers=None means N_SELLERS (8) per segment. small5 is emitted so the
    RNG stream matches, then dropped by build_market.
    """
    names = list(groups.keys())
    Xs, ys, ys_clean, yids, contams = [], [], [], [], []
    src_row_idx = []
    sources, seller_ids, qs, costs = [], [], [], []
    kind_arr = []  # 'packet' | 'small5' | 'small_s'
    appear_at = []
    used = {g: set() for g in names}
    sid = 0
    c5 = float(S1_FIXED) / float(PKT_ROWS)
    cs = float(s_rows) / float(PKT_ROWS)
    n_per = int(N_SELLERS if n_sellers is None else n_sellers)

    def emit_seller(g, q, appear_buyer):
        nonlocal sid
        Xsell, ysell = groups[g]["X_sell"], groups[g]["y_sell"]
        yid_sell = groups[g].get("y_id_sell")
        n = len(Xsell)
        seller_rows = []  # track source-row ids for this seller (disjointness)
        for _ in range(PKTS_PER_SELLER):
            avail = [i for i in range(n) if i not in used[g]]
            if len(avail) < PKT_ROWS:
                pool = avail if avail else list(range(n))
                idx = rng.choice(np.asarray(pool), size=PKT_ROWS, replace=True)
            else:
                idx = rng.choice(np.asarray(avail), size=PKT_ROWS, replace=False)
                used[g].update(idx.tolist())
            seller_rows.append(np.asarray(idx, int))
            Xj, yj, yidj, yc, cm = _inject(
                dataset, Xsell, ysell, yid_sell, idx, q, rng,
            )
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
            contams.append(cm); src_row_idx.append(np.asarray(idx, int))
            sources.append(g); seller_ids.append(sid); qs.append(float(q))
            costs.append(1.0); kind_arr.append("packet"); appear_at.append(int(appear_buyer))
        pkt_now = set()
        for r in seller_rows[:PKTS_PER_SELLER]:
            pkt_now.update(int(x) for x in r.tolist())

        def _draw_small_idx(k):
            # Keep packet vs small-purchase disjoint for this seller.
            # pkt_now ⊆ used on the default 8-seller path (packets mark used).
            avail = [i for i in range(n) if i not in used[g] and i not in pkt_now]
            if len(avail) >= k:
                idx = rng.choice(np.asarray(avail), size=k, replace=False)
                used[g].update(idx.tolist())
                return np.asarray(idx, int)
            disjoint = [i for i in range(n) if i not in pkt_now]
            pool = avail if avail else (disjoint if disjoint else list(range(n)))
            return rng.choice(np.asarray(pool), size=k, replace=True)

        # small5
        idx = _draw_small_idx(S1_FIXED)
        seller_rows.append(np.asarray(idx, int))
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(c5); kind_arr.append("small5"); appear_at.append(int(appear_buyer))
        # small_s
        idx = _draw_small_idx(s_rows)
        seller_rows.append(np.asarray(idx, int))
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(cs); kind_arr.append("small_s"); appear_at.append(int(appear_buyer))
        pkt_set = set()
        for r in seller_rows[:PKTS_PER_SELLER]:
            pkt_set.update(int(x) for x in r.tolist())
        small_set = set(int(x) for x in seller_rows[PKTS_PER_SELLER].tolist()) | set(
            int(x) for x in seller_rows[PKTS_PER_SELLER + 1].tolist()
        )
        inter = pkt_set & small_set
        if inter:
            raise RuntimeError(
                f"row overlap within seller {sid} source={g}: "
                f"{len(inter)} rows shared between packets and purchased samples"
            )
        sid += 1

    for g in names:
        q_assign = list(Q_LEVELS) * (n_per // len(Q_LEVELS))
        rng.shuffle(q_assign)
        for q in q_assign:
            emit_seller(g, q, 1)

    F = packet_fishers(Xs)
    w = np.array([noise_weight(q) for q in qs], float)
    kind_arr = np.asarray(kind_arr)
    is_small = kind_arr != "packet"
    return dict(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_row_idx,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_small=is_small, used=used, names=names,
        s_rows=s_rows, small_cost_s=cs, small_cost_5=c5,
        appear_at=np.asarray(appear_at, int),
        offer_id=np.arange(len(costs), dtype=int),
        entry_seller_id=None,
        n_sellers_per_segment=n_per,
    )

def build_market(rng, groups, dataset, s_rows: int,
                 n_sellers: int | None = None):
    """Reuse adaptive builder then drop small5 offers (only small_s is kept)."""
    m = build_market_adaptive(
        rng, groups, dataset, s_rows, n_sellers=n_sellers,
    )
    keep = m["kind"] != "small5"
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in m.items()}
    # list fields
    for k in ("Xs", "ys", "ys_clean", "contams", "yids", "src_row_idx"):
        if k in m:
            out[k] = [m[k][j] for j in np.where(keep)[0]]
    out["kind"] = m["kind"][keep]
    out["is_small"] = out["kind"] != "packet"
    out["F"] = m["F"][keep]
    out["F_qual"] = m["F_qual"][keep]
    out["c"] = m["c"][keep]
    out["seller_ids"] = m["seller_ids"][keep]
    out["sources"] = m["sources"][keep]
    out["qs"] = m["qs"][keep]
    out["s_rows"] = s_rows
    out["small_cost_s"] = float(s_rows) / float(PKT_ROWS)
    return out

def idx_of(seller_ids, kind_arr, s, kind):
    for j in range(len(seller_ids)):
        if seller_ids[j] == s and kind_arr[j] == kind:
            return j
    return -1

def rank_packets(F, M0, Sigma, sigma2, seller_ids, kind_arr):
    pk = kind_arr == "packet"
    try:
        loss0 = _risk(Sigma, M0, sigma2)
    except np.linalg.LinAlgError:
        loss0 = 1e9
    best = {}
    for j in np.where(pk)[0]:
        s = int(seller_ids[j])
        try:
            gain = loss0 - _risk(Sigma, M0 + F[j], sigma2)
        except np.linalg.LinAlgError:
            gain = 0.0
        if s not in best or gain > best[s]:
            best[s] = gain
    return sorted(best.items(), key=lambda kv: -kv[1])

