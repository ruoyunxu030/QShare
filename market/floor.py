"""Append delayed entry sellers and extra floor small-purchases to a market."""

from __future__ import annotations

import numpy as np

from market.build import _inject
from market.const import PKT_ROWS, PKTS_PER_SELLER
from market.features import noise_weight
from market.packets import packet_fishers

def append_entry_seller(
    market: dict,
    rng,
    groups,
    dataset: str,
    s_rows: int,
    group: str,
    appear_at: int = 15,
    q: float = 0.0,
    n_extra_small: int = 0,
):
    """Add one delayed seller on an already-built (small5-dropped) market.

    Packets and small_s rows come from leftover unused rows of `group`.
    Does not replay the base-market RNG. Raises if leftover rows are too few
    to keep packets disjoint from small purchases and from existing sellers.
    n_extra_small: extra small_s offers for platform entry_fund. Default 0.
    """
    g = str(group)
    Xsell, ysell = groups[g]["X_sell"], groups[g]["y_sell"]
    yid_sell = groups[g].get("y_id_sell")
    n = len(Xsell)
    used = market["used"]
    if g not in used:
        used[g] = set()
    need = int(PKTS_PER_SELLER) * int(PKT_ROWS) + int(s_rows) * (1 + max(int(n_extra_small), 0))
    avail0 = [i for i in range(n) if i not in used[g]]
    if len(avail0) < need:
        raise RuntimeError(
            f"entry seller: group {g} leftover {len(avail0)} < {need} "
            f"(packets {PKTS_PER_SELLER}×{PKT_ROWS} + small_s {s_rows})"
        )
    sid = int(np.max(market["seller_ids"])) + 1
    oid0 = int(np.max(market["offer_id"])) + 1
    cs = float(s_rows) / float(PKT_ROWS)
    Xs = list(market["Xs"])
    ys = list(market["ys"])
    ys_clean = list(market["ys_clean"])
    yids = list(market["yids"])
    contams = list(market["contams"])
    src_row_idx = list(market.get("src_row_idx") or [None] * len(Xs))
    sources = list(np.asarray(market["sources"]))
    seller_ids = list(np.asarray(market["seller_ids"], int))
    qs = list(np.asarray(market["qs"], float))
    costs = list(np.asarray(market["c"], float))
    kind_arr = list(np.asarray(market["kind"]))
    appear = list(np.asarray(market["appear_at"], int))
    offer_id = list(np.asarray(market["offer_id"], int))
    pkt_now = set()
    for _ in range(PKTS_PER_SELLER):
        avail = [i for i in range(n) if i not in used[g]]
        idx = rng.choice(np.asarray(avail), size=PKT_ROWS, replace=False)
        used[g].update(int(x) for x in idx.tolist())
        pkt_now.update(int(x) for x in idx.tolist())
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(1.0); kind_arr.append("packet"); appear.append(int(appear_at))
        offer_id.append(oid0); oid0 += 1
    avail = [i for i in range(n) if i not in used[g] and i not in pkt_now]
    if len(avail) < int(s_rows):
        raise RuntimeError(
            f"entry seller: group {g} leftover after packets {len(avail)} < small {s_rows}"
        )
    idx = rng.choice(np.asarray(avail), size=int(s_rows), replace=False)
    used[g].update(int(x) for x in idx.tolist())
    Xj, yj, yidj, yc, cm = _inject(
        dataset, Xsell, ysell, yid_sell, idx, q, rng,
    )
    Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
    contams.append(cm); src_row_idx.append(np.asarray(idx, int))
    sources.append(g); seller_ids.append(sid); qs.append(float(q))
    costs.append(cs); kind_arr.append("small_s"); appear.append(int(appear_at))
    offer_id.append(oid0); oid0 += 1
    fund_oids = []
    for _ in range(max(int(n_extra_small), 0)):
        avail = [i for i in range(n) if i not in used[g]]
        if len(avail) < int(s_rows):
            raise RuntimeError(
                f"entry seller extra small: group {g} leftover {len(avail)} < {s_rows}"
            )
        idx = rng.choice(np.asarray(avail), size=int(s_rows), replace=False)
        used[g].update(int(x) for x in idx.tolist())
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(cs); kind_arr.append("small_s"); appear.append(int(appear_at))
        fund_oids.append(int(oid0))
        offer_id.append(oid0); oid0 += 1
    F = packet_fishers(Xs)
    w = np.array([noise_weight(q_) for q_ in qs], float)
    kind_arr = np.asarray(kind_arr)
    out = dict(market)
    out.update(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_row_idx,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_small=kind_arr != "packet", used=used,
        appear_at=np.asarray(appear, int),
        offer_id=np.asarray(offer_id, int),
        entry_seller_id=int(sid),
        entry_group=g,
        entry_fund_offer_ids=fund_oids,
    )
    return out

def append_floor_purchases(
    market: dict,
    rng,
    groups,
    dataset: str,
    s_rows: int,
    n_extra: int,
):
    """Add extra small_s per seller for the evidence floor. Separate RNG.

    Prefers unused leftover rows. If leftover < s, sample with replacement
    from this seller's non-packet rows. Default n_extra=0 is unused;
    call only when floor inventory is needed. Does not replay market RNG.
    """
    n_extra = max(int(n_extra), 0)
    if n_extra <= 0:
        market.setdefault("floor_offer_ids", {})
        market.setdefault("floor_reuse_n", 0)
        return market
    s_rows = int(s_rows)
    used = market.get("used") or {}
    Xs = list(market["Xs"])
    ys = list(market["ys"])
    ys_clean = list(market["ys_clean"])
    yids = list(market["yids"])
    contams = list(market["contams"])
    src_row_idx = list(market.get("src_row_idx") or [None] * len(Xs))
    sources = list(np.asarray(market["sources"]))
    seller_ids = list(np.asarray(market["seller_ids"], int))
    qs = list(np.asarray(market["qs"], float))
    costs = list(np.asarray(market["c"], float))
    kind_arr = list(np.asarray(market["kind"]))
    appear = list(np.asarray(market["appear_at"], int))
    offer_id = list(np.asarray(market["offer_id"], int))
    oid0 = int(max(offer_id)) + 1 if offer_id else 0
    cs = float(s_rows) / float(PKT_ROWS)
    first_small = {}
    extras: dict[int, list[int]] = {}
    pkt_src: dict[int, set[int]] = {}
    appear_s: dict[int, int] = {}
    src_s: dict[int, str] = {}
    q_s: dict[int, float] = {}
    for i, (s, k) in enumerate(zip(seller_ids, kind_arr)):
        s = int(s)
        appear_s[s] = int(appear[i])
        src_s[s] = str(sources[i])
        q_s[s] = float(qs[i])
        if k == "packet" and src_row_idx[i] is not None:
            pkt_src.setdefault(s, set()).update(int(x) for x in np.asarray(src_row_idx[i]).tolist())
        if k != "small_s":
            continue
        if s not in first_small:
            first_small[s] = int(offer_id[i])
        else:
            extras.setdefault(s, []).append(int(offer_id[i]))
    reuse = int(market.get("floor_reuse_n") or 0)
    for s in sorted(appear_s):
        have = list(extras.get(s, []))
        need = n_extra - len(have)
        if need <= 0:
            extras[s] = have
            continue
        g = src_s[s]
        if g not in used:
            used[g] = set()
        Xsell, ysell = groups[g]["X_sell"], groups[g]["y_sell"]
        yid_sell = groups[g].get("y_id_sell")
        n = len(Xsell)
        pkt = pkt_src.get(s, set())
        q = q_s[s]
        for _ in range(need):
            avail = [i for i in range(n) if i not in used[g] and i not in pkt]
            if len(avail) >= s_rows:
                idx = rng.choice(np.asarray(avail), size=s_rows, replace=False)
                used[g].update(int(x) for x in idx.tolist())
            else:
                pool = avail if avail else [i for i in range(n) if i not in pkt]
                if not pool:
                    pool = list(range(n))
                idx = rng.choice(np.asarray(pool), size=s_rows, replace=True)
                reuse += 1
            Xj, yj, yidj, yc, cm = _inject(
                dataset, Xsell, ysell, yid_sell, idx, q, rng,
            )
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
            contams.append(cm); src_row_idx.append(np.asarray(idx, int))
            sources.append(g); seller_ids.append(s); qs.append(q)
            costs.append(cs); kind_arr.append("small_s"); appear.append(appear_s[s])
            have.append(int(oid0))
            offer_id.append(oid0); oid0 += 1
        extras[s] = have
    F = packet_fishers(Xs)
    w = np.array([noise_weight(q_) for q_ in qs], float)
    kind_arr = np.asarray(kind_arr)
    out = dict(market)
    out.update(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_row_idx, used=used,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_small=kind_arr != "packet",
        appear_at=np.asarray(appear, int),
        offer_id=np.asarray(offer_id, int),
        floor_offer_ids={int(s): [int(x) for x in extras.get(s, [])] for s in appear_s},
        floor_reuse_n=int(reuse),
    )
    return out

