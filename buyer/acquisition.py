"""One buyer on a frozen market: task/seed from a derived RNG, then a policy."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from market.const import PKT_ROWS
from market.data import draw_seed_cifar, second_moment
from market.features import (
    cifar_oracle_train_pool,
    draw_seed,
    fit_eval,
    mixed_test,
    oracle_mse,
    task_kind,
    weighted_loss,
)
from market.purchase import strategy_p_variant
from shared_evidence.evidence import BuyerEvidenceStore, EvidenceStore

EPS_EXPLORE = 0.05
SHARE_POLICIES = ("indep", "share_naive", "share_ts")
AGG_POLICIES = ("share_agg", "share_agg_ts", "share_agg_ucb")
FUND_POLICIES = ("fund80_agg",)
FUND_BASE = {"fund80_agg": "share_agg"}
FUND_L = {"fund80_agg": 80}
FLOOR_POLICIES = ("floor40_row", "floor80_row", "floor40_agg", "floor80_agg")
FLOOR_BASE = {
    "floor40_row": "share_naive", "floor80_row": "share_naive",
    "floor40_agg": "share_agg", "floor80_agg": "share_agg",
}
FLOOR_F = {"floor40_row": 40, "floor80_row": 80, "floor40_agg": 40, "floor80_agg": 80}
PAPER_POLICIES_BASE = (
    "indep", "share_naive", "share_ts", "share_agg", "share_agg_ts",
    "share_agg_ucb",
    "floor40_row", "floor80_row", "floor40_agg", "floor80_agg",
)
PAPER_POLICIES_ENTRY = PAPER_POLICIES_BASE + ("fund80_agg",)
# CIFAR-10N was not run with UCB; keep the committed policy set.
PAPER_POLICIES_CIFAR = tuple(p for p in PAPER_POLICIES_BASE if p != "share_agg_ucb")
# Keep policy IDs fixed: they seed each buyer's RNG.
POLICY_ID = {
    "indep": 0, "share_naive": 1, "share_ts": 3,
    "share_agg": 9, "share_agg_ts": 11,
    "share_agg_ucb": 20,
    "fund80_agg": 15,
    "floor40_row": 16, "floor80_row": 17, "floor40_agg": 18, "floor80_agg": 19,
}


def slice_market(market: dict, keep: np.ndarray) -> dict:
    keep = np.asarray(keep, bool)
    n = len(market["c"])
    out = {}
    for k, v in market.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[j] for j in np.where(keep)[0]]
        else:
            out[k] = v
    out["is_small"] = out["kind"] != "packet"
    return out


def visible_market(market: dict, buyer_t: int) -> dict:
    appear = market.get("appear_at")
    if appear is None:
        return market
    keep = np.asarray(appear, int) <= int(buyer_t)
    if bool(np.all(keep)):
        return market
    return slice_market(market, keep)


def draw_buyer_task(buyer_rng, groups, market, n_seed, d, lam, K,
                   cifar=False, seed_pool=None):
    names = list(groups.keys())
    if cifar:
        demand = list(names)[: int(min(K, len(names)))]
        M0, X0, y0 = draw_seed_cifar(buyer_rng, seed_pool, n_seed, d, lam)
    else:
        demand = list(buyer_rng.choice(names, size=min(K, len(names)), replace=False))
        M0, X0, y0 = draw_seed(buyer_rng, groups, market["used"], n_seed, d, lam)
    Sigmas = [second_moment(groups[g]["X_query"]) for g in demand]
    Sigma_pre = sum(Sigmas) / len(Sigmas)
    X_mix, y_mix, yid_mix = mixed_test(groups, demand)
    return demand, Sigmas, Sigma_pre, M0, X0, y0, X_mix, y_mix, yid_mix


def _row_array(row_w, j, n):
    if row_w is None:
        return np.ones(n, float)
    w = row_w.get(j, row_w.get(int(j)))
    if w is None:
        return np.ones(n, float)
    w = np.asarray(w, float)
    if w.ndim == 0:
        return np.full(n, float(w))
    return w


def _naive_share_caps(store, vis_sids, s_rows, B, beta, small_cost):
    vis_sids = [int(s) for s in vis_sids]
    n_vis = max(len(vis_sids), 1)
    n_distinct = {s: store.n_distinct(s) for s in vis_sids}
    n_insuf = sum(1 for s in vis_sids if n_distinct[s] < int(s_rows))
    if n_insuf == 0:
        n_small = 0
    else:
        n_small = max(
            1,
            int(np.floor(beta * B * n_insuf / n_vis / max(small_cost, 1e-9))),
        )
    skip = {s for s in vis_sids if n_distinct[s] >= int(s_rows)}
    return n_small, skip, n_distinct


def _pi_from_store(store, vis_sids, policy, rng):
    pi_override = {}
    post = {}
    for s in vis_sids:
        a, b, mean, n = store.seller_posterior(s)
        post[int(s)] = (float(a), float(b), float(mean), int(n))
        if n <= 0:
            continue
        if policy == "share_ts":
            pi_override[s] = float(np.clip(rng.beta(a, b), 1e-3, 1.0 - 1e-3))
        else:
            pi_override[s] = float(np.clip(mean, 1e-3, 1.0 - 1e-3))
    return pi_override, post


def _cal_share_caps(bstore: BuyerEvidenceStore, vis_sids, B, beta, small_cost):
    vis_sids = [int(s) for s in vis_sids]
    n_vis = max(len(vis_sids), 1)
    n_map = {s: bstore.n_buyers(s) for s in vis_sids}
    n_insuf = sum(1 for s in vis_sids if n_map[s] < 1)
    if n_insuf == 0:
        n_small = 0
    else:
        n_small = max(
            1,
            int(np.floor(beta * B * n_insuf / n_vis / max(small_cost, 1e-9))),
        )
    skip = {s for s in vis_sids if n_map[s] >= 1}
    return n_small, skip


def sample_ts_plan_buyer(bstore: BuyerEvidenceStore, vis_sids, rng) -> dict[int, float]:
    out = {}
    for s in vis_sids:
        m, v, _n, *_ = bstore.publish(int(s))
        v = max(float(v), 1e-4)
        out[int(s)] = float(np.clip(rng.normal(m, v), 0.05, 1.0))
    return out


def ucb_plan_buyer(bstore: BuyerEvidenceStore, vis_sids) -> dict[int, float]:
    """Same published (mean, sd) as the QShare lower bound, with the sign flipped."""
    from shared_evidence.lemma42 import Z_LB
    out = {}
    for s in vis_sids:
        m, v, _n, *_ = bstore.publish(int(s))
        out[int(s)] = float(np.clip(float(m) + Z_LB * float(v), 0.05, 1.0))
    return out


def buyer_pi_hats(gamma_by_row, vis) -> dict[int, tuple[float, int]]:
    """π̂_jb = mean γ on rows this buyer bought from j."""
    offer_id = np.asarray(vis["offer_id"], int)
    seller_ids = np.asarray(vis["seller_ids"], int)
    loc = {int(oid): i for i, oid in enumerate(offer_id)}
    by_s: dict[int, list[float]] = {}
    for rid, g in gamma_by_row.items():
        oid = int(rid[0])
        j = loc.get(oid)
        if j is None:
            continue
        s = int(seller_ids[j])
        by_s.setdefault(s, []).append(float(g))
    return {s: (float(np.mean(vs)), int(len(vs))) for s, vs in by_s.items()}


def export_record_tables(
    vis,
    gamma_by_row,
    demand,
    dataset,
    market_kind,
    inst,
    policy,
    buyer_t,
    funded_oids,
):
    """Logging only. No RNG, no store writes, no decision change."""
    from market.contam import obs_dirty_mask

    offer_id = np.asarray(vis["offer_id"], int)
    seller_ids = np.asarray(vis["seller_ids"], int)
    sources = np.asarray(vis["sources"])
    qs = np.asarray(vis["qs"], float)
    loc = {int(oid): i for i, oid in enumerate(offer_id)}
    ys, ys_clean, contams = vis["ys"], vis.get("ys_clean"), vis.get("contams")
    src_row_idx = vis.get("src_row_idx") or [None] * len(offer_id)
    yids = vis.get("yids") or [None] * len(offer_id)
    funded = {int(o) for o in funded_oids}
    demand_set = {str(z) for z in demand}
    cifar = str(dataset) == "cifar10n"

    by_s: dict[int, list[tuple]] = {}
    row_out = []
    for rid, g in gamma_by_row.items():
        oid, li = int(rid[0]), int(rid[1])
        j = loc.get(oid)
        if j is None:
            continue
        dirty = np.nan
        if ys_clean is not None and ys_clean[j] is not None:
            mask = obs_dirty_mask(ys[j], ys_clean[j])
            if 0 <= li < len(mask):
                dirty = bool(mask[li])
        elif contams is not None and contams[j] is not None and 0 <= li < len(contams[j]):
            dirty = bool(contams[j][li])
        src = np.nan
        if cifar and yids[j] is not None:
            yid = np.asarray(yids[j])
            if 0 <= li < len(yid):
                src = int(yid[li])
        elif src_row_idx[j] is not None:
            arr = np.asarray(src_row_idx[j])
            if 0 <= li < len(arr):
                src = int(arr[li])
        s = int(seller_ids[j])
        is_funded = int(oid in funded)
        row_out.append(dict(
            record_key=f"{dataset}|{market_kind}|{int(inst)}|{policy}|{int(buyer_t)}|{s}",
            offer_id=oid, local_i=li, source_row_id=src,
            funded=is_funded, gamma=float(g), contaminated=dirty,
        ))
        by_s.setdefault(s, []).append((float(g), dirty, is_funded, float(1.0 - qs[j]), str(sources[j])))

    recs = []
    for s, xs in by_s.items():
        gs = np.asarray([x[0] for x in xs], float)
        dirty = np.asarray([x[1] for x in xs], float)
        fund = np.asarray([x[2] for x in xs], int)
        pi = float(xs[0][3])
        seg = str(xs[0][4])
        ok = np.isfinite(dirty)
        qjb = float(1.0 - np.mean(dirty[ok])) if ok.any() else np.nan
        gbar = float(np.mean(gs))
        recs.append(dict(
            dataset=dataset, market=market_kind, instance=int(inst),
            policy=policy, buyer_t=int(buyer_t), seller=int(s),
            segment=seg, true_pi=pi,
            in_task=int(seg in demand_set),
            n_rows=int(len(xs)),
            n_funded_rows=int(fund.sum()),
            gamma_bar=gbar, q_jb=qjb,
        ))
    return recs, row_out


def means_from_buyer_store(bstore: BuyerEvidenceStore, vis_sids) -> dict[int, float]:
    out = {}
    for s in vis_sids:
        m, _v, n, *_ = bstore.publish(int(s))
        if n <= 0:
            continue
        out[int(s)] = float(np.clip(m, 0.05, 1.0))
    return out


def _share_kwargs(policy, store, vis_sids, s_rows, B, beta, small_cost, rng,
                  vis=None, demand=None, M0=None,
                  Sigma_pre=None, Sigmas=None, rho=None, sigma2=None, z_idx=0,
                  plan_pi=None, buyer_store=None):
    policy = FUND_BASE.get(policy, policy)
    policy = FLOOR_BASE.get(policy, policy)
    if policy == "indep" or (store is None and buyer_store is None):
        return {}
    vis_sids = [int(s) for s in vis_sids]
    extra_sellers, extra_budget = None, 0.0
    if policy in AGG_POLICIES and buyer_store is not None:
        n_small, skip = _cal_share_caps(buyer_store, vis_sids, B, beta, small_cost)
        pi_override = means_from_buyer_store(buyer_store, vis_sids)
        kw = dict(
            n_small_override=n_small,
            skip_small_sellers=skip,
            pi_override=pi_override,
            extra_small_sellers=extra_sellers,
            extra_small_budget=extra_budget,
            allow_sellers=set(vis_sids),
        )
        if plan_pi is not None:
            kw["pi_plan"] = {int(s): float(np.clip(p, 0.05, 1.0)) for s, p in plan_pi.items()}
        return kw
    if store is None:
        return {}
    n_small, skip, n_distinct = _naive_share_caps(
        store, vis_sids, s_rows, B, beta, small_cost,
    )
    pi_override, _post = _pi_from_store(store, vis_sids, policy, rng)
    kw = dict(
        n_small_override=n_small,
        skip_small_sellers=skip,
        pi_override=pi_override,
        extra_small_sellers=extra_sellers,
        extra_small_budget=extra_budget,
        allow_sellers=set(vis_sids),
    )
    return kw


def _payments(sel, c, seller_ids, kind_arr, Xs, qs):
    sel = np.asarray(sel, bool)
    rows = []
    for s in np.unique(seller_ids):
        m = sel & (seller_ids == s)
        if not m.any():
            continue
        pkt = m & (kind_arr == "packet")
        n_rows = int(sum(len(Xs[j]) for j in np.where(m)[0]))
        q = float(qs[m][0])
        rows.append(dict(
            seller=int(s),
            true_pi=float(1.0 - q),
            paid=float(c[m].sum()),
            packets_bought=int(pkt.sum()),
            rows_bought=n_rows,
            packet_paid=float(c[pkt].sum()) if pkt.any() else 0.0,
        ))
    return rows


def run_one_buyer(
    policy: str,
    vis: dict,
    groups,
    demand,
    Sigmas,
    Sigma_pre,
    M0, X0, y0,
    X_mix, y_mix, yid_mix,
    cfg: dict,
    rng,
    store: EvidenceStore | None,
    buyer_store: BuyerEvidenceStore | None = None,
    plan_pi: dict | None = None,
    exhaust_budget: bool = False,
    platform_small_idx=None,
    extra_skip_sellers=None,
):
    """Run one policy for one buyer. Returns (summary, payments, gamma_by_row)."""
    dataset = cfg["dataset"]
    beta = float(cfg["beta"])
    s_rows = int(cfg["s_rows"])
    B = float(cfg["B"])
    rho = float(cfg["rho"])
    lam, sigma2 = cfg["lam"], cfg["sigma2"]
    kind = task_kind(dataset)
    F, c = vis["F"], vis["c"]
    Xs, ys, yids = vis["Xs"], vis["ys"], vis["yids"]
    ys_clean = vis.get("ys_clean")
    qs, seller_ids = vis["qs"], vis["seller_ids"]
    sources, kind_arr = vis["sources"], vis["kind"]
    offer_id = vis["offer_id"]
    small_cost = float(s_rows) / float(PKT_ROWS)
    vis_sids = sorted({int(s) for s in seller_ids})
    random_small = str(cfg.get("sampling_rule", "value")) == "uniform"
    cifar = dataset == "cifar10n" or cfg.get("cifar", False)

    excesses, losses, small_spends, packet_spends = [], [], [], []
    pay_sum = {}
    gamma_lists: dict[tuple[int, int], list[float]] = {}
    pkt_sellers = set()
    n_checked_acc = []
    pi_small_lists: dict[int, list[float]] = {}
    pi_final_lists: dict[int, list[float]] = {}
    small_sid_acc: set[int] = set()

    for zi, z_name in enumerate(demand):
        Xz = groups[z_name]["X_test"]
        yz = groups[z_name]["y_test"]
        yidz = groups[z_name].get("y_id_test")
        if cifar:
            mse_star = cifar_oracle_train_pool(
                vis.get("oracle_sources") or cfg.get("oracle_sources") or {},
                groups[z_name]["classes"], lam, Xz, yz,
            )
            mse_star_mix = cifar_oracle_train_pool(
                vis.get("oracle_sources") or cfg.get("oracle_sources") or {},
                groups[z_name]["classes"], lam, X_mix, y_mix,
            )
        else:
            mse_star = oracle_mse(groups, z_name, lam, Xz, yz)
            mz = Ridge(alpha=lam, fit_intercept=False).fit(
                groups[z_name]["X_sell"], groups[z_name]["y_sell"],
            )
            mse_star_mix = float(np.mean((mz.predict(X_mix) - y_mix) ** 2))
        L_star = weighted_loss(mse_star_mix, mse_star, rho)
        acq_kw = _share_kwargs(
            policy, store, vis_sids, s_rows, B, beta, small_cost, rng,
            vis=vis, demand=demand, M0=M0, Sigma_pre=Sigma_pre,
            Sigmas=Sigmas, rho=rho, sigma2=sigma2, z_idx=zi,
            plan_pi=plan_pi, buyer_store=buyer_store,
        )
        if exhaust_budget:
            acq_kw["exhaust_budget"] = True
        if platform_small_idx:
            acq_kw["platform_small_idx"] = list(platform_small_idx)
        if extra_skip_sellers:
            skip = set(acq_kw.get("skip_small_sellers") or [])
            skip |= {int(s) for s in extra_skip_sellers}
            acq_kw["skip_small_sellers"] = skip

        sel, alph, _tr, _qe, _qt, nchk, psp, _pr, sw, row_w, pi_s, pi_f, small_ids = strategy_p_variant(
            F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, beta, s_rows,
            seller_ids, sources, Xs, ys, qs, X0, y0, lam, kind, kind_arr,
            zi, rng, reuse=True, use_quality=True, random_small=random_small,
            ys_clean=ys_clean, yids=yids, **acq_kw,
        )
        primary_w = True
        for s, v in (pi_s or {}).items():
            pi_small_lists.setdefault(int(s), []).append(float(v))
        for s, v in (pi_f or {}).items():
            pi_final_lists.setdefault(int(s), []).append(float(v))
        for s in (small_ids or []):
            small_sid_acc.add(int(s))

        pkt_w = None
        if primary_w and sw:
            pkt_w = np.array([float(sw.get(int(s), 1.0)) for s in seller_ids], float)
        ev = fit_eval(
            sel, X0, y0, Xs, ys, yids, qs, lam,
            X_mix, y_mix, yid_mix, Xz, yz, yidz,
            also_weighted=False, pkt_weights=pkt_w, primary_weighted=primary_w,
            row_weights=row_w,
        )
        loss = weighted_loss(ev["mse_mix"], ev["mse_z"], rho)
        losses.append(loss)
        excesses.append(loss - L_star)
        small_spends.append(float(psp))
        packet_spends.append(float(c[sel & (kind_arr == "packet")].sum()) if sel.any() else 0.0)
        n_checked_acc.append(int(nchk))

        buyer_sel = np.asarray(sel, bool).copy()
        if platform_small_idx:
            for pj in platform_small_idx:
                if 0 <= int(pj) < len(buyer_sel):
                    buyer_sel[int(pj)] = False
        for rec in _payments(buyer_sel, c, seller_ids, kind_arr, Xs, qs):
            s = rec["seller"]
            acc = pay_sum.setdefault(s, dict(
                true_pi=rec["true_pi"], paid=0.0, packets_bought=0.0,
                rows_bought=0.0, packet_paid=0.0, n=0,
            ))
            acc["paid"] += rec["paid"]
            acc["packets_bought"] += rec["packets_bought"]
            acc["rows_bought"] += rec["rows_bought"]
            acc["packet_paid"] += rec["packet_paid"]
            if rec["packets_bought"] > 0:
                pkt_sellers.add(s)

        if row_w is not None:
            for j in np.where(np.asarray(sel, bool))[0]:
                oid = int(offer_id[j])
                w = _row_array(row_w, j, len(Xs[j]))
                for i, gi in enumerate(w):
                    gamma_lists.setdefault((oid, i), []).append(float(gi))

    n_z = max(len(demand), 1)
    plat_paid_by_s: dict[int, float] = {}
    plat_rows_by_s: dict[int, int] = {}
    if platform_small_idx:
        for pj in platform_small_idx:
            pj = int(pj)
            if 0 <= pj < len(c):
                s = int(seller_ids[pj])
                plat_paid_by_s[s] = plat_paid_by_s.get(s, 0.0) + float(c[pj])
                plat_rows_by_s[s] = plat_rows_by_s.get(s, 0) + int(len(Xs[pj]))
    plat_paid = float(sum(plat_paid_by_s.values()))
    plat_rows = int(sum(plat_rows_by_s.values()))
    plat_seller = next(iter(plat_paid_by_s)) if len(plat_paid_by_s) == 1 else None
    payments = []
    seen_pay = set()
    for s, acc in pay_sum.items():
        seen_pay.add(int(s))
        payments.append(dict(
            seller=int(s),
            true_pi=float(acc["true_pi"]),
            paid=float(acc["paid"] / n_z),
            packets_bought=float(acc["packets_bought"] / n_z),
            rows_bought=float(acc["rows_bought"] / n_z),
            packet_paid=float(acc["packet_paid"] / n_z),
            bought_packet=int(s in pkt_sellers),
            platform_paid=float(plat_paid_by_s.get(int(s), 0.0)),
        ))
    for s, paid in plat_paid_by_s.items():
        if int(s) in seen_pay:
            continue
        qv = float(qs[seller_ids == s][0]) if np.any(seller_ids == s) else np.nan
        payments.append(dict(
            seller=int(s), true_pi=float(1.0 - qv),
            paid=0.0, packets_bought=0.0, rows_bought=0.0, packet_paid=0.0,
            bought_packet=0, platform_paid=float(paid),
        ))
    gamma_by_row = {k: float(np.mean(v)) for k, v in gamma_lists.items()}
    pkt_spend = float(np.mean(packet_spends)) if packet_spends else 0.0
    pr_spend = float(np.mean(small_spends)) if small_spends else 0.0
    summary = dict(
        excess=float(np.mean(excesses)),
        test_loss=float(np.mean(losses)) if losses else np.nan,
        small_purchase_spend=pr_spend,
        packet_spend=pkt_spend,
        n_checked=float(np.mean(n_checked_acc)) if n_checked_acc else 0.0,
        n_z=n_z,
        platform_paid=float(plat_paid),
        platform_seller=plat_seller,
        platform_rows=int(plat_rows),
        platform_by_seller={
            int(s): dict(paid=float(plat_paid_by_s[s]), rows=int(plat_rows_by_s.get(s, 0)))
            for s in plat_paid_by_s
        },
        pi_small={int(s): float(np.mean(vs)) for s, vs in pi_small_lists.items()},
        pi_final={int(s): float(np.mean(vs)) for s, vs in pi_final_lists.items()},
        small_sids=sorted(small_sid_acc),
    )
    return summary, payments, gamma_by_row, pkt_sellers
