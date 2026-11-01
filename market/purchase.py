"""Fixed-β purchased-sample policy: screen sellers, then fill packets."""

from __future__ import annotations

import numpy as np

from market.build import idx_of, rank_packets
from market.const import PKT_ROWS
from market.features import _risk, best_alpha

def _M_weighted(F, M0, selected, w_pkt):
    """M = M0 + Σ_j selected w_j F_j."""
    M = np.asarray(M0, float).copy()
    for j in np.where(selected)[0]:
        M += float(w_pkt[j]) * F[j]
    return M

def strategy_p_variant(
    F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, beta, s_rows,
    seller_ids, sources, Xs, ys, qs, X0, y0, lam, kind, kind_arr,
    z_idx, rng, reuse=True, use_quality=True,
    random_small=False, allow_sellers=None,
    ys_clean=None, yids=None,
    n_small_override=None, skip_small_sellers=None, pi_override=None,
    extra_small_sellers=None, extra_small_budget=0.0,
    pi_plan=None,
    exhaust_budget=False,
    platform_small_idx=None,
):
    """Fixed-β purchased-sample policy with contamination posteriors.

    Mixture π_j p_clean + (1-π_j) p_noise; π̂_j = Beta(2,1) posterior mean.
    Amazon: discrete Gauss clean PMF on {1..5}, p_noise=1/5.
    DomainNet: logistic p_clean, p_noise=1/C.
    Design M uses seller π̂_j; training uses per-row clean posteriors (3 rounds).
    n_small_override: if set (can be 0), replace floor(βB / small_cost).
    skip_small_sellers: do not buy small_s from these ids.
    pi_override: unchecked sellers use this π̂ (shared-evidence policies).
    extra_small_sellers / extra_small_budget: additional small purchases after the main cap.
    pi_plan: if set, use these π only for α / packet fill (TS-plan).
    exhaust_budget: after packet fill, spend leftover (<1) on unused small_s.
    platform_small_idx: offer indices bought by the platform (entry_fund).
    They join EM / train but are not charged to B.
    random_small=True: uniform seller coverage (sampling_rule=uniform).
    """
    from market.contam import (
        SCREEN_ROUNDS,
        estimate_contamination_on_mask,
        packet_row_weights_from_gamma,
        seed_sigma_c2,
    )

    seller_ids = np.asarray(seller_ids)
    kind_arr = np.asarray(kind_arr)
    is_small = kind_arr != "packet"
    unique = [int(s) for s in np.unique(seller_ids)]
    small_cost = float(s_rows) / float(PKT_ROWS)
    if n_small_override is None:
        n_small = max(1, int(np.floor(beta * B / max(small_cost, 1e-9))))
    else:
        n_small = max(0, int(n_small_override))
    if allow_sellers is not None:
        allow_sellers = {int(x) for x in allow_sellers}
    skip_small_sellers = {int(x) for x in (skip_small_sellers or [])}
    ranked = rank_packets(F, M0, Sigma_pre, sigma2, seller_ids, kind_arr)
    if random_small:
        sellers = [s for s, _ in ranked]
        rng.shuffle(sellers)
        ranked = [(s, 0.0) for s in sellers]
    if allow_sellers is not None:
        ranked = [(s, g) for s, g in ranked if int(s) in allow_sellers]
    ranked_small = [(s, g) for s, g in ranked if int(s) not in skip_small_sellers]

    small_mask = np.zeros(len(c), bool)
    q_est, q_true_chk, traj = {}, {}, []
    rem, small_purchase_spend, n_checked = float(B), 0.0, 0
    for s, g in ranked_small[:n_small]:
        pj = idx_of(seller_ids, kind_arr, s, "small_s")
        if pj < 0 or c[pj] > rem + 1e-9:
            continue
        small_mask[pj] = True
        rem -= c[pj]
        small_purchase_spend += c[pj]
        n_checked += 1
        q_true_chk[s] = float(qs[pj])
        traj.append(dict(
            step=len(traj), phase="purchase", rem_B=rem,
            chosen_seller=int(s), chosen_pkt=int(pj), is_small=True,
            q_est=np.nan, q_true=float(qs[pj]), score=float(g),
        ))
    extra_small_budget = float(extra_small_budget or 0.0)
    if extra_small_budget > 1e-12:
        extra_rem = extra_small_budget
        already = {int(s) for s in q_true_chk}
        for s in extra_small_sellers or []:
            s = int(s)
            if extra_rem < small_cost - 1e-9:
                break
            if s in already:
                continue
            if allow_sellers is not None and s not in allow_sellers:
                continue
            pj = idx_of(seller_ids, kind_arr, s, "small_s")
            if pj < 0 or c[pj] > rem + 1e-9:
                continue
            small_mask[pj] = True
            rem -= c[pj]
            extra_rem -= float(c[pj])
            small_purchase_spend += c[pj]
            n_checked += 1
            already.add(s)
            q_true_chk[s] = float(qs[pj])
            traj.append(dict(
                step=len(traj), phase="explore", rem_B=rem,
                chosen_seller=int(s), chosen_pkt=int(pj), is_small=True,
                q_est=np.nan, q_true=float(qs[pj]), score=0.0,
            ))

    dataset = "amazon" if kind == "regression" else "domainnet"
    C = int(np.asarray(y0).shape[1]) if np.asarray(y0).ndim == 2 else None
    seller_w = {s: 1.0 for s in unique}
    row_w = {}
    sigma_c2 = seed_sigma_c2(X0, y0, lam) if dataset == "amazon" else None

    if use_quality and n_checked >= 1:
        pi_small, _, sigma_c2, _ = estimate_contamination_on_mask(
            dataset, seller_ids, sources, small_mask, Xs, ys, yids, ys_clean,
            X0, y0, lam, sigma_c2=sigma_c2, C=C, n_rounds=1,
            leave_seller_out=True,
        )
        pi_mean = float(np.mean(list(pi_small.values()))) if pi_small else 0.5
        seller_w = {s: float(pi_small.get(s, pi_mean)) for s in unique}
        for s in q_true_chk:
            q_est[s] = float(seller_w.get(int(s), 1.0))
        for tr in traj:
            s = int(tr["chosen_seller"])
            tr["q_est"] = float(seller_w.get(s, 1.0))
            tr["w"] = float(seller_w.get(s, 1.0))
            tr["pi_hat"] = float(seller_w.get(s, 1.0))

    small_sids = {int(s) for s in q_true_chk}
    if pi_override:
        for s, pval in pi_override.items():
            s = int(s)
            if s in small_sids:
                continue
            seller_w[s] = float(np.clip(float(pval), 1e-3, 1.0 - 1e-3))
            q_est[s] = seller_w[s]

    w_pkt = np.array([float(seller_w.get(int(s), 1.0)) for s in seller_ids], float)
    if pi_plan:
        for j, s in enumerate(seller_ids):
            s = int(s)
            if s in small_sids:
                continue
            if s in pi_plan:
                w_pkt[j] = float(np.clip(float(pi_plan[s]), 1e-3, 1.0 - 1e-3))
    if not use_quality:
        w_pkt[:] = 1.0
        seller_w = {s: 1.0 for s in unique}

    Fq = F * w_pkt[:, None, None]
    pk = ~is_small
    alpha = best_alpha(Fq[pk], c[pk], M0, Sigma_pre, Sigmas, B, rho, sigma2)
    B1 = float(np.floor(alpha * B))

    def _fill(Sigma, budget_cap, selected0):
        selected = np.asarray(selected0, bool).copy()
        while True:
            spent = float(c[selected].sum()) if selected.any() else 0.0
            rem = float(budget_cap) - spent
            if rem < 1e-9:
                break
            M = _M_weighted(F, M0, selected, w_pkt)
            try:
                loss = _risk(Sigma, M, sigma2)
            except np.linalg.LinAlgError:
                break
            best_j, best_score = -1, 0.0
            for j in range(len(c)):
                if selected[j] or c[j] > rem + 1e-9 or is_small[j]:
                    continue
                s = int(seller_ids[j])
                if allow_sellers is not None and s not in allow_sellers:
                    continue
                wj = float(w_pkt[j])
                try:
                    gain = loss - _risk(Sigma, M + wj * F[j], sigma2)
                except np.linalg.LinAlgError:
                    continue
                score = gain / c[j]
                if score > best_score:
                    best_score, best_j = score, j
            if best_j < 0 or best_score <= 0:
                break
            selected[best_j] = True
        return selected

    if reuse:
        selected = small_mask.copy()
        spent = float(c[selected].sum())
        s1 = _fill(Sigma_pre, max(B1, spent), selected)
        s2 = _fill(Sigmas[z_idx], B, s1)
    else:
        selected = np.zeros(len(c), bool)
        B1_pkt = max(B1 - small_purchase_spend, 0.0)
        B_pkt = max(B - small_purchase_spend, 0.0)
        s1 = _fill(Sigma_pre, B1_pkt, selected)
        s2 = _fill(Sigmas[z_idx], B_pkt, s1)

    if exhaust_budget:
        # Leftover is < 1 when n_small * small_cost is not integer in B.
        # Spend it on unused small_s so instance totals hit B per buyer.
        while True:
            rem = float(B) - float(c[s2].sum())
            if rem < small_cost - 1e-9:
                break
            added = False
            for s, g in ranked_small:
                pj = idx_of(seller_ids, kind_arr, s, "small_s")
                if pj < 0 or s2[pj] or c[pj] > rem + 1e-9:
                    continue
                if allow_sellers is not None and int(s) not in allow_sellers:
                    continue
                s2[pj] = True
                small_mask[pj] = True
                small_purchase_spend += float(c[pj])
                n_checked += 1
                q_true_chk[int(s)] = float(qs[pj])
                added = True
                break
            if not added:
                break

    if platform_small_idx:
        for pj in platform_small_idx:
            pj = int(pj)
            if pj < 0 or pj >= len(c):
                continue
            s2[pj] = True
            small_mask[pj] = True
            s = int(seller_ids[pj])
            if s not in q_true_chk:
                n_checked += 1
                q_true_chk[s] = float(qs[pj])

    # training row weights
    pi_final = {}
    if use_quality and (
        n_checked >= 1 or (pi_override and bool(np.any(s2)))
    ):
        pi_final, gmap, _, _ = estimate_contamination_on_mask(
            dataset, seller_ids, sources, s2, Xs, ys, yids, ys_clean,
            X0, y0, lam, sigma_c2=sigma_c2, C=C, n_rounds=SCREEN_ROUNDS,
            leave_seller_out=True,
        )
        row_w = packet_row_weights_from_gamma(seller_ids, s2, Xs, gmap, pi_final)
    else:
        row_w = {j: 1.0 for j in np.where(s2)[0]}

    # logging only: planning π after small (seller_w) and π after packets
    pi_small_out = {int(s): float(w) for s, w in seller_w.items()}
    pi_final_out = {int(s): float(p) for s, p in (pi_final or {}).items()}
    small_sids_out = {int(s) for s in q_true_chk}
    return (
        s2, alpha, traj, q_est, q_true_chk, n_checked, small_purchase_spend,
        small_mask, seller_w, row_w, pi_small_out, pi_final_out, small_sids_out,
    )

