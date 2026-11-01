"""Seller contamination posterior used for design weights and training γ.

Amazon:  p(y|x,j) = π_j · DiscreteGauss(y; μ(x), σ_c²) + (1-π_j)/5
DomainNet: p(y|x,j) = π_j · p_clean(y|x) + (1-π_j)/C
  (injection redraws Unif{0..C-1}, so contam likelihood is 1/C)

π_j via EM with Beta(2,1) prior, posterior mean (α=2, β=1):
  π̂_j = (2 + Σ γ_i) / (3 + n_j)

Dirty label: observed ≠ original (not the Bernoulli flip itself).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

EM_ITERS = 20
SCREEN_ROUNDS = 3
AMZ_SUPPORT = np.arange(1, 6, dtype=float)
BETA_A, BETA_B = 2.0, 1.0  # Beta(2,1) prior


def obs_dirty_mask(y_obs, y_clean) -> np.ndarray:
    """True iff observed label differs from original clean label."""
    yo = np.asarray(y_obs)
    yc = np.asarray(y_clean)
    if yo.ndim == 1:
        return yo.astype(float) != yc.astype(float)
    return np.argmax(yo, axis=1) != np.argmax(yc, axis=1)


def y_to_class_id(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    return np.argmax(y, axis=1).astype(int)


def seed_sigma_c2(X0, y0, lam: float) -> float:
    """Cross-fitted residual variance on clean seed (scalar MSE)."""
    from market.quality import crossfit_resid_sigma2
    return float(crossfit_resid_sigma2(X0, y0, lam))


def discrete_gauss_pmf(y_obs: np.ndarray, mu: np.ndarray, sigma_c2: float) -> np.ndarray:
    """Normalized discrete Gaussian PMF on {1..5} evaluated at y_obs."""
    y_obs = np.asarray(y_obs, float).ravel()
    mu = np.asarray(mu, float).ravel()
    sigma_c2 = max(float(sigma_c2), 1e-12)
    # (n, 5)
    logp = -0.5 * (AMZ_SUPPORT[None, :] - mu[:, None]) ** 2 / sigma_c2
    m = np.max(logp, axis=1, keepdims=True)
    p = np.exp(logp - m)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    idx = np.clip(np.rint(y_obs).astype(int) - 1, 0, 4)
    return p[np.arange(len(y_obs)), idx]


def fit_ref_amazon(X, y, lam, sample_weight=None):
    return Ridge(alpha=lam, fit_intercept=False).fit(
        X, y, sample_weight=sample_weight,
    )


def fit_ref_domainnet(X, y_id, C: int, lam: float, sample_weight=None):
    # map ridge-style λ to Logistic C ≈ 1/λ
    C_reg = float(max(1.0 / max(lam, 1e-8), 1e-4))
    clf = LogisticRegression(
        C=C_reg, solver="lbfgs",
        max_iter=400, fit_intercept=False, warm_start=False,
    )
    # ensure all classes present for sklearn
    y_id = np.asarray(y_id, int)
    classes = np.arange(C)
    if sample_weight is None:
        sample_weight = np.ones(len(y_id))
    # inject tiny weight on missing classes via synthetic points if needed
    present = set(np.unique(y_id))
    if len(present) < C:
        X_pad, y_pad, w_pad = [X], [y_id], [sample_weight]
        for c in classes:
            if c not in present:
                X_pad.append(np.zeros((1, X.shape[1])))
                y_pad.append(np.array([c]))
                w_pad.append(np.array([1e-6]))
        X = np.vstack(X_pad)
        y_id = np.concatenate(y_pad)
        sample_weight = np.concatenate(w_pad)
    clf.fit(X, y_id, sample_weight=sample_weight)
    return clf


def predict_mu_amazon(model, X) -> np.ndarray:
    return np.asarray(model.predict(X), float).ravel()


def predict_p_clean_domainnet(model, X, y_id, C: int) -> np.ndarray:
    y_id = np.asarray(y_id, int)
    proba = model.predict_proba(X)
    # align columns to 0..C-1
    full = np.full((len(X), C), 1.0 / C)
    for i, c in enumerate(model.classes_):
        full[:, int(c)] = proba[:, i]
    return full[np.arange(len(X)), np.clip(y_id, 0, C - 1)]


def em_pi_gamma(p_clean: np.ndarray, p_noise: float, seller_of_row: np.ndarray,
                n_iters: int = EM_ITERS):
    """EM for π_j with Beta(2,1) prior (posterior mean). Returns π dict, γ array."""
    p_clean = np.asarray(p_clean, float)
    seller_of_row = np.asarray(seller_of_row, int)
    sellers = sorted(set(seller_of_row.tolist()))
    by_s = {s: np.where(seller_of_row == s)[0] for s in sellers}
    pi = {s: 0.7 for s in sellers}
    gamma = np.full(len(p_clean), 0.7)
    p_n = max(float(p_noise), 1e-12)

    for _ in range(n_iters):
        for s in sellers:
            idx = by_s[s]
            pc = np.clip(p_clean[idx], 1e-300, 1.0)
            a = pi[s] * pc
            b = (1.0 - pi[s]) * p_n
            gamma[idx] = a / np.maximum(a + b, 1e-300)
        for s in sellers:
            idx = by_s[s]
            n_j = len(idx)
            pi[s] = float((BETA_A + float(gamma[idx].sum())) / (BETA_A + BETA_B + n_j))
            pi[s] = float(np.clip(pi[s], 1e-3, 1.0 - 1e-3))
    return pi, gamma


def _packets_of_mask(seller_ids, mask):
    seller_ids = np.asarray(seller_ids)
    mask = np.asarray(mask, bool)
    return [int(s) for s in np.unique(seller_ids[mask])] if mask.any() else []


def crossfit_p_clean_amazon(
    seller_ids, sources, mask, Xs, ys, X0, y0, lam, sigma_c2,
    row_weights_by_pkt=None, leave_seller_out=True,
):
    """Same-source reference fit; return list of (s,j,i,p_clean).

    leave_seller_out=True (default): μ_{-j} excludes seller j.
    """
    seller_ids = np.asarray(seller_ids)
    sources = np.asarray(sources)
    mask = np.asarray(mask, bool)
    rows = []  # (s, j, i, p_c, y_obs)
    sellers = _packets_of_mask(seller_ids, mask)
    for s in sellers:
        src = sources[seller_ids == s][0]
        s_pkts = [j for j in range(len(Xs)) if seller_ids[j] == s and mask[j]]
        other = [
            j for j in range(len(Xs))
            if mask[j] and sources[j] == src
            and (not leave_seller_out or seller_ids[j] != s)
        ]
        X_list, y_list, w_list = [X0], [y0], [np.ones(len(X0))]
        for j in other:
            X_list.append(Xs[j])
            y_list.append(ys[j])
            if row_weights_by_pkt is not None and j in row_weights_by_pkt:
                wj = np.asarray(row_weights_by_pkt[j], float)
                if wj.ndim == 0:
                    wj = np.full(len(Xs[j]), float(wj))
            else:
                wj = np.ones(len(Xs[j]))
            w_list.append(wj)
        Xtr = np.vstack(X_list)
        ytr = np.concatenate(y_list, axis=0)
        wtr = np.concatenate(w_list)
        keep = wtr > 1e-8
        model = fit_ref_amazon(Xtr[keep], ytr[keep], lam, sample_weight=wtr[keep])
        for j in s_pkts:
            mu = predict_mu_amazon(model, Xs[j])
            yj = np.asarray(ys[j], float).ravel()
            pc = discrete_gauss_pmf(yj, mu, sigma_c2)
            for i in range(len(yj)):
                rows.append((s, j, i, float(pc[i]), float(yj[i])))
    return rows


def crossfit_p_clean_domainnet(
    seller_ids, sources, mask, Xs, ys, yids, X0, y0, C, lam,
    row_weights_by_pkt=None, leave_seller_out=True,
):
    seller_ids = np.asarray(seller_ids)
    sources = np.asarray(sources)
    mask = np.asarray(mask, bool)
    y0_id = y_to_class_id(y0)
    rows = []
    sellers = _packets_of_mask(seller_ids, mask)
    for s in sellers:
        src = sources[seller_ids == s][0]
        s_pkts = [j for j in range(len(Xs)) if seller_ids[j] == s and mask[j]]
        other = [
            j for j in range(len(Xs))
            if mask[j] and sources[j] == src
            and (not leave_seller_out or seller_ids[j] != s)
        ]
        X_list, y_list, w_list = [X0], [y0_id], [np.ones(len(X0))]
        for j in other:
            X_list.append(Xs[j])
            yidj = yids[j] if yids[j] is not None else y_to_class_id(ys[j])
            y_list.append(np.asarray(yidj, int))
            if row_weights_by_pkt is not None and j in row_weights_by_pkt:
                wj = np.asarray(row_weights_by_pkt[j], float)
                if wj.ndim == 0:
                    wj = np.full(len(Xs[j]), float(wj))
            else:
                wj = np.ones(len(Xs[j]))
            w_list.append(wj)
        Xtr = np.vstack(X_list)
        ytr = np.concatenate(y_list)
        wtr = np.concatenate(w_list)
        keep = wtr > 1e-8
        model = fit_ref_domainnet(Xtr[keep], ytr[keep], C, lam, sample_weight=wtr[keep])
        for j in s_pkts:
            yidj = yids[j] if yids[j] is not None else y_to_class_id(ys[j])
            yidj = np.asarray(yidj, int)
            pc = predict_p_clean_domainnet(model, Xs[j], yidj, C)
            for i in range(len(yidj)):
                rows.append((s, j, i, float(pc[i]), int(yidj[i])))
    return rows


def rows_to_arrays(rows):
    if not rows:
        return np.array([]), np.array([]), []
    p_c = np.array([r[3] for r in rows], float)
    sellers = np.array([r[0] for r in rows], int)
    keys = [(r[0], r[1], r[2]) for r in rows]
    return p_c, sellers, keys


def gamma_map_from_rows(keys, gamma):
    return {k: float(gamma[i]) for i, k in enumerate(keys)}


def estimate_contamination_on_mask(
    dataset, seller_ids, sources, mask, Xs, ys, yids, ys_clean,
    X0, y0, lam, sigma_c2=None, C=None, n_rounds=1,
    leave_seller_out=True,
):
    """Screen purchased packets for π and per-row γ. n_rounds=1 at design; 3 at train.

    Returns π, gamma_by_(s,j,i), sigma_c2, meta.
    """
    seller_ids = np.asarray(seller_ids)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {}, {}, float(sigma_c2 or 1.0), {}

    if dataset == "amazon":
        if sigma_c2 is None:
            sigma_c2 = seed_sigma_c2(X0, y0, lam)
        p_noise = 1.0 / 5.0
        row_w = None
        pi, gmap = {}, {}
        for rnd in range(n_rounds):
            rows = crossfit_p_clean_amazon(
                seller_ids, sources, mask, Xs, ys, X0, y0, lam, sigma_c2,
                row_weights_by_pkt=row_w, leave_seller_out=leave_seller_out,
            )
            p_c, s_arr, keys = rows_to_arrays(rows)
            pi, gamma = em_pi_gamma(p_c, p_noise, s_arr)
            gmap = gamma_map_from_rows(keys, gamma)
            # build row_w for next round
            row_w = {}
            for j in np.where(mask)[0]:
                s = int(seller_ids[j])
                n = len(Xs[j])
                row_w[j] = np.array([gmap.get((s, j, i), pi.get(s, 0.5)) for i in range(n)])
        return pi, gmap, float(sigma_c2), dict(p_noise=p_noise, n_rounds=n_rounds)

    # domainnet
    if C is None:
        C = int(np.asarray(y0).shape[1]) if np.asarray(y0).ndim == 2 else int(len(np.unique(y_to_class_id(y0))))
    p_noise = 1.0 / float(C)
    row_w = None
    pi, gmap = {}, {}
    for rnd in range(n_rounds):
        rows = crossfit_p_clean_domainnet(
            seller_ids, sources, mask, Xs, ys, yids, X0, y0, C, lam,
            row_weights_by_pkt=row_w, leave_seller_out=leave_seller_out,
        )
        p_c, s_arr, keys = rows_to_arrays(rows)
        pi, gamma = em_pi_gamma(p_c, p_noise, s_arr)
        gmap = gamma_map_from_rows(keys, gamma)
        row_w = {}
        for j in np.where(mask)[0]:
            s = int(seller_ids[j])
            n = len(Xs[j])
            row_w[j] = np.array([gmap.get((s, j, i), pi.get(s, 0.5)) for i in range(n)])
    return pi, gmap, float(sigma_c2 or 1.0), dict(p_noise=p_noise, n_rounds=n_rounds, C=C)


def packet_row_weights_from_gamma(seller_ids, selected, Xs, gamma_map, pi):
    seller_ids = np.asarray(seller_ids)
    row_w = {}
    for j in np.where(selected)[0]:
        s = int(seller_ids[j])
        n = len(Xs[j])
        row_w[j] = np.array([
            float(gamma_map.get((s, j, i), pi.get(s, 0.5))) for i in range(n)
        ])
    return row_w
