"""Ridge risk, packet selection, and train/eval helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from market.const import ALPHAS, PKT_ROWS

def noise_weight(q: float) -> float:
    """Effective Fisher / sample weight: (1−q)² signal attenuation."""
    return float(max(1.0 - q, 0.0) ** 2)

def _M_of_bool(F, M0, selected):
    M = np.asarray(M0, float).copy()
    for j, s in enumerate(selected):
        if s:
            M = M + F[j]
    return M

def _risk(Q, M, sigma2=1.0):
    return float(sigma2 * np.trace(Q @ np.linalg.inv(M)))

def greedy_plus_swap_cpu(F, c, M0, Q, B, sigma2=1.0, max_swap_rounds: int = 5):
    F = np.asarray(F, float)
    c = np.asarray(c, float)
    N = F.shape[0]
    selected = np.zeros(N, dtype=bool)
    spent = 0.0
    M = np.asarray(M0, float).copy()
    # greedy
    while True:
        try:
            loss = _risk(Q, M, sigma2)
        except np.linalg.LinAlgError:
            break
        best_j, best_score = -1, 0.0
        for j in range(N):
            if selected[j] or spent + c[j] > B + 1e-9:
                continue
            try:
                gain = loss - _risk(Q, M + F[j], sigma2)
            except np.linalg.LinAlgError:
                continue
            score = gain / c[j] if c[j] > 0 else gain
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0 or best_score <= 0:
            break
        selected[best_j] = True
        spent += c[best_j]
        M = M + F[best_j]
    # 1-swap with incremental M (avoid full rebuild)
    for _ in range(max_swap_rounds):
        try:
            loss = _risk(Q, M, sigma2)
        except np.linalg.LinAlgError:
            break
        spent = float(c[selected].sum()) if selected.any() else 0.0
        in_idx = np.where(selected)[0]
        out_idx = np.where(~selected)[0]
        best = None
        for i in in_idx:
            for j in out_idx:
                if spent - c[i] + c[j] > B + 1e-9:
                    continue
                try:
                    loss2 = _risk(Q, M - F[i] + F[j], sigma2)
                except np.linalg.LinAlgError:
                    continue
                gain = loss - loss2
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, i, j)
        if best is None:
            break
        _, i, j = best
        selected[i] = False
        selected[j] = True
        M = M - F[i] + F[j]
    return selected

def draw_seed(rng, groups, used, n_seed, d, lam):
    blocks_X, blocks_y = [], []
    for g, dat in groups.items():
        Xs, ys = dat["X_sell"], dat["y_sell"]
        for i in range(len(Xs)):
            if i not in used.get(g, set()):
                blocks_X.append(Xs[i])
                blocks_y.append(ys[i])
    if len(blocks_X) < n_seed:
        blocks_X = [dat["X_sell"][i] for dat in groups.values() for i in range(len(dat["X_sell"]))]
        blocks_y = [dat["y_sell"][i] for dat in groups.values() for i in range(len(dat["y_sell"]))]
    idx = rng.choice(len(blocks_X), size=n_seed, replace=False)
    X0 = np.stack([blocks_X[i] for i in idx], axis=0)
    y0 = np.stack([blocks_y[i] for i in idx], axis=0)
    M0 = lam * np.eye(d) + X0.T @ X0
    return M0, X0, y0

def select_open(F, c, M0, Sigma, B, sigma2):
    return greedy_plus_swap_cpu(F, c, M0, Sigma, B, sigma2=sigma2)

def select_extend(F, c, M0, Sigma, B, owned, sigma2):
    """Keep owned packets; fill remainder of budget with greedy+swap on rest."""
    owned = np.asarray(owned, bool)
    spent = float(c[owned].sum()) if owned.any() else 0.0
    rem = float(B) - spent
    if rem < 1e-9 or not (~owned).any():
        return owned.copy()
    M1 = _M_of_bool(F, M0, owned)
    remain = ~owned
    Fr, cr = F[remain], c[remain]
    loc = greedy_plus_swap_cpu(Fr, cr, M1, Sigma, rem, sigma2=sigma2)
    out = owned.copy()
    out[np.where(remain)[0][loc]] = True
    return out

def two_stage_proxy(F, c, M0, Sigma_pre, Sigmas, B, alpha, rho, sigma2):
    B1 = float(np.floor(alpha * B))
    s1 = select_open(F, c, M0, Sigma_pre, B1, sigma2)
    M1 = _M_of_bool(F, M0, s1)
    pre = _risk(Sigma_pre, M1, sigma2)
    post = 0.0
    for Sig in Sigmas:
        s2 = select_extend(F, c, M0, Sig, B, s1, sigma2)
        M2 = _M_of_bool(F, M0, s2)
        post += _risk(Sig, M2, sigma2)
    post /= len(Sigmas)
    return s1, pre + float(rho) * post

def best_alpha(F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, alphas=None):
    best_a, best_proxy = 0.5, np.inf
    for a in (alphas if alphas is not None else ALPHAS):
        _, proxy = two_stage_proxy(F, c, M0, Sigma_pre, Sigmas, B, a, rho, sigma2)
        if proxy < best_proxy:
            best_proxy, best_a = proxy, a
    return best_a

def mse_of(model, X, y):
    pred = model.predict(X)
    return float(np.mean((pred - y) ** 2))

def acc_of(model, X, y_id):
    if y_id is None:
        return float("nan")
    pred = model.predict(X)
    if pred.ndim == 1:
        return float("nan")
    return float(np.mean(np.argmax(pred, axis=1) == np.asarray(y_id)))

def gather_selected(sel, Xs, ys, yids, qs, pkt_weights=None, row_weights=None):
    """Gather selected packets.

    Priority: row_weights[j] (per-row array) > pkt_weights[j] (scalar) > noise_weight(qs[j]).
    """
    X_list, y_list, id_list, w_list = [], [], [], []
    for j, take in enumerate(sel):
        if not take:
            continue
        X_list.append(Xs[j])
        y_list.append(ys[j])
        n_j = len(Xs[j])
        if row_weights is not None and j in row_weights:
            wj = np.asarray(row_weights[j], float)
            if wj.ndim == 0:
                wj = np.full(n_j, float(wj))
            w_list.extend(wj.tolist())
        else:
            wj = float(pkt_weights[j]) if pkt_weights is not None else noise_weight(qs[j])
            w_list.extend([wj] * n_j)
        if yids[j] is not None:
            id_list.append(yids[j])
    if not X_list:
        return None, None, None, None
    X = np.vstack(X_list)
    y = np.concatenate(y_list, axis=0)
    yid = np.concatenate(id_list, axis=0) if id_list else None
    w = np.asarray(w_list, float)
    return X, y, yid, w

def fit_eval(
    sel,
    X0,
    y0,
    Xs,
    ys,
    yids,
    qs,
    lam,
    X_mix,
    y_mix,
    yid_mix,
    X_z,
    y_z,
    yid_z,
    also_weighted: bool = False,
    pkt_weights=None,
    primary_weighted: bool = False,
    row_weights=None,
):
    """Fit ridge on selected packets (+ seed).

    If primary_weighted and (row_weights or pkt_weights) given, the primary mse_*
    columns come from sample-weighted ridge.
    also_weighted still fills mse_*_w from weighted fit when requested.
    """
    Xp, yp, yidp, wp = gather_selected(
        sel, Xs, ys, yids, qs, pkt_weights=pkt_weights, row_weights=row_weights,
    )
    if Xp is None:
        X_all, y_all = X0, y0
        w_all = np.ones(len(X0))
        yid_all = None
    else:
        X_all = np.vstack([X0, Xp])
        y_all = np.concatenate([y0, yp], axis=0)
        w_all = np.concatenate([np.ones(len(X0)), wp], axis=0)
        yid_all = yidp

    use_w_primary = bool(
        primary_weighted and (row_weights is not None or pkt_weights is not None)
    )
    if use_w_primary:
        ridge = Ridge(alpha=lam, fit_intercept=False).fit(
            X_all, y_all, sample_weight=w_all
        )
    else:
        ridge = Ridge(alpha=lam, fit_intercept=False).fit(X_all, y_all)
    mse_mix = mse_of(ridge, X_mix, y_mix)
    mse_z = mse_of(ridge, X_z, y_z)
    acc_mix = acc_of(ridge, X_mix, yid_mix) if yid_mix is not None else np.nan
    acc_z = acc_of(ridge, X_z, yid_z) if yid_z is not None else np.nan
    out = dict(mse_mix=mse_mix, mse_z=mse_z, acc_mix=acc_mix, acc_z=acc_z)

    if also_weighted or use_w_primary:
        wr = Ridge(alpha=lam, fit_intercept=False).fit(X_all, y_all, sample_weight=w_all)
        out["mse_mix_w"] = mse_of(wr, X_mix, y_mix)
        out["mse_z_w"] = mse_of(wr, X_z, y_z)
        out["acc_mix_w"] = acc_of(wr, X_mix, yid_mix) if yid_mix is not None else np.nan
        out["acc_z_w"] = acc_of(wr, X_z, yid_z) if yid_z is not None else np.nan
    return out

def oracle_mse(groups, z_name, lam, X_test, y_test):
    model = Ridge(alpha=lam, fit_intercept=False).fit(
        groups[z_name]["X_sell"], groups[z_name]["y_sell"]
    )
    return mse_of(model, X_test, y_test)

def mixed_test(groups, demand):
    Xt = np.vstack([groups[g]["X_test"] for g in demand])
    yt = np.concatenate([groups[g]["y_test"] for g in demand], axis=0)
    if "y_id_test" in groups[demand[0]]:
        yid = np.concatenate([groups[g]["y_id_test"] for g in demand], axis=0)
    else:
        yid = None
    return Xt, yt, yid

def weighted_loss(pre, post, rho):
    return float(pre) + float(rho) * float(post)

def task_kind(dataset: str) -> str:
    return "regression" if dataset == "amazon" else "classification"

def cifar_oracle_train_pool(sources, classes, lam, X_test, y_test):
    """Fit on clean one-hots from seller pools (train split), never test labels."""
    Xs, ys = [], []
    cls = set(int(c) for c in np.asarray(classes).tolist())
    for _name, dat in sources.items():
        y_clean = dat["y_clean_sell"]
        mask = np.isin(y_clean, list(cls))
        if not mask.any():
            continue
        Xs.append(dat["X_sell"][mask])
        # clean one-hot
        C = dat["y_sell"].shape[1]
        Y = np.eye(C)[y_clean[mask]]
        ys.append(Y)
    if not Xs:
        return float("nan")
    model = Ridge(alpha=lam, fit_intercept=False).fit(np.vstack(Xs), np.concatenate(ys, 0))
    return float(np.mean((model.predict(X_test) - y_test) ** 2))

