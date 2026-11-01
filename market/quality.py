"""Cross-fitted residual variance on a labeled set."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def crossfit_resid_sigma2(X, y, lam: float, n_folds: int = 5, rng=None) -> float:
    """Cross-fitted residual variance on a labeled set (seed or seller pool).

    For multi-output (one-hot classification ridge), mean over all output dims.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(X)
    if n < 2:
        model = Ridge(alpha=lam, fit_intercept=False).fit(X, y)
        return max(float(np.mean((model.predict(X) - y) ** 2)), 1e-12)
    rng = np.random.default_rng(0 if rng is None else rng)
    n_folds = int(max(2, min(n_folds, n)))
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    sq, cnt = 0.0, 0
    for i in range(n_folds):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(tr) < 1 or len(te) < 1:
            continue
        model = Ridge(alpha=lam, fit_intercept=False).fit(X[tr], y[tr])
        err = (model.predict(X[te]) - y[te]) ** 2
        sq += float(np.sum(err))
        cnt += int(err.size)
    if cnt == 0:
        model = Ridge(alpha=lam, fit_intercept=False).fit(X, y)
        return max(float(np.mean((model.predict(X) - y) ** 2)), 1e-12)
    return max(sq / cnt, 1e-12)
