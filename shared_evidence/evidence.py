"""Platform store: per-row mean γ and seller Beta(2,1) posteriors."""

from __future__ import annotations

from dataclasses import dataclass, field

BETA_A, BETA_B = 2.0, 1.0


@dataclass
class EvidenceStore:
    """One store per (market instance, policy).

    Row key is (offer_id, local_i). Seller posterior uses each distinct bought
    row once, with γ̄_r = mean of buyers who purchased that row.
    """

    seller_of_offer: dict[int, int]
    row_sum: dict[tuple[int, int], float] = field(default_factory=dict)
    row_n: dict[tuple[int, int], int] = field(default_factory=dict)

    def update(self, gamma_by_row: dict[tuple[int, int], float]) -> None:
        for rid, g in gamma_by_row.items():
            self.row_sum[rid] = self.row_sum.get(rid, 0.0) + float(g)
            self.row_n[rid] = self.row_n.get(rid, 0) + 1

    def row_mean(self, rid: tuple[int, int]) -> float:
        return self.row_sum[rid] / max(self.row_n[rid], 1)

    def rows_of_seller(self, sid: int) -> list[tuple[int, int]]:
        sid = int(sid)
        return [rid for rid in self.row_sum if self.seller_of_offer.get(rid[0]) == sid]

    def n_distinct(self, sid: int) -> int:
        return len(self.rows_of_seller(sid))

    def seller_posterior(self, sid: int) -> tuple[float, float, float, int]:
        rids = self.rows_of_seller(sid)
        if not rids:
            a, b = BETA_A, BETA_B
            return a, b, a / (a + b), 0
        gbar = [self.row_mean(r) for r in rids]
        a = BETA_A + float(sum(gbar))
        b = BETA_B + float(sum(1.0 - g for g in gbar))
        return a, b, a / (a + b), len(rids)

    def beta_std(self, sid: int) -> float:
        a, b, _m, _n = self.seller_posterior(sid)
        den = (a + b) ** 2 * (a + b + 1.0)
        return float((a * b / den) ** 0.5) if den > 0 else 0.0


FIXED_MU0 = 2.0 / 3.0
FIXED_TAU0_2 = 1.0 / 18.0
TAU0_FLOOR = 0.0025


@dataclass
class BuyerEvidenceStore:
    """One record per (buyer, seller): π̂_jb = mean γ on rows that buyer bought from j."""

    n: dict[int, int] = field(default_factory=dict)
    s1: dict[int, float] = field(default_factory=dict)
    s2: dict[int, float] = field(default_factory=dict)
    nrows: dict[int, int] = field(default_factory=dict)
    prior: str = "eb"  # "eb" | "fixed"

    def n_buyers(self, sid: int) -> int:
        return int(self.n.get(int(sid), 0))

    def n_rows(self, sid: int) -> int:
        return int(self.nrows.get(int(sid), 0))

    def update_buyer(self, sid: int, pi_hat: float, n_rows: int, t: int) -> None:
        sid = int(sid)
        p = float(pi_hat)
        self.n[sid] = self.n.get(sid, 0) + 1
        self.s1[sid] = self.s1.get(sid, 0.0) + p
        self.s2[sid] = self.s2.get(sid, 0.0) + p * p
        self.nrows[sid] = self.nrows.get(sid, 0) + int(n_rows)

    def _sigma2_tau(self) -> tuple[float, float, float]:
        """Return (sigma2, mu0, tau0_2) from current accumulators."""
        js = [s for s, nj in self.n.items() if nj >= 1]
        js2 = [s for s in js if self.n[s] >= 2]
        if self.prior == "fixed" or len(js) < 2:
            mu0, tau0_2 = FIXED_MU0, FIXED_TAU0_2
        else:
            means = [self.s1[s] / self.n[s] for s in js]
            mu0 = float(sum(means) / len(means))
            # tau0_2 filled after sigma2 is known
            tau0_2 = None
        if not js2:
            sigma2 = FIXED_TAU0_2 if self.prior == "fixed" or len(js) < 2 else FIXED_TAU0_2
            if self.prior == "eb" and len(js) >= 2:
                means = [self.s1[s] / self.n[s] for s in js]
                var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
                tau0_2 = max(var_m, TAU0_FLOOR)
                sigma2 = tau0_2
            else:
                mu0, tau0_2, sigma2 = FIXED_MU0, FIXED_TAU0_2, FIXED_TAU0_2
            return sigma2, mu0, tau0_2
        sse = 0.0
        df = 0
        for s in js2:
            nj = self.n[s]
            sse += self.s2[s] - self.s1[s] ** 2 / nj
            df += nj - 1
        sigma2 = float(sse / max(df, 1))
        sigma2 = max(sigma2, 1e-8)
        if self.prior == "fixed" or len(js) < 2:
            return sigma2, FIXED_MU0, FIXED_TAU0_2
        means = [self.s1[s] / self.n[s] for s in js]
        mu0 = float(sum(means) / len(means))
        var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
        mean_se = float(sum(sigma2 / self.n[s] for s in js) / len(js))
        tau0_2 = var_m - mean_se
        if tau0_2 < TAU0_FLOOR:
            tau0_2 = TAU0_FLOOR
        return sigma2, mu0, float(tau0_2)

    def publish(self, sid: int) -> tuple[float, float, int, float, float, float]:
        """m_j, v_j, n_j, mu0, tau0_2, sigma2. No evidence returns the prior."""
        sid = int(sid)
        sigma2, mu0, tau0_2 = self._sigma2_tau()
        nj = self.n_buyers(sid)
        if nj <= 0:
            return float(mu0), float(tau0_2 ** 0.5), 0, mu0, tau0_2, sigma2
        prec = 1.0 / tau0_2 + nj / sigma2
        mj = (mu0 / tau0_2 + self.s1[sid] / sigma2) / prec
        vj = prec ** -0.5
        return float(mj), float(vj), nj, mu0, tau0_2, sigma2
