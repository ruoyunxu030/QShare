"""Lemma 4.2 slack δ for snapshot vs current EB hyperparameters."""

from __future__ import annotations

Z_LB = 0.8416  # Φ^{-1}(0.8)


def delta_lemma42(mu0, tau0_2, sigma2, mu0_s, tau0_2_s, sigma2_s, z: float = Z_LB) -> float:
    """δ from lemma 4.2. Current (μ0, τ0², σ²) vs snapshot starred values.

    δ = |μ0−μ0*| + |√ρ−√ρ*|/(√ρ+√ρ*) + z·(|σ−σ*|/√ρ + σ*·|1/√ρ − 1/√ρ*|)
    ρ = σ²/τ0², ρ* = σ*²/τ0*².
    """
    tau0_2 = max(float(tau0_2), 1e-15)
    tau0_2_s = max(float(tau0_2_s), 1e-15)
    sigma2 = max(float(sigma2), 1e-15)
    sigma2_s = max(float(sigma2_s), 1e-15)
    rho = sigma2 / tau0_2
    rho_s = sigma2_s / tau0_2_s
    sr, srs = rho ** 0.5, rho_s ** 0.5
    sig, sigs = sigma2 ** 0.5, sigma2_s ** 0.5
    d_mu = abs(float(mu0) - float(mu0_s))
    den = sr + srs
    d_rho = abs(sr - srs) / den if den > 0 else 0.0
    d_sig = float(z) * (abs(sig - sigs) / sr + sigs * abs(1.0 / sr - 1.0 / srs))
    return float(d_mu + d_rho + d_sig)
