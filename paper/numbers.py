#!/usr/bin/env python3
"""Print body-text numbers that are not in Tables 2–7."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MKT = ROOT / "results" / "market"
STO = ROOT / "results" / "store"


def main():
    bv = pd.read_csv(MKT / "buyer_variance.csv")
    print("buyer variance (crossed RE, score error, QShare)")
    for _, r in bv.iterrows():
        print(f"  {r['dataset']}: {float(r['var_buyer']):.4f}")

    io = pd.read_csv(MKT / "task_inout.csv")
    print("in-task minus out-task score error")
    for _, r in io.iterrows():
        print(
            f"  {r['dataset']}: {float(r['diff']):.3f} "
            f"[{float(r['diff_lo']):.3f}, {float(r['diff_hi']):.3f}]"
        )

    fg = pd.read_csv(MKT / "funded_gap.csv")
    print(f"funded vs own-row error, max |diff| = {float(fg['abs_diff'].max()):.4f}")

    fb = pd.read_csv(MKT / "first_buyer_auc.csv")
    print("AUC after the first buyer, Amazon")
    for _, r in fb.iterrows():
        print(f"  {r['method']}: {float(r['auc']):.2f}")

    sc = pd.read_csv(STO / "scan_all.csv")
    print("scan-all / QShare at the same 99,750-write snapshot")
    for _, r in sc.iterrows():
        v = float(r["scan_over_qshare"])
        print(f"  {r['workload']}: {v:.1f}x" if v < 10 else f"  {r['workload']}: {int(round(v))}x")

    rf = pd.read_csv(STO / "refresh_interval.csv")
    print("refresh median s:", ", ".join(f"{float(x):.2f}" for x in rf["refresh_med_s"]))


if __name__ == "__main__":
    main()
