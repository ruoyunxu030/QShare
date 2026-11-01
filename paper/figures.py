"""Figures 3 to 5 of the paper, one PDF per panel plus one legend strip per figure."""
import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MKT = ROOT / "results" / "market"
STO = ROOT / "results" / "store"
OUT = ROOT / "paper" / "out"


def _use_libertine():
    """Use Linux Libertine from TeX Live when it is installed, as in the paper."""
    family = "serif"
    if shutil.which("kpsewhich"):
        for f in ["LinLibertine_R.otf", "LinLibertine_RB.otf", "LinLibertine_RI.otf"]:
            p = subprocess.run(["kpsewhich", f], capture_output=True, text=True).stdout.strip()
            if p:
                fm.fontManager.addfont(p)
                family = "Linux Libertine O"
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.cal": family,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    })


C = {"indep": "#7f8b96", "share_naive": "#d4a72c", "share_agg": "#86c5bc",
     "floor80_agg": "#2f8f83", "fund80_agg": "#d8863a"}
L = {"indep": "Independent", "share_naive": "Row-level sharing", "share_agg": "QShare without floor",
     "floor80_agg": "QShare", "fund80_agg": "Entry subsidy"}
MK = {"indep": "o", "share_naive": "s", "share_agg": "^", "floor80_agg": "D"}
P4 = (1.72, 1.42)  # four panels across the text width
P3 = (2.3, 1.42)   # three panels across the text width
FIG4 = (165.6 / 72.0, 128.0 / 72.0)  # 165.6 pt wide; 128 pt so the y-label fits


def _panel(size):
    return plt.subplots(figsize=size, constrained_layout=True)


def _save(fig, name, extra_dirs=()):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    for d in extra_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{name}.pdf")
    plt.close(fig)


def _legend(name, handles, labels, ncol, extra_dirs=()):
    fig = plt.figure(figsize=(7.0, 0.22))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False, handlelength=2.2, columnspacing=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    for d in extra_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _handle(p, ls="-", marker=None):
    return Line2D([0], [0], color=C[p], ls=ls, marker=marker, lw=1.3, markersize=3.5)


def fig3():
    """Value of sharing on the base markets: loss reduction by arrival, cumulative evidence spend."""
    pb = pd.read_csv(MKT / "per_buyer.csv")
    for k, ds in enumerate(["amazon", "domainnet"]):
        x = pb[(pb.dataset == ds) & (pb.market == "base")]
        x = x.assign(bin=((x.buyer_t - 1) // 5) * 5 + 3)
        m = x.groupby(["policy", "instance", "bin"]).excess.mean().unstack("policy")
        fig, a = _panel(P4)
        for p in ["share_naive", "share_agg"]:
            g = (100 * (1 - m[p] / m["indep"])).groupby("bin").mean()
            a.plot(g.index, g.values, marker=MK[p], color=C[p])
        a.axhline(0, color=C["indep"], lw=0.9, ls="--")
        a.set_ylabel("Loss reduction (%)")
        a.set_xticks([3, 13, 23])
        a.set_xticklabels(["1–5", "11–15", "21–25"])
        a.set_xlabel("Buyer arrival")
        _save(fig, f"fig3_{'ab'[k]}")
        sp = (x.groupby(["policy", "buyer_t"])[["small_purchase_spend", "platform_paid"]]
              .mean().sum(axis=1).unstack("policy").cumsum())
        fig, b = _panel(P4)
        for p in ["indep", "share_agg", "floor80_agg"]:
            b.plot(sp.index, sp[p].values, color=C[p], ls="--" if p == "indep" else "-")
        b.set_xlabel("Buyer arrival")
        b.set_ylabel("Evidence spend\n(packet prices)")
        _save(fig, f"fig3_{'cd'[k]}")
    _legend("fig3_legend",
            [_handle("indep", "--"), _handle("share_naive", marker="s"), _handle("share_agg", marker="^"),
             _handle("floor80_agg")],
            [L["indep"], L["share_naive"], L["share_agg"], L["floor80_agg"]], 4)


def fig5():
    """Lock-out and entry over time, with bootstrap bands over market instances."""
    fo = pd.read_csv(MKT / "first_order.csv")
    en = pd.read_csv(MKT / "entry_newcomer.csv")
    rng = np.random.default_rng(0)  # one stream for all bands, in the order drawn below

    def band(mat):
        mat = np.asarray(mat, float)
        n = mat.shape[0]
        bs = mat[rng.integers(0, n, (2000, n))].mean(1)
        return mat.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)

    T = np.arange(1, 31)
    for k, ds in enumerate(["amazon", "domainnet"]):
        x = fo[(fo.dataset == ds) & (fo.market == "base") & (fo.true_pi == 1.0)]
        fig, a = _panel(P4)
        for p in ["indep", "share_naive", "share_agg", "floor80_agg"]:
            first = x[x.policy == p].set_index(["instance", "seller"]).first_order_t
            mat = [[100 * (g <= t).mean() for t in T] for _, g in first.groupby(level=0)]
            mm, lo, hi = band(mat)
            a.plot(T, mm, color=C[p], ls="--" if p == "indep" else "-")
            a.fill_between(T, lo, hi, color=C[p], alpha=0.15, lw=0)
        a.set_xlabel("Buyer arrival")
        a.set_ylim(0, 102)
        a.set_xticks([1, 10, 20, 30])
        a.set_ylabel("Best sellers with\nan order (%)")
        _save(fig, f"fig5_{'ab'[k]}")
        e = en[en.dataset == ds]
        T2 = np.arange(15, 31)
        fig, b = _panel(P4)
        for p in ["indep", "share_naive", "share_agg", "fund80_agg", "floor80_agg"]:
            f = e[e.policy == p].first_packet_t.values
            mat = [[100 * float((not np.isnan(v)) and v <= t) for t in T2] for v in f]
            mm, lo, hi = band(mat)
            b.step(T2, mm, where="post", color=C[p], ls="--" if p == "indep" else "-")
        b.set_xlabel("Buyer arrival")
        b.set_ylim(0, 102)
        b.set_xticks([15, 20, 25, 30])
        b.set_ylabel("New seller has\nsold (%)")
        _save(fig, f"fig5_{'cd'[k]}")
    _legend("fig5_legend",
            [_handle("indep", "--"), _handle("share_naive"), _handle("share_agg"), _handle("fund80_agg"),
             _handle("floor80_agg")],
            [L["indep"], L["share_naive"], L["share_agg"], L["fund80_agg"], L["floor80_agg"]], 5)


def fig4():
    """Evidence store performance against the number of sellers."""
    Ns = [1000, 10000, 100000]
    names = ["Row log", "Row aggregates", "Record log", "QShare"]
    SC = {
        "Row log": "#c4cad0",
        "Row aggregates": "#7f8b96",
        "Record log": "#d4a72c",
        "QShare": "#2f8f83",
    }
    SM = {
        "Row log": "o",
        "Row aggregates": "s",
        "Record log": "^",
        "QShare": "D",
    }
    df = pd.read_csv(STO / "stores_by_n.csv").set_index(["name", "n_sellers"])
    vals = {}
    for name in names:
        w, q, b = [], [], []
        for n in Ns:
            r = df.loc[(name, n)]
            w.append(1e3 * float(r["write_s"]))
            q.append(1e3 * float(r["q2_median_s"]))
            b.append(float(r["bytes"]) / 1e6)
        vals[name] = (w, q, b)
    ylab = ["Write per buyer (ms)", "Query latency (ms)", "Storage (MB)"]
    for i in range(3):
        fig, a = plt.subplots(figsize=FIG4)
        for name in names:
            a.plot(Ns, np.asarray(vals[name][i]), marker=SM[name], color=SC[name])
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylabel(ylab[i])
        a.set_xlabel("Number of sellers $N$")
        a.set_xticks(Ns)
        fig.subplots_adjust(left=0.30, right=0.98, bottom=0.26, top=0.80)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"fig4_{'abc'[i]}.pdf")
        plt.close(fig)
    _legend(
        "fig4_legend",
        [Line2D([0], [0], color=SC[k], marker=SM[k], lw=1.3, markersize=3.5) for k in names],
        names,
        4,
    )


def main():
    argparse.ArgumentParser(description="Write Figures 3 to 5 of the paper to paper/out/.").parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _use_libertine()
    fig3()
    fig4()
    fig5()


if __name__ == "__main__":
    main()
