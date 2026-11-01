#!/usr/bin/env python3
"""Run paper market instances and write result shards."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _k in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market.contam import obs_dirty_mask  # noqa: E402
from market.floor import append_entry_seller, append_floor_purchases  # noqa: E402
SEED_BASE = 20261019
K_FIX, RHO_FIX, B_DEFAULT, N_SEED = 3, 7.0, 20.0, 100
from market.build import build_market  # noqa: E402
from market.const import PKT_ROWS  # noqa: E402
from market.data import _get_bundle  # noqa: E402
from buyer.acquisition import (  # noqa: E402
    PAPER_POLICIES_BASE,
    PAPER_POLICIES_CIFAR,
    PAPER_POLICIES_ENTRY,
    FLOOR_F,
    FUND_L,
    POLICY_ID,
    buyer_pi_hats,
    draw_buyer_task,
    export_record_tables,
    run_one_buyer,
    sample_ts_plan_buyer,
    ucb_plan_buyer,
    visible_market,
)
from shared_evidence.cifar_workers import (  # noqa: E402
    SIDE_INFO,
    acc_quartile_cuts,
    append_floor_purchases_cifar,
    build_stratified_worker_market,
    leftover_rows,
    load_annotations,
    worker_stats,
)
from shared_evidence.evidence import BuyerEvidenceStore, EvidenceStore  # noqa: E402
from shared_evidence.parallel import choose_parallelism, pin_cpu_threads  # noqa: E402
from shared_evidence.report_market import write_report  # noqa: E402

OUT_DIR = ROOT / "results" / "market"
ACQ_LOCK = {
    "amazon": dict(beta=0.35, s_rows=20, sampling_rule="value"),
    "domainnet": dict(beta=0.25, s_rows=10, sampling_rule="value"),
    "cifar10n": dict(beta=0.25, s_rows=5, sampling_rule="uniform"),
}
DS_ID = {"domainnet": 1, "amazon": 2, "cifar10n": 3}
PHASE_TEST = 2
BUYER_SALT = 1001
ENTRY_SALT = 1002
TS_PLAN_SALT = 1004
CIFAR_STRAT_SALT = 1005
FLOOR_SALT = 1006
ENTRY_AT = 15
N_BUYERS = 30
FUND_L_MAX = 80
FLOOR_F_MAX = 80
BASE_INST0 = 230
ENTRY_INST0 = 240
CIFAR_INST0 = 220
N_BASE = 20
N_ENTRY = 20
N_CIFAR = 10
ROW_POLICIES = (
    "indep", "share_naive", "share_ts",
    "floor40_row", "floor80_row",
)
CAL_PRIOR = {
    "share_agg": "eb", "share_agg_ts": "eb",
    "share_agg_ucb": "eb",
    "fund80_agg": "eb",
    "floor40_agg": "eb", "floor80_agg": "eb",
}


def _cfg(dataset: str, inst: int) -> dict:
    lock = ACQ_LOCK[dataset]
    return dict(
        dataset=dataset, beta=lock["beta"], s_rows=lock["s_rows"],
        B=B_DEFAULT, K=K_FIX, rho=RHO_FIX, n_seed=N_SEED, inst=int(inst),
        kappa=1.0, sampling_rule=lock["sampling_rule"],
        cifar=dataset == "cifar10n",
    )


def _market_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst), PHASE_TEST]
    )


def _buyer_rng(dataset, s_rows, B, inst, buyer_t):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, int(buyer_t), BUYER_SALT]
    )


def _entry_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, ENTRY_SALT]
    )


def _floor_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, FLOOR_SALT]
    )


def _cifar_strat_rng(s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID["cifar10n"], int(s_rows), int(B), int(inst),
         PHASE_TEST, CIFAR_STRAT_SALT]
    )


def _plan_rng(dataset, s_rows, B, inst, buyer_t, policy):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, int(buyer_t), TS_PLAN_SALT, POLICY_ID.get(policy, 9)]
    )


def _n_extra_small(s_rows: int) -> int:
    return int(math.ceil(FLOOR_F_MAX / float(s_rows)))


def _dump_row_gamma(store, vis, dataset, market_kind, inst, policy):
    rows = []
    offer_id = np.asarray(vis["offer_id"], int)
    seller_ids = np.asarray(vis["seller_ids"], int)
    loc = {int(oid): i for i, oid in enumerate(offer_id)}
    qs = np.asarray(vis["qs"], float)
    ys, ys_clean, contams = vis["ys"], vis.get("ys_clean"), vis.get("contams")
    for rid, n in store.row_n.items():
        oid, li = int(rid[0]), int(rid[1])
        j = loc.get(oid)
        if j is None:
            continue
        dirty = np.nan
        if ys_clean is not None and ys_clean[j] is not None:
            m = obs_dirty_mask(ys[j], ys_clean[j])
            if 0 <= li < len(m):
                dirty = bool(m[li])
        elif contams is not None and contams[j] is not None and 0 <= li < len(contams[j]):
            dirty = bool(contams[j][li])
        rows.append(dict(
            dataset=dataset, market=market_kind, instance=int(inst),
            policy=policy, offer_id=oid, local_i=li,
            seller=int(seller_ids[j]),
            gamma_mean=float(store.row_mean(rid)),
            n_buyers=int(n),
            dirty=dirty,
            true_pi=float(1.0 - qs[j]),
        ))
    return rows


def _fund_idx(vis, market, policy, t, s_rows):
    if int(t) != ENTRY_AT or policy not in FUND_L:
        return None
    L = FUND_L[policy]
    n = int(math.ceil(L / float(s_rows)))
    oids = list(market.get("entry_fund_offer_ids") or [])[:n]
    loc = {int(o): i for i, o in enumerate(np.asarray(vis["offer_id"], int))}
    idx = [loc[int(o)] for o in oids if int(o) in loc]
    return idx or None


def _floor_idx(vis, market, policy, ej, used_oids, s_rows):
    if policy not in FLOOR_F:
        return None
    F = FLOOR_F[policy]
    loc = {int(o): i for i, o in enumerate(np.asarray(vis["offer_id"], int))}
    by_s = market.get("floor_offer_ids") or {}
    idx = []
    vis_sids = sorted({int(s) for s in vis["seller_ids"]})
    for s in vis_sids:
        have = len(ej.get(int(s), set()))
        need = int(F) - have
        if need <= 0:
            continue
        n = int(math.ceil(need / float(s_rows)))
        oids = [int(o) for o in by_s.get(int(s), []) if int(o) not in used_oids]
        take = oids[:n]
        for o in take:
            used_oids.add(int(o))
            if int(o) in loc:
                idx.append(loc[int(o)])
    return idx or None


def run_market_instance(bundle, dataset, market_kind, inst, n_buyers=N_BUYERS,
                       policies=None, ann=None, quartile_cuts=None,
                       export_records=False):
    cfg = _cfg(dataset, inst)
    cfg.update(lam=bundle["lam"], sigma2=bundle["sigma2"], d=bundle["d"])
    if policies is None:
        policies = list(PAPER_POLICIES_ENTRY if market_kind == "entry" else PAPER_POLICIES_BASE)
    t0 = time.time()
    seed_pool = None
    backfill = []
    seller_qbin = {}
    n_extra = _n_extra_small(cfg["s_rows"])
    if dataset == "cifar10n":
        groups = bundle["groups"]
        w_rng = _cifar_strat_rng(cfg["s_rows"], cfg["B"], inst)
        market, seed_pool = build_stratified_worker_market(
            w_rng, ann, cfg["s_rows"], quartile_cuts,
        )
        market = append_floor_purchases_cifar(
            market, _floor_rng(dataset, cfg["s_rows"], cfg["B"], inst),
            ann, cfg["s_rows"], n_extra,
        )
        cfg["oracle_sources"] = market["oracle_sources"]
        backfill = market.get("backfill") or []
        seller_qbin = {int(k): int(v) for k, v in (market.get("seller_qbin") or {}).items()}
    else:
        groups = bundle["groups"]
        m_rng = _market_rng(dataset, cfg["s_rows"], cfg["B"], inst)
        market = build_market(m_rng, groups, dataset, cfg["s_rows"], n_sellers=8)
        if market_kind == "entry":
            e_rng = _entry_rng(dataset, cfg["s_rows"], cfg["B"], inst)
            names = list(groups.keys())
            g = str(names[int(e_rng.integers(len(names)))])
            market = append_entry_seller(
                market, e_rng, groups, dataset, cfg["s_rows"], g,
                appear_at=ENTRY_AT, q=0.0,
                n_extra_small=n_extra,
            )
        market = append_floor_purchases(
            market, _floor_rng(dataset, cfg["s_rows"], cfg["B"], inst),
            groups, dataset, cfg["s_rows"], n_extra,
        )
    seller_of = {int(oid): int(s) for oid, s in zip(market["offer_id"], market["seller_ids"])}
    src_of = {int(s): str(g) for s, g in zip(market["seller_ids"], market["sources"])}
    row_stores = {p: EvidenceStore(seller_of) for p in policies if p in ROW_POLICIES}
    buyer_stores = {p: BuyerEvidenceStore(prior=CAL_PRIOR[p]) for p in policies if p in CAL_PRIOR}
    ej = {p: {} for p in policies}
    floor_used = {p: set() for p in policies}
    buyer_rows, pay_rows, post_rows, gamma_rows, plat_rows = [], [], [], [], []
    rec_rows, detail_rows, pi_hat_rows = [], [], []
    funded_oids = set()
    if export_records:
        for vs in (market.get("floor_offer_ids") or {}).values():
            funded_oids.update(int(x) for x in vs)
        funded_oids.update(int(x) for x in (market.get("entry_fund_offer_ids") or []))
    t_first_acq = None
    all_sids = sorted({int(x) for x in market["seller_ids"]})
    for t in range(1, n_buyers + 1):
        b_rng = _buyer_rng(dataset, cfg["s_rows"], cfg["B"], inst, t)
        demand, Sigmas, Sigma_pre, M0, X0, y0, X_mix, y_mix, yid_mix = draw_buyer_task(
            b_rng, groups, market, cfg["n_seed"], cfg["d"], cfg["lam"], cfg["K"],
            cifar=dataset == "cifar10n", seed_pool=seed_pool,
        )
        vis = visible_market(market, t)
        if dataset == "cifar10n":
            vis["oracle_sources"] = market["oracle_sources"]
        vis_sids = sorted({int(s) for s in vis["seller_ids"]})
        for policy in policies:
            pol_rng = np.random.default_rng(
                [SEED_BASE, DS_ID[dataset], int(cfg["s_rows"]), int(cfg["B"]),
                 int(inst), PHASE_TEST, int(t), BUYER_SALT, POLICY_ID.get(policy, 9)]
            )
            store = row_stores.get(policy)
            bstore = buyer_stores.get(policy)
            plan_pi = None
            if policy == "share_agg_ts" and bstore is not None:
                plan_pi = sample_ts_plan_buyer(
                    bstore, vis_sids, _plan_rng(dataset, cfg["s_rows"], cfg["B"], inst, t, policy),
                )
            elif policy == "share_agg_ucb" and bstore is not None:
                plan_pi = ucb_plan_buyer(bstore, vis_sids)
            plat_idx = _fund_idx(vis, market, policy, t, cfg["s_rows"])
            if plat_idx is None:
                plat_idx = _floor_idx(
                    vis, market, policy, ej[policy], floor_used[policy], cfg["s_rows"],
                )
            t_pol = time.time()
            skip_s = None
            if plat_idx:
                skip_s = {int(vis["seller_ids"][j]) for j in plat_idx}
            summary, payments, gamma_by_row, _pkt = run_one_buyer(
                policy, vis, groups, demand, Sigmas, Sigma_pre,
                M0, X0, y0, X_mix, y_mix, yid_mix, cfg, pol_rng, store,
                buyer_store=bstore, plan_pi=plan_pi,
                exhaust_budget=True,
                platform_small_idx=plat_idx,
                extra_skip_sellers=skip_s,
            )
            dt = time.time() - t_pol
            if t_first_acq is None and policy == "indep":
                t_first_acq = dt
            if store is not None and gamma_by_row:
                store.update(gamma_by_row)
            if bstore is not None and gamma_by_row:
                for s, (ph, nr) in buyer_pi_hats(gamma_by_row, vis).items():
                    bstore.update_buyer(s, ph, nr, t)
            for rid in gamma_by_row:
                s = seller_of.get(int(rid[0]))
                if s is None:
                    continue
                ej[policy].setdefault(int(s), set()).add(rid)
            buyer_rows.append(dict(
                dataset=dataset, market=market_kind, instance=int(inst),
                buyer_t=int(t), policy=policy,
                excess=summary["excess"],
                test_loss=summary.get("test_loss", np.nan),
                small_purchase_spend=summary["small_purchase_spend"],
                packet_spend=summary["packet_spend"],
                n_checked=summary["n_checked"],
                platform_paid=float(summary.get("platform_paid") or 0.0),
                seconds=dt,
            ))
            by_s = summary.get("platform_by_seller") or {}
            if by_s:
                for s, rec in by_s.items():
                    plat_rows.append(dict(
                        dataset=dataset, market=market_kind, instance=int(inst),
                        buyer_t=int(t), policy=policy, seller=int(s),
                        platform_paid=float(rec["paid"]),
                        platform_rows=int(rec["rows"]),
                        L=FUND_L.get(policy) or FLOOR_F.get(policy),
                    ))
            elif summary.get("platform_paid"):
                plat_rows.append(dict(
                    dataset=dataset, market=market_kind, instance=int(inst),
                    buyer_t=int(t), policy=policy,
                    seller=summary.get("platform_seller"),
                    platform_paid=float(summary["platform_paid"]),
                    platform_rows=int(summary.get("platform_rows") or 0),
                    L=FUND_L.get(policy) or FLOOR_F.get(policy),
                ))
            for rec in payments:
                s = int(rec["seller"])
                pay_rows.append(dict(
                    dataset=dataset, market=market_kind, instance=int(inst),
                    buyer_t=int(t), policy=policy,
                    source=src_of.get(s, ""),
                    qbin=seller_qbin.get(s, ""),
                    appear_at=int(market["appear_at"][market["seller_ids"] == s][0]),
                    **rec,
                ))
            if store is not None:
                for s in all_sids:
                    a, b, mean, n = store.seller_posterior(s)
                    q = float(market["qs"][market["seller_ids"] == s][0])
                    post_rows.append(dict(
                        dataset=dataset, market=market_kind, instance=int(inst),
                        buyer_t=int(t), policy=policy, seller=int(s),
                        source=src_of.get(int(s), ""),
                        true_pi=float(1.0 - q),
                        post_mean=float(mean), post_a=float(a), post_b=float(b),
                        n_distinct=int(n), n_rows_ev=int(n),
                        pub_mean=float(mean), pub_std=float(store.beta_std(s)),
                        n_buyers=np.nan, mu0=np.nan, tau0_2=np.nan, sigma2=np.nan,
                        store_kind="row",
                        appear_at=int(market["appear_at"][market["seller_ids"] == s][0]),
                        qbin=seller_qbin.get(int(s), ""),
                    ))
            if bstore is not None:
                for s in all_sids:
                    m, v, n, mu0, tau0_2, sig2 = bstore.publish(s)
                    q = float(market["qs"][market["seller_ids"] == s][0])
                    n_ej = len(ej[policy].get(int(s), set()))
                    post_rows.append(dict(
                        dataset=dataset, market=market_kind, instance=int(inst),
                        buyer_t=int(t), policy=policy, seller=int(s),
                        source=src_of.get(int(s), ""),
                        true_pi=float(1.0 - q),
                        post_mean=float(m), post_a=np.nan, post_b=np.nan,
                        n_distinct=int(n_ej), n_rows_ev=int(n_ej),
                        pub_mean=float(np.clip(m, 0.05, 1.0)),
                        pub_std=float(v),
                        n_buyers=int(n), mu0=float(mu0), tau0_2=float(tau0_2),
                        sigma2=float(sig2),
                        store_kind="buyer",
                        appear_at=int(market["appear_at"][market["seller_ids"] == s][0]),
                        qbin=seller_qbin.get(int(s), ""),
                    ))
            if t == n_buyers and store is not None:
                gamma_rows.extend(_dump_row_gamma(
                    store, vis, dataset, market_kind, inst, policy,
                ))
            if export_records and gamma_by_row:
                recs, details = export_record_tables(
                    vis, gamma_by_row, demand, dataset, market_kind,
                    inst, policy, t, funded_oids,
                )
                rec_rows.extend(recs)
                detail_rows.extend(details)
            if export_records:
                pi_s = summary.get("pi_small") or {}
                pi_f = summary.get("pi_final") or {}
                qs = np.asarray(vis["qs"], float)
                sids = np.asarray(vis["seller_ids"], int)
                for s in sorted(set(int(x) for x in vis_sids) | set(pi_s) | set(pi_f)):
                    hit = sids == int(s)
                    qv = float(qs[hit][0]) if hit.any() else np.nan
                    pi_hat_rows.append(dict(
                        dataset=dataset, market=market_kind, instance=int(inst),
                        policy=policy, buyer_t=int(t), seller=int(s),
                        true_pi=float(1.0 - qv) if np.isfinite(qv) else np.nan,
                        pi_small=float(pi_s[int(s)]) if int(s) in pi_s else np.nan,
                        pi_final=float(pi_f[int(s)]) if int(s) in pi_f else np.nan,
                        did_small=int(int(s) in set(int(x) for x in (summary.get("small_sids") or []))),
                    ))
        extra = f" acq0={t_first_acq:.1f}s" if t == 1 and t_first_acq else ""
        print(f"  {dataset}/{market_kind}/inst{inst} buyer {t}/{n_buyers}{extra}", flush=True)
    meta = dict(
        dataset=dataset, market=market_kind, instance=int(inst),
        n_buyers=n_buyers, seconds=time.time() - t0,
        t_first_acq=t_first_acq,
        n_sellers=int(len(np.unique(market["seller_ids"]))),
        entry_seller_id=market.get("entry_seller_id"),
        entry_group=market.get("entry_group"),
        n_extra_small=n_extra,
        small_cost=float(cfg["s_rows"]) / float(PKT_ROWS),
        sampling_rule=cfg["sampling_rule"],
        chosen_workers=market.get("chosen_workers"),
        seller_qbin=seller_qbin,
        backfill=backfill,
        quartile_cuts=list(market.get("quartile_cuts") or []) if dataset == "cifar10n" else [],
        floor_reuse_n=int(market.get("floor_reuse_n") or 0),
    )
    out = (
        pd.DataFrame(buyer_rows), pd.DataFrame(pay_rows),
        pd.DataFrame(post_rows),
        pd.DataFrame(gamma_rows) if gamma_rows else pd.DataFrame(),
        meta,
        pd.DataFrame(plat_rows) if plat_rows else pd.DataFrame(),
    )
    if export_records:
        return out + (
            pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(),
            pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(),
            pd.DataFrame(pi_hat_rows) if pi_hat_rows else pd.DataFrame(),
        )
    return out


def _job_worker(wid, shard, out_dir: str, quartile_cuts):
    pin_cpu_threads(1)
    cache = {}
    ann = None
    out = Path(out_dir)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cuts = tuple(quartile_cuts) if quartile_cuts else None
    for job in shard:
        tag = f"{job['dataset']}_{job['market']}_{job['inst']}"
        done = shard_dir / f"{tag}.done"
        if done.exists():
            print(f"[W{wid}] skip {tag}", flush=True)
            continue
        if job["dataset"] == "cifar10n" and ann is None:
            ann = load_annotations()
        bundle = _get_bundle(cache, job["dataset"])
        print(f"[W{wid}] {tag} buyers={job.get('n_buyers', N_BUYERS)}", flush=True)
        t0 = time.time()
        try:
            ret = run_market_instance(
                bundle, job["dataset"], job["market"], job["inst"],
                n_buyers=int(job.get("n_buyers", N_BUYERS)),
                policies=job.get("policies"), ann=ann, quartile_cuts=cuts,
                export_records=bool(job.get("export_records")),
            )
            if job.get("export_records"):
                buyers, pays, posts, gammas, meta, plats, recs, details, pihats = ret
            else:
                buyers, pays, posts, gammas, meta, plats = ret
                recs = details = pihats = None
        except Exception as exc:
            (shard_dir / f"{tag}.err").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"[W{wid}] error {tag} {exc}", flush=True)
            continue
        buyers.to_csv(shard_dir / f"{tag}_buyers.csv", index=False)
        pays.to_csv(shard_dir / f"{tag}_pays.csv", index=False)
        posts.to_csv(shard_dir / f"{tag}_posts.csv", index=False)
        if len(gammas):
            gammas.to_csv(shard_dir / f"{tag}_gamma.csv", index=False)
        if len(plats):
            plats.to_csv(shard_dir / f"{tag}_plat.csv", index=False)
        if job.get("export_records"):
            recs.to_csv(shard_dir / f"{tag}_records.csv", index=False)
            details.to_csv(shard_dir / f"{tag}_record_rows.csv.gz", index=False,
                           compression="gzip")
            if pihats is not None and len(pihats):
                pihats.to_csv(shard_dir / f"{tag}_pi_hats.csv", index=False)
        (shard_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        done.write_text("ok\n")
        print(f"[W{wid}] done {tag} {time.time() - t0:.1f}s", flush=True)


def _run_instance_jobs(jobs, out_dir: Path, nw: int, quartile_cuts):
    if not jobs:
        return
    nw = max(1, min(int(nw), len(jobs)))
    shards = [jobs[i::nw] for i in range(nw)]
    ctx = mp.get_context("spawn")
    procs = []
    cuts = list(quartile_cuts) if quartile_cuts else None
    for w in range(nw):
        if not shards[w]:
            continue
        p = ctx.Process(target=_job_worker, args=(w, shards[w], str(out_dir), cuts))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode:
            raise SystemExit(f"worker failed {p.exitcode}")


def concat_shards(out_dir: Path):
    shard_dir = out_dir / "shards"

    def _cat(glob):
        ps = sorted(shard_dir.glob(glob))
        return pd.concat([pd.read_csv(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()

    buyers, pays, posts, gammas, plats = (
        _cat("*_buyers.csv"), _cat("*_pays.csv"),
        _cat("*_posts.csv"), _cat("*_gamma.csv"), _cat("*_plat.csv"),
    )
    recs = _cat("*_records.csv")
    if len(recs):
        recs.to_csv(out_dir / "records.csv", index=False)
    rrows = _cat("*_record_rows.csv.gz")
    if len(rrows):
        rrows.to_csv(out_dir / "record_rows.csv.gz", index=False, compression="gzip")
    pihats = _cat("*_pi_hats.csv")
    if len(pihats):
        pihats.to_csv(out_dir / "pi_hats.csv", index=False)
    metas = []
    for p in sorted(shard_dir.glob("*_meta.json")):
        rec = json.loads(p.read_text())
        rec.pop("floor_reuse_n", None)
        metas.append(rec)
    if len(buyers):
        from shared_evidence.report_market import slim_buyers
        slim_buyers(buyers).to_csv(out_dir / "per_buyer.csv", index=False)
    if len(pays):
        pays.to_csv(out_dir / "payments.csv", index=False)
    if len(posts):
        posts.to_csv(out_dir / "posterior.csv", index=False)
    if len(gammas):
        gammas.to_csv(out_dir / "row_gamma.csv", index=False)
    if len(plats):
        plats.to_csv(out_dir / "platform_cost.csv", index=False)
    (out_dir / "run_meta.json").write_text(json.dumps(metas, indent=2) + "\n")
    return buyers, pays, posts, gammas, metas


def write_inventory(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    leftover = {}
    skip = {}
    extra_floor = 8 * FLOOR_F_MAX
    for ds, s in (("amazon", 20), ("domainnet", 10)):
        bundle = _get_bundle({}, ds)
        df = leftover_rows(
            bundle["groups"], s,
            extra_entry_rows=FUND_L_MAX,
            extra_floor_rows=extra_floor,
        )
        df.to_csv(out_dir / f"leftover_{ds}.csv", index=False)
        leftover[ds] = dict(
            min_leftover=int(df.leftover.min()),
            need_entry=int(df.need_entry.iloc[0]),
            extra_floor_rows=extra_floor,
            min_after_floor=int(df.leftover_after_floor.min()),
            all_enough=bool(df.enough.all()),
            rows=df.to_dict(orient="records"),
        )
        skip[ds] = []
    wst = worker_stats()
    qinfo = {}
    if SIDE_INFO.exists():
        ann = load_annotations()
        raw = acc_quartile_cuts(ann)
        qinfo = {k: raw[k] for k in raw if k != "workers"}
    (out_dir / "cifar_worker_stats.json").write_text(json.dumps(wst, indent=2) + "\n")
    (out_dir / "cifar_quartiles.json").write_text(json.dumps(qinfo, indent=2) + "\n")
    info = dict(
        leftover=leftover,
        skip_entry=skip,
        domainnet_entry=leftover["domainnet"]["all_enough"],
        amazon_entry=leftover["amazon"]["all_enough"],
        cifar_workers=wst,
        cifar_quartiles=qinfo,
        side_info_present=SIDE_INFO.exists(),
        note="floor80 may exhaust leftover; replacement, no skip.",
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    (out_dir / "inventory.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print("INVENTORY", json.dumps(
        {k: info[k] for k in ("domainnet_entry", "amazon_entry", "side_info_present")},
    ), flush=True)
    return info


def plan_jobs(inv: dict, n_cifar: int) -> list[dict]:
    jobs = []
    for ds in ("amazon", "domainnet"):
        for i in range(N_BASE):
            jobs.append(dict(dataset=ds, market="base", inst=BASE_INST0 + i,
                             policies=list(PAPER_POLICIES_BASE), export_records=True))
        for i in range(N_ENTRY):
            jobs.append(dict(dataset=ds, market="entry", inst=ENTRY_INST0 + i,
                             policies=list(PAPER_POLICIES_ENTRY), export_records=True))
    qn = (inv.get("cifar_quartiles") or {}).get("n_workers", 0)
    if inv.get("side_info_present") and qn >= 24:
        for i in range(n_cifar):
            jobs.append(dict(dataset="cifar10n", market="annotator_strat",
                             inst=CIFAR_INST0 + i, policies=list(PAPER_POLICIES_CIFAR),
                             export_records=True))
    return jobs


def run_full(out_dir: Path, nw: int, jobs: list[dict], quartile_cuts):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / "shards"
    pending = [
        j for j in jobs
        if not (shard / f"{j['dataset']}_{j['market']}_{j['inst']}.done").exists()
    ]
    print(f"RUN jobs={len(jobs)} pending={len(pending)} nw={nw}", flush=True)
    t0 = time.time()
    _run_instance_jobs(pending, out_dir, nw, quartile_cuts)
    buyers, pays, posts, gammas, metas = concat_shards(out_dir)
    info = dict(
        seconds=time.time() - t0,
        n_jobs=len(jobs),
        n_pending_at_start=len(pending),
        n_buyer_rows=int(len(buyers)),
    )
    (out_dir / "run_timing.json").write_text(json.dumps(info, indent=2) + "\n")
    print("RUN", info, flush=True)
    return buyers, pays, posts, gammas, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["inventory", "run", "report", "all"])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--nw", type=int, default=None)
    ap.add_argument("--n-cifar", type=int, default=N_CIFAR)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dataset", default="amazon")
    ap.add_argument("--market", default="base")
    ap.add_argument("--instance", type=int, default=230)
    ap.add_argument("--n-buyers", type=int, default=3)
    ap.add_argument("--policies", default="share_agg,floor80_agg")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pin_cpu_threads(1)
    plan = choose_parallelism("market")
    nw = args.nw or plan.n_workers
    print(
        f"parallel device={plan.device} nw={nw} threads/worker={plan.threads_per_worker} "
        f"cpu={plan.n_cpu} gpu={plan.n_gpu} | {plan.reason}",
        flush=True,
    )
    inv = None
    if args.quick:
        bundle = _get_bundle({}, args.dataset)
        pols = [x.strip() for x in args.policies.split(",") if x.strip()]
        buyers, pays, posts, _g, meta, _plats, recs, details, pihats = run_market_instance(
            bundle, args.dataset, args.market, int(args.instance),
            n_buyers=int(args.n_buyers), policies=pols,
            export_records=True,
        )
        from shared_evidence.report_market import slim_buyers
        slim_buyers(buyers).to_csv(out_dir / "per_buyer.csv", index=False)
        pays.to_csv(out_dir / "payments.csv", index=False)
        if len(posts):
            posts.to_csv(out_dir / "posterior.csv", index=False)
        if len(recs):
            recs.to_csv(out_dir / "records.csv", index=False)
        if len(details):
            details.to_csv(out_dir / "record_rows.csv.gz", index=False, compression="gzip")
        if pihats is not None and len(pihats):
            pihats.to_csv(out_dir / "pi_hats.csv", index=False)
        (out_dir / "run_meta.json").write_text(
            json.dumps({k: meta.get(k) for k in meta}, default=str, indent=2) + "\n"
        )
        print(buyers.to_string(index=False), flush=True)
        return
    if args.stage in ("inventory", "run", "all"):
        inv = write_inventory(out_dir)
    if args.stage in ("run", "all"):
        if inv is None:
            inv = json.loads((out_dir / "inventory.json").read_text())
        jobs = plan_jobs(inv, int(args.n_cifar))
        (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        cuts = (inv.get("cifar_quartiles") or {}).get("cuts")
        run_full(out_dir, nw, jobs, cuts)
    if args.stage in ("report", "all"):
        concat_shards(out_dir)
        write_report(out_dir)


if __name__ == "__main__":
    main()
