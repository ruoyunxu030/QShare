# QShare

QShare is a quality evidence service for online data markets. It shares buyers' quality evidence as one record per buyer and seller, keeps threshold queries over the published seller estimates exact as purchases arrive, and funds small purchases for sellers with too little evidence to reduce seller lock-out.

## Setup

Python 3.11.

```bash
pixi install          # or: python -m pip install -r requirements.txt
```

Prepared data go to `data_real/` (set `QSHARE_DATA` to use another path). Stages marked GPU need a GPU.

## 1. Prepare data

```bash
# Amazon Reviews 2023 (BGE embeddings)
pixi run python -m experiments.real.prepare_amazon --stage download
pixi run python -m experiments.real.prepare_amazon --stage preprocess
pixi run python -m experiments.real.prepare_amazon --stage embed        # GPU
pixi run python -m experiments.real.prepare_amazon --stage pca
pixi run python -m experiments.real.prepare_amazon --stage split
pixi run python -m experiments.real.prepare_meta --dataset amazon

# DomainNet (CLIP embeddings)
pixi run python -m experiments.real.prepare_domainnet --stage download
pixi run python -m experiments.real.prepare_domainnet --stage extract
pixi run python -m experiments.real.prepare_domainnet --stage select
pixi run python -m experiments.real.prepare_domainnet --stage embed_clip   # GPU
pixi run python -m experiments.real.prepare_domainnet --stage pca
pixi run python -m experiments.real.prepare_meta --dataset domainnet

# CIFAR-10N (real annotator labels)
pixi run python -m experiments.real.prepare_cifar10n --stage download
pixi run python -m experiments.real.prepare_cifar10n --stage embed      # GPU
pixi run python -m experiments.real.prepare_cifar10n --stage features
```

## 2. Run the markets

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# quick check: one Amazon instance, 3 buyers
pixi run python -m shared_evidence.run_market --quick --dataset amazon --market base \
    --instance 230 --n-buyers 3 --policies indep,share_agg,floor80_agg --out results/quick

# full paper run (about 3.6 h with 70 processes)
pixi run python -m shared_evidence.run_market --stage run --nw 70 --out results/market
pixi run python -m shared_evidence.run_market --stage report --out results/market
pixi run python -m shared_evidence.seller_ranking --run-dir results/market --out results/market
```

`seller_ranking` writes Table 3 and the body-text numbers and needs the full run.

## 3. Benchmark the stores

```bash
pixi run python -m shared_evidence.store --idle           # paper protocol, one process on an idle machine
pixi run python -m shared_evidence.store --n 1000 --out results/store/smoke   # short check
```

Paper times were measured on two Intel Xeon Gold 6530 CPUs with SQLite 3.54 in WAL mode. Absolute times depend on hardware.

## 4. Tests

```bash
pixi run python -m shared_evidence.test_store         # maintained sums and threshold queries vs full recomputation
pixi run python -m shared_evidence.test_drift_bound   # drift bound of the stored lower bounds
```

## 5. Rebuild the paper tables and figures

```bash
pixi run python -m paper.tables_figures   # Tables 2–7 and Figures 3–5 into paper/out/
pixi run python -m paper.numbers          # body-text numbers not in the tables
```

Both read the committed `results/`. Table 1 is analytic, and Figures 1 and 2 are diagrams.

## License

MIT. See `LICENSE`.
