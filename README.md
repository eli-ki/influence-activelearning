# Tree-Influence Active Learning for LightGBM

Pool-based batch active learning on **ADS / RALIF image benchmarks** using [tree-influence](https://github.com/jjbrophy47/tree_influence) (BoostIn) and LightGBM on flattened pixels.

## Core method

Each round `t`:

1. Train **M_t** on `L_t_train` (true labels only; excludes fixed **R_ref**).
2. Pseudo-label all `u ∈ U_t` with **M_t**.
3. Train temp model **M̃_t** on `L_t_train ∪ U_t` (scoring only).
4. Fit BoostIn on **M̃_t**; compute `I = influence(train, R_ref)`.
5. `score(u) = Σ_r I[row(u), r]` (BoostIn: positive = u decreases loss on r).
6. Pre-filter top `k×B`, **k-means++** diversity → query batch **B**.
7. Eval model: retrain on true-labeled `L_t \ R_ref`; report metric on held-out **test T**.

**M̃_t is never used for final evaluation** — only for acquisition scoring.

## Assumptions and limitations

| Assumption | Implication |
|------------|-------------|
| Flattened pixels | Images are vectorized for LightGBM (not CNN features like the original papers) |
| Temp train on all `U_t` pseudo | Scores reflect pseudo-labeled joint fit, not clean add-one counterfactual |
| BoostIn fixed-structure | Split changes when adding points are ignored (Brophy et al. JMLR 2023) |
| BoostIn binary-only | Influence methods require exactly two classes; ADS/RALIF sets are multiclass → use baselines |
| Pseudo-labels from **M_t** | Wrong labels bias **M̃_t**; use top-K aggregation ablation |
| **BADGE-adapted** | GBDT leaf-embedding analogue, not original neural BADGE |

Oracle calibration (add-one retrain on ~50 random candidates per round) logs Spearman ρ between proxy scores and true Δmetric(R_ref). Used for **verification only**, not acquisition.

## Install

Requires Python 3.9+. Vision datasets need **torchvision**; **tree-influence** (BoostIn) requires Python 3.9–3.10:

```bash
pip install -e ".[all]"   # vision + influence + tests (Python 3.9 or 3.10)
pip install -e ".[vision,dev]"   # baselines on ADS/RALIF data without tree-influence
```

Downloads are cached under `data/` (override with `INFLUENCE_AL_DATA_DIR`).

## Run experiments

```bash
# Baselines on CIFAR-10 (multiclass — influence not supported)
python -m influence_al.experiments.run --dataset cifar10 --compare random uncertainty badge

# Single method
python -m influence_al.experiments.run --dataset fashion_mnist --method random --seed 42

# Multi-seed
python -m influence_al.experiments.run --dataset cifar10 --method uncertainty --n-seeds 5
```

### Datasets (ADS + RALIF only)

| Name | Papers | Notes |
|------|--------|-------|
| `cifar10` | ADS, RALIF | 50k train / 10k test |
| `cifar100` | RALIF | 50k train / 10k test |
| `cinic10` | ADS | ~270k train (auto-download) |
| `tiny_imagenet` | ADS, RALIF | 200 classes (auto-download) |
| `svhn` | ADS, RALIF | 73k train / 26k test |
| `svhn_extra` | ADS | ~531k extra split (large pool) |
| `fashion_mnist` | RALIF | 60k train / 10k test |
| `inaturalist` | RALIF | iNaturalist 2018 species (large; subsample via config) |

> **Not included:** ADS **Cheap-10** (custom web-scrape, no public mirror).

Use `max_pool_samples` in [`configs/datasets/`](configs/datasets/) to subsample large pools for tractable LGBM runs.

### Methods

| Method | Description |
|--------|-------------|
| `influence` | Temp-model BoostIn influence U → R_ref (binary only) |
| `influence_shapley` | Phase 2: Shapley pre-filter + influence |
| `random` | Uniform baseline |
| `uncertainty` | Entropy / residual |
| `margin` | Smallest class margin (classification) |
| `loss` | Highest predicted loss |
| `feature_kmeans` | Diversity-only |
| `badge` | Uncertainty + k-means++ on leaf embeddings |

## Configuration

See [`configs/default.yaml`](configs/default.yaml). Per-dataset overrides in [`configs/datasets/`](configs/datasets/).

Key settings:

- `max_pool_samples`: subsample the official training split
- `r_ref_fraction`: fraction of initial L0 held as fixed reference (never trained on)
- `influence.reference`: `r_ref` (default) or `r_pseudo` (ablation)
- `diversity.prefilter_multiplier`: top `k×B` before k-means++
- `oracle.enabled`: Spearman calibration logging

## Tests

```bash
pytest
```

## Robustness & label-efficiency study

```bash
python -m influence_al.experiments.study --dry-run
python -m influence_al.experiments.study --study cifar10_robustness --scenarios clean
python -m influence_al.experiments.study --study cifar10_robustness
```

Outputs under `results/study/cifar10_robustness/`. Scenarios: **clean**, **ood_20**, **ood_35** (ADS-style pool corruption).

## Project layout

```
src/influence_al/
  data/          Vision loaders, pool, R_ref split
  models/        LGBM trainer (M_t vs M̃_t)
  acquisition/   Influence scorer, baselines, diversity, Shapley pre-filter
  loop/          AL engine
  evaluation/    Metrics, oracle calibration, label-efficiency
  experiments/   CLI, robustness study runner
```

## References

- Ghorbani et al. — Active Data Shapley (ADS)
- Xia & Henao TMLR 2023 — RALIF
- Brophy et al. JMLR 2023 — tree-influence / BoostIn
