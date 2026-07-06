# Tree-Influence Active Learning for LightGBM

Pool-based batch active learning for tabular data using [tree-influence](https://github.com/jjbrophy47/tree_influence) (BoostIn) and LightGBM.

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
| Temp train on all `U_t` pseudo | Scores reflect pseudo-labeled joint fit, not clean add-one counterfactual |
| BoostIn fixed-structure | Split changes when adding points are ignored (Brophy et al. JMLR 2023) |
| BoostIn binary-only | Influence methods require datasets with exactly two classes (labels 0 and 1) |
| Pseudo-labels from **M_t** | Wrong labels bias **M̃_t**; use top-K aggregation ablation |
| **BADGE-adapted** | GBDT leaf-embedding analogue, not original neural BADGE |

Oracle calibration (add-one retrain on ~50 random candidates per round) logs Spearman ρ between proxy scores and true Δmetric(R_ref). Used for **verification only**, not acquisition.

## Install

Requires Python 3.9+. **tree-influence** (BoostIn) requires Python 3.9–3.10; install with:

```bash
pip install -e ".[all]"   # Python 3.9 or 3.10
# or without influence scoring (baselines only):
pip install -e ".[dev]"
```

## Run experiments

```bash
# Single run — influence method (binary classification only)
python -m influence_al.experiments.run --dataset breast_cancer --method influence --seed 42

# Compare methods on a multiclass dataset (baselines only; influence requires binary labels)
python -m influence_al.experiments.run --dataset iris --compare random uncertainty badge

# Multi-seed
python -m influence_al.experiments.run --dataset breast_cancer --method influence --n-seeds 5
```

### Datasets

`iris`, `breast_cancer`, `credit_g`, `phoneme`, `diabetes`, `california_housing`

### Methods

| Method | Description |
|--------|-------------|
| `influence` | Temp-model BoostIn influence U → R_ref |
| `influence_shapley` | Phase 2: Shapley pre-filter + influence |
| `random` | Uniform baseline |
| `uncertainty` | Entropy / residual |
| `margin` | Smallest class margin (classification) |
| `loss` | Highest predicted loss |
| `feature_kmeans` | Diversity-only |
| `badge` | Uncertainty + k-means++ on leaf embeddings |

## Configuration

See [`configs/default.yaml`](configs/default.yaml). Method overrides in [`configs/methods/`](configs/methods/).

Key settings:

- `r_ref_fraction`: fraction of initial L0 held as fixed reference (never trained on)
- `influence.reference`: `r_ref` (default) or `r_pseudo` (ablation)
- `diversity.prefilter_multiplier`: top `k×B` before k-means++
- `oracle.enabled`: Spearman calibration logging

## Tests

```bash
pytest
```

## Project layout

```
src/influence_al/
  data/          Pool, R_ref split, dataset loaders
  models/        LGBM trainer (M_t vs M̃_t)
  acquisition/   Influence scorer, baselines, diversity, Shapley pre-filter
  loop/          AL engine
  evaluation/    Metrics, oracle calibration
  experiments/   CLI
```

## References

- Brophy et al. JMLR 2023 — tree-influence / BoostIn
- Ghorbani et al. — Active Data Shapley (Phase 2 pre-filter)
- Xia & Henao TMLR 2023 — RALIF (reference set ablation)
