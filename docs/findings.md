# Findings log

Running notes per module, used to assemble the final slide deck (technical challenges,
method comparison, final remarks). Newest entries at the top.

---

## Module 2 - Dataset, EDA & preprocessing (2026-06-24)

**Goal:** understand the data and turn it into model-ready structures (sparse matrix +
confidence weights), with charts for the deck.

**What was built**
- `src/recsys/data/preprocess.py`: `IndexMapping` (id <-> contiguous matrix index),
  `build_interaction_matrix` (sparse CSR), `log_scale` (log1p), and `confidence`
  (Hu/Koren/Volinsky `c = 1 + alpha*log(1+r/eps)`).
- `src/recsys/data/eda.py`: `gini`, `summary_stats`, `popularity_curve`, `top_k_play_share`.
- `scripts/build_processed.py`: caches `data/processed/` (interactions.parquet with
  log_weight + confidence columns, user_item_matrix.npz, mappings.npz).
- `notebooks/01_eda.ipynb`: executed EDA with figures saved to `docs/figures/`.
- Tests: 34 total, 93% coverage.

**Key numbers**
- 1,892 users x 17,632 artists, 92,834 interactions. **Sparsity 99.72%.**
- Median **50 artists per user** (the dataset keeps ~top-50 per user); median
  **1 listener per artist** - a massive cold/long tail.
- Popularity concentration: **plays Gini = 0.893.** Top 50 artists = 34.8% of all plays,
  top 100 = 43.7%, top 500 = 67.9%.

**Figures (for the deck):** `play_count_distribution.png`, `activity_distributions.png`,
`long_tail_lorenz.png`.

**Why it matters**
- Extreme sparsity + heavy tail means collaborative methods will struggle on niche artists;
  log-scaling/confidence weighting is justified before CF and matrix factorisation.
- The high Gini is the evidence behind "accuracy is not enough": a popularity baseline can
  score acceptably while ignoring 99% of the catalogue (popularity bias).

**Definition of done:** met. Processed artifacts reproducible, EDA charts generated, all
helpers tested at >= 80% coverage.

**Next:** Module 3 - extend the evaluation harness (Recall@K alongside Precision@K) and
formalise the non-personalised baseline comparison.

---

## Phase 0 - Thin end-to-end slice (2026-06-24)

**Goal:** prove the whole pipeline works before deepening anything: load data -> popularity
baseline -> one Flask screen -> Precision@K on a held-out split.

**What was built**
- `scripts/download_data.py`: standard-library downloader for Last.fm HetRec 2011 into
  `data/raw/` (2.6 MB zip from GroupLens).
- `src/recsys` package: data loaders (encoding-robust), per-user `leave_n_out_split`,
  `BaseRecommender` interface, `PopularityRecommender`, `precision_at_k`, and the `evaluate`
  harness.
- `app/`: Flask prototype - pick a user, see top-10 recommendations + the offline metric.
- `tests/`: 21 tests, 90% coverage. A committed tiny fixture in `data/sample/` lets tests
  run without the full download.

**Numbers (popularity "plays" baseline)**
- Dataset: 1,892 users, 17,632 artists, 92,834 interactions.
- Split: per-user random hold-out, `test_frac=0.2`, seed 42.
- **Precision@10 = 0.0602** over 1,884 evaluable users. This is the floor every later
  method must beat.

**Technical challenges / decisions**
- Implicit feedback only: listening counts are treated as confidence, evaluation is
  ranking-based (Precision@K), not RMSE.
- Per-user split (not global) so every test user is also in training - required for
  personalised methods later.
- Popularity is defined three ways (`plays`, `listeners`, `damped`); "plays" is the default.
  This sets up the popularity-bias discussion later: top recs are all mainstream pop
  (Britney Spears, Rihanna, Katy Perry...), which is exactly the bias to critique.
- `homebrew conda` is actually a mamba shim; the env lives at
  `/Users/paugratacosfuste/mamba/envs/recsys-music`. Use that python directly.

**Definition of done:** met. App runs, shows recommendations for a chosen user, reports one
offline metric; tests green at >= 80% coverage.

**Next:** Module 2 - deepen EDA + preprocessing (play-count distribution / long tail,
sparsity, Lorenz/Gini, confidence weighting, cached parquet artifacts).
