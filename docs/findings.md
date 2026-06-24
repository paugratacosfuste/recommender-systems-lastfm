# Findings log

Running notes per module, used to assemble the final slide deck (technical challenges,
method comparison, final remarks). Newest entries at the top.

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
