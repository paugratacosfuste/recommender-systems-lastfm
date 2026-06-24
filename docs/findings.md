# Findings log

Running notes per module, used to assemble the final slide deck (technical challenges,
method comparison, final remarks). Newest entries at the top.

---

## Module 5 - Content-based filtering & beyond-accuracy metrics (2026-06-25)

**Goal:** add a content-based method (artist tags) and the beyond-accuracy metrics that
make the "accuracy is not enough" argument measurable.

**What was built**
- Tag loaders (`load_tags`, `load_user_tagged_artists`).
- `src/recsys/models/content.py`: `build_artist_tag_profiles` (TF-IDF over the artist x tag
  matrix) and `ContentBasedRecommender` (play-weighted user profile in tag space, cosine
  recommend). Can recommend cold-start artists CF cannot.
- `src/recsys/eval/beyond_accuracy.py`: `catalogue_coverage`, `intra_list_diversity`
  (1 - mean pairwise tag cosine), `novelty` (mean self-information), `build_item_popularity`,
  and `BeyondAccuracyInputs`. `evaluate()`/`compare_models()` optionally report them.
- App: Content-based added to the switcher; page now shows coverage / diversity / novelty.
- `scripts/evaluate_baselines.py` extended with content + beyond-accuracy columns.
- Tests: 65 total, 93% coverage.

**Full method comparison (k=10, held-out split, seed 42, 1,884 users)**

| Method                | P@10   | NDCG@10 | Coverage | Diversity | Novelty |
|-----------------------|-------:|--------:|---------:|----------:|--------:|
| Item-item CF          | 0.1749 | 0.2174  | 0.156    | 0.616     | 4.75    |
| User-user CF          | 0.1292 | 0.1598  | 0.018    | 0.619     | 2.98    |
| Content-based         | 0.0994 | 0.1144  | 0.143    | 0.338     | 6.57    |
| Popularity (listeners)| 0.0691 | 0.0798  | 0.0015   | 0.627     | 2.44    |
| Popularity (plays)    | 0.0602 | 0.0653  | 0.0014   | 0.676     | 2.68    |
| Popularity (damped)   | 0.0430 | 0.0441  | 0.0012   | 0.765     | 3.44    |

**The "accuracy is not enough" story (deck centrepiece)**
- **Popularity bias is now quantified:** popularity methods reach ~0.001 coverage - the
  same ~25 artists served to all 1,884 users. CF and content cover ~15% of the catalogue.
- **No single winner across objectives:** item-item CF wins accuracy *and* coverage;
  content-based wins novelty (6.57, the most non-obvious picks) but has the lowest diversity
  (0.34) because tag-similar recommendations are internally homogeneous (filter bubble).
- **Damped popularity** trades accuracy for the highest intra-list diversity (0.77) -
  the deliberate accuracy-vs-bias trade-off flagged back in Module 3, now confirmed.
- Diversity caveat: it is measured in tag space, so methods that ignore tags (popularity)
  can still score high; report it alongside coverage/novelty, not alone.

**Definition of done:** met. Content model in app + switcher; accuracy and beyond-accuracy
reported and contrasted across all six methods; tests green at >= 80%.

**Next:** Module 6 - matrix factorisation (implicit ALS, hand-implemented), adding
latency / scalability to the comparison.

---

## Module 4 - Collaborative filtering (2026-06-24)

**Goal:** first personalised methods (memory-based kNN) + richer ranking metrics.

**What was built**
- `src/recsys/models/cf.py`: hand-implemented sparse cosine similarity
  (`cosine_similarity_rows`, with optional top-k neighbour pruning), `ItemKNNRecommender`
  (item-item) and `UserKNNRecommender` (user-user), both on the log-scaled implicit matrix.
- MAP (`average_precision_at_k`) and NDCG (`ndcg_at_k`) added to `eval/metrics.py`;
  `evaluate()` now reports Precision, Recall, MAP, NDCG, and `fit_seconds`.
- App switcher now lists Item-item CF and User-user CF (personalised methods first); the
  page shows all four accuracy metrics.
- `scripts/evaluate_baselines.py` extended -> `docs/method_comparison.csv`.
- Tests: 53 total, 95% coverage.

**Method comparison (k=10, held-out split, seed 42, 1,884 users)**

| Method                | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | fit (s) |
|-----------------------|-------------:|----------:|-------:|--------:|--------:|
| **Item-item CF**      | **0.1749**   | 0.1775    | 0.1135 | 0.2174  | 0.02    |
| User-user CF          | 0.1292       | 0.1329    | 0.0805 | 0.1598  | 0.03    |
| Popularity (listeners)| 0.0691       | 0.0703    | 0.0351 | 0.0798  | 0.01    |
| Popularity (plays)    | 0.0602       | 0.0615    | 0.0267 | 0.0653  | 0.01    |
| Popularity (damped)   | 0.0430       | 0.0437    | 0.0153 | 0.0441  | 0.01    |

**Why it matters**
- **Personalisation more than doubles accuracy:** Item-item CF hits Precision@10 0.175 vs
  the best baseline 0.069 (~2.5x), and NDCG 0.217 vs 0.080. Clear evidence that "who
  co-listens with what" beats "what is globally popular".
- Item-item beats user-user here - artist-artist co-occurrence is a stronger, more stable
  signal than user-user overlap given extreme sparsity.
- CF can surface niche artists the popularity baseline never would, which should show up as
  better coverage/novelty once those beyond-accuracy metrics arrive (Module 5).
- Both fit in well under a second; scalability is fine at this dataset size (revisit with
  top-k pruning / approximate neighbours if scaling up).

**Definition of done:** met. CF wired into app + switcher; full accuracy metrics vs the
baseline recorded; tests green at >= 80%.

**Next:** Module 5 - content-based filtering (artist tags, TF-IDF) and the first
beyond-accuracy metrics (coverage, intra-list diversity, novelty).

---

## Module 3 - Non-personalised baseline & evaluation harness (2026-06-24)

**Goal:** formalise the popularity baseline and grow the evaluation harness into a fair,
reusable comparison tool (the 30% grade backbone).

**What was built**
- `recall_at_k` added to `src/recsys/eval/metrics.py`.
- `evaluate()` now reports Precision@K, Recall@K, `n_users`, and `fit_seconds`;
  `compare_models()` scores several named models on one split and returns a sorted table.
- App method switcher enabled: three popularity strategies (plays / listeners / damped)
  are selectable, each showing its own offline Precision@K and Recall@K.
- `scripts/evaluate_baselines.py`: writes the comparison table (later renamed
  `docs/method_comparison.csv` in Module 4 as more methods were added).
- Tests: 40 total, 94% coverage.

**Baseline comparison (k=10, held-out split, seed 42, 1,884 users)**

| Strategy   | Precision@10 | Recall@10 |
|------------|-------------:|----------:|
| listeners  | 0.0691       | 0.0703    |
| plays      | 0.0602       | 0.0615    |
| damped     | 0.0430       | 0.0437    |

**Why it matters**
- "Listeners" (broad appeal) beats "plays" - total plays let a few superfans distort the
  ranking, so counting *distinct* listeners is a better popularity signal here.
- "Damped" deliberately sacrifices accuracy by suppressing mega-hits; we expect it to pay
  off later on beyond-accuracy metrics (novelty / coverage), illustrating the core
  accuracy-vs-bias trade-off.
- **0.069 is the number to beat:** every personalised method from Module 4 onward must
  exceed the best non-personalised baseline to justify its complexity.

**Definition of done:** met. Baseline scored and wired into the app; harness reports two
accuracy metrics; comparison reproducible via script; tests green at >= 80%.

**Next:** Module 4 - collaborative filtering (item-item kNN on the implicit matrix),
adding MAP and NDCG to the harness.

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
