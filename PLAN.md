# PLAN.md - Music Recommender, Phased Build

Phased, module-by-module plan mapped to the seven assignment modules and the
50% technical / 30% evaluation / 20% UX grading split. Agile: a thin end-to-end slice first,
then deepen each module. Dataset: Last.fm HetRec 2011 (implicit feedback).

## Dataset reference (Last.fm HetRec 2011)

- `user_artists.dat`: user, artist, weight (listening count) -> the implicit signal.
- `artists.dat`: artist id, name, url.
- `tags.dat` + `user_taggedartists.dat`: tag vocabulary and user-applied artist tags ->
  content features.
- `user_friends.dat`: social graph (optional, for a social-baseline stretch goal).
- ~1.9k users, ~17.6k artists, ~92k user-artist interactions. Sparse, heavy popularity tail.

## Implicit-feedback design principles (apply throughout)

- No ratings: treat listening counts as confidence, not preference scores.
- Log-scale plays; build confidence weights (Hu, Koren, Volinsky 2008):
  `c_ui = 1 + alpha * log(1 + r_ui / eps)`.
- Evaluation is ranking-based on held-out interactions (per-user leave-some-out), not RMSE.

---

## Phase 0 - Thin end-to-end slice (vertical, do first)

Goal: prove the whole pipeline works before deepening anything.
- Load data -> popularity baseline -> one Flask screen -> Precision@K on a held-out split.
- Touches Modules 1, 2 (minimal), 3 (minimal), and seeds the eval harness.
- DoD: pick a user in the Flask app, see top-N popular artists, and read one offline metric.

---

## Module 1 - Prototype UI (UX track, 20%)

- Scope: Flask app shell. User selector (dropdown / id input), top-N recommendation list
  showing artist name + tags, and a method switcher (wired to popularity first).
- Grows over the course into a side-by-side method comparison view and simple
  "why recommended" explanations.
- Algorithms: none yet (renders whatever the active model returns).
- Metrics added: none (UX, qualitative).
- DoD: app runs via `flask --app app/app.py run`, renders recommendations for a chosen user,
  swaps methods without code edits.

## Module 2 - Setup + dataset + EDA + preprocessing (technical 50%)

- Scope: loaders for all `.dat` files; EDA notebook (play-count distribution / long tail,
  interactions per user/artist, sparsity, popularity concentration via Lorenz/Gini); build the
  sparse user-item matrix; confidence weighting + log-scaling; per-user train/test split
  (leave-some-out). Cache processed artifacts as parquet.
- Algorithms: none (data engineering).
- Metrics added: dataset statistics only.
- Tests (TDD): split correctness (no leakage, every test user present in train), confidence
  weighting math, matrix shape/sparsity.
- DoD: `data/processed/` parquet built reproducibly from raw; EDA notebook complete; split +
  weighting functions tested; sample fixture committed.

## Module 3 - Non-personalised baseline (technical 50% + starts eval 30%)

- Scope: most-popular recommender (by total plays and by distinct-user count), plus a damped
  popularity variant. This is the comparison floor for every later method.
- Build the first real evaluation harness here.
- Metrics added: **Precision@K, Recall@K**.
- Tests (TDD): Precision@K / Recall@K against hand-computed toy cases; popularity ranking.
- DoD: baseline wired into the app and scored on the held-out split; numbers logged in
  `docs/findings.md`.

## Module 4 - Collaborative filtering (technical 50%)

- Scope: memory-based CF on the implicit confidence matrix. Primary: **item-item kNN with
  cosine similarity** (scalable, strong on implicit data). Secondary for comparison: user-user
  kNN. Hand-implement the similarity + scoring; use scipy sparse ops for speed.
- Metrics added: **MAP, NDCG** (extend the harness).
- Tests (TDD): cosine similarity, neighbour selection, recommendation aggregation on the
  fixture.
- DoD: CF in the app and method switcher; full accuracy metrics vs the baseline in
  `docs/findings.md`.

## Module 5 - Content-based filtering (technical 50% + beyond-accuracy eval)

- Scope: build artist tag profiles with **TF-IDF** over the tag vocabulary. User profile =
  play-weighted aggregate of the tags of artists they listen to. Recommend by cosine
  similarity between the user profile and artist tag vectors. Naturally handles cold-start
  artists better than CF.
- Metrics added (beyond-accuracy): **catalogue coverage, intra-list diversity** (1 - mean
  pairwise tag cosine), **novelty** (self-information / inverse popularity of recommended
  items).
- Tests (TDD): TF-IDF vectorisation, profile aggregation, diversity/novelty metrics.
- DoD: content model in the app; accuracy + diversity/novelty reported and contrasted with
  CF (expect CF to win accuracy, content to win diversity/coverage).

## Module 6 - Matrix factorisation (technical 50% + scalability eval)

- Scope: **implicit ALS** (Hu/Koren/Volinsky 2008) hand-implemented (alternating least
  squares over the confidence matrix), with the `implicit` library as a benchmark to validate
  correctness and speed. Optional stretch: BPR.
- Metrics added: **latency / scalability** (fit time, recommend latency, memory) plus all
  accuracy + beyond-accuracy metrics.
- Tests (TDD): one ALS update step on a tiny matrix vs a closed-form check; reconstruction
  improves over iterations; recommend output shape.
- DoD: MF in the app; hyperparameters (factors, regularisation, alpha) tuned on a validation
  split; full metric row in the comparison table; scalability notes in `docs/findings.md`.

## Module 7 - Evaluation consolidation (the 30%)

- Scope: unified evaluation across ALL methods on the SAME split, producing one comparison
  matrix and plots. Critical discussion: accuracy vs diversity/novelty trade-offs, popularity
  bias per method, cold-start behaviour, scalability.
- Metrics consolidated:
  - Accuracy: Precision@K, Recall@K, MAP, NDCG.
  - Beyond-accuracy: coverage, intra-list diversity, novelty, popularity bias
    (e.g. average recommended-item popularity + Gini of recommendation frequency).
  - Operational: latency / throughput / memory.
- Tests (TDD): the harness aggregates per-user metrics correctly; deterministic given a seed.
- DoD: one notebook/script regenerates the full comparison matrix and figures used in the
  deck; written critical analysis in `docs/findings.md`.

---

## Cross-cutting tracks

- **Evaluation track (30%)**: `src/recsys/eval` is built incrementally (Module 3 -> 7),
  fully tested, and reused by both notebooks and the app. Same split everywhere for fair
  comparison.
- **UX track (20%)**: the Flask app accretes features each module - user picker (M1),
  method switcher (M3), comparison view (M4+), tags + explanations (M5), and a final polish
  pass for the demo.
- **Deck track**: `docs/findings.md` is appended after every module so the final slide deck
  (technical challenges, method comparison, final remarks) assembles from real notes.

## Suggested sequencing

Phase 0 slice -> M2 deepen -> M3 -> M4 -> M5 -> M6 -> M7 consolidation, with the UX and
findings tracks advanced a little inside each module rather than left to the end.
