# CLAUDE.md - Recommender Systems Individual Project

Music recommender prototype for ESADE MSc (Prof. Marc Torrens). Built module-by-module
on Last.fm HetRec 2011 implicit feedback. This file defines HOW we work; see PLAN.md for
WHAT we build.

## Project scope

- Project root is this `Individual Assignment/` folder. All code, data, docs, and the app
  live here, not at the degree-wide git repo root.
- Grading drives effort: 50% technical implementations, 30% evaluation, 20% UX. Evaluation
  is the high-leverage, usually under-built area - invest there.

## Environment

- Python via **conda**. Environment is defined in `environment.yml`.
- Create / update / activate:
  - `conda env create -f environment.yml` (first time)
  - `conda env update -f environment.yml --prune` (after dependency changes)
  - `conda activate recsys-music`
- Add a dependency by editing `environment.yml`, then run the update command above. Do not
  `pip install` ad hoc without recording it.

## Running things

- App (Flask): `flask --app app/app.py run --debug` (auto-reloads on edit). Served at
  http://127.0.0.1:5000.
- Notebooks: `jupyter lab` from the project root. Notebooks import from the `src/recsys`
  package; they never define reusable logic inline - that lives in `src`.
- Data download: `python scripts/download_data.py` (fetches and unpacks Last.fm HetRec 2011
  into `data/raw/`). Preprocessing scripts write cached parquet into `data/processed/`.

## Repository structure

```
Individual Assignment/
  CLAUDE.md, PLAN.md, README.md, environment.yml, .gitignore
  data/
    raw/        # downloaded dataset (gitignored)
    processed/  # cached parquet artifacts (gitignored)
    sample/     # tiny fixture committed for tests
  scripts/      # download_data.py, one-off preprocessing entrypoints
  src/recsys/
    data/       # loading, preprocessing, confidence weighting, train/test split
    models/     # base, popularity, cf, content, mf  (one model = one module)
    eval/       # metrics (accuracy + beyond-accuracy) and the evaluation harness
    utils/
  app/          # Flask app: app.py, templates/, static/
  notebooks/    # EDA and experiments only
  tests/        # mirrors src/recsys
  docs/         # findings.md (running deck notes)
```

Keep files small and cohesive (200-400 lines typical, 800 max). One algorithm family per
module file. Extract shared helpers into `utils/`.

## Data handling

- Raw and processed data are gitignored. Reproducibility comes from `scripts/download_data.py`
  plus documented preprocessing, not from committing data.
- A tiny `data/sample/` fixture IS committed so tests run without the full download.
- Always cite the dataset source and document any filtering/sampling in `docs/findings.md`
  and the relevant notebook.

## Git workflow

- Work on branch `pau_branch`. Commit directly to it (no per-module branches).
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`.
- Commit when a module's definition of done is met, or at meaningful checkpoints.
- Never push unless Pau explicitly asks. No co-author/attribution lines (disabled globally).

## Testing and definition of done (TDD)

- Test-driven: write the failing test first, then the minimal implementation, then refactor.
- Target **80% coverage** (`pytest --cov`). Tests live in `tests/` mirroring `src/`.
- Prioritise testing deterministic, pure functions: metrics, log-scaling/confidence weighting,
  train/test split, similarity computations. Recommender models are tested through their
  public interface (`fit` / `recommend`) on the committed `data/sample/` fixture.
- A module is DONE when:
  1. Its code is importable from `src/recsys` and used by the Flask app.
  2. It reports its evaluation metrics on the held-out split (metrics are sane / improve on
     the previous baseline).
  3. Tests pass and coverage stays >= 80%.
  4. Findings appended to `docs/findings.md`.

## Code style

- PEP 8, formatted with **black**, linted with **ruff**.
- Type hints on all public function signatures. NumPy-style docstrings on public functions
  and classes.
- Immutable patterns by default: return new objects, avoid in-place mutation of shared data
  (numpy/scipy array ops are the pragmatic exception for performance - keep them local).
- Validate inputs at boundaries (data loaders, app request handlers). Handle errors
  explicitly; never silently swallow.
- No hardcoded paths or magic numbers - use a small config module / constants.

## Evaluation cadence

- The evaluation harness (`src/recsys/eval`) is built incrementally from Module 3 onward and
  reused everywhere. Every new method is scored on the SAME train/test split so comparisons
  are fair.
- Accuracy metrics: Precision@K, Recall@K, MAP, NDCG. Beyond-accuracy: catalogue coverage,
  intra-list diversity, novelty, popularity bias, latency/scalability.

## Slide deck

- After each module, append to `docs/findings.md`: technical challenges hit, metric values,
  comparison notes, and screenshots. The final deck (technical challenges, method comparison,
  final remarks) is assembled from these notes.

## Communication preferences

- Never use the em dash character. Use a hyphen `-` instead.
- Be direct and concise. Pau has a builder mindset; lead with the action/result.
- For manual UI tasks, give one step at a time and wait, rather than dumping a full plan.
- Learning mode is active: surface brief educational insights and invite Pau to write the
  small, decision-heavy pieces of code (algorithm choices, business logic) rather than doing
  everything automatically.
