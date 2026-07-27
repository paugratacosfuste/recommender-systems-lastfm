# Music Recommender - Individual Project

**🔗 Live demo:** https://music-recommender-lastfm.vercel.app
*(fully static - loads instantly, no server to wake up)*

A progressive music recommender prototype built on the Last.fm HetRec 2011 implicit-feedback
dataset, for the ESADE Recommender Systems course (Prof. Marc Torrens). It compares eight
methods on accuracy and beyond-accuracy metrics, with a Flask app, a PDF report, and a slide
deck.

See `CLAUDE.md` for how the project is organised and `PLAN.md` for the module-by-module plan.

## Quickstart

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate recsys-music

# 2. Download the dataset (Last.fm HetRec 2011 -> data/raw/)
python scripts/download_data.py

# 3. Run the tests
pytest

# 4. Run the prototype app
flask --app app/app.py run --debug
# open http://127.0.0.1:5000

# 5. (optional) Rebuild the static demo that is deployed publicly
python scripts/build_static_site.py
python -m http.server 8000 --directory web   # open http://localhost:8000
```

## How the live demo is deployed

The Flask app in `app/` builds its recommender at import time: it loads the dataset, fits
all eight models and evaluates each one over the held-out users. That is ~13s of work
before the first byte is served, which is fine locally but fatal on a free host that spins
containers down when idle - the cost lands on the next visitor and the request times out.

Since the models are deterministic and the inputs are finite, none of that has to happen at
request time. `scripts/build_static_site.py` runs the pipeline once offline and writes every
result - 1,892 listeners x 8 methods x top-10, plus each user's taste profile and every
metric - into `web/` as static JSON. The deployed site is those files plus ~11 KB of vanilla
JavaScript: no server, no database, nothing to cold-start. Switching method costs no network
request at all, and switching listener costs one ~13 KB shard.

This is the same batch-precompute-then-serve split that production recommenders use for
daily-mix style features, and it is why the demo now loads in under half a second.

## Dataset

Last.fm HetRec 2011 (GroupLens). User-artist listening counts, artist tags, and a user
friendship graph. Implicit feedback only (no star ratings). Source:
https://grouplens.org/datasets/hetrec-2011/

The raw and processed data are not committed (see `.gitignore`); they are reproduced by
`scripts/download_data.py`. A tiny fixture in `data/sample/` is committed so tests run
without the full download.
