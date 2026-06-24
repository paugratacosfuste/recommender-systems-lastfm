# Music Recommender - Individual Project

A progressive music recommender prototype built on the Last.fm HetRec 2011 implicit-feedback
dataset, for the ESADE Recommender Systems course (Prof. Marc Torrens).

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
```

## Dataset

Last.fm HetRec 2011 (GroupLens). User-artist listening counts, artist tags, and a user
friendship graph. Implicit feedback only (no star ratings). Source:
https://grouplens.org/datasets/hetrec-2011/

The raw and processed data are not committed (see `.gitignore`); they are reproduced by
`scripts/download_data.py`. A tiny fixture in `data/sample/` is committed so tests run
without the full download.
