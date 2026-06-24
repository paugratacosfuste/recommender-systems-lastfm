"""Project paths and shared constants.

Centralising paths here avoids hardcoded strings scattered across modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

# Raw dataset filenames (Last.fm HetRec 2011).
USER_ARTISTS_FILE = "user_artists.dat"
ARTISTS_FILE = "artists.dat"

# Standardised column names used everywhere downstream of the loaders.
USER_COL = "user_id"
ITEM_COL = "artist_id"
WEIGHT_COL = "weight"

DEFAULT_TOP_N = 10
DEFAULT_SEED = 42
