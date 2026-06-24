"""Load the raw Last.fm ``.dat`` files into tidy, standardised DataFrames.

The loaders rename the dataset's original columns to the project-wide names defined
in :mod:`recsys.config` (``user_id``, ``artist_id``, ``weight``) so that every model
and metric depends on one stable schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL


def _read_tsv(path: Path | str) -> pd.DataFrame:
    """Read a tab-separated ``.dat`` file, tolerating non-UTF-8 encodings.

    Last.fm artist names contain accented and non-ASCII characters; some files are
    not valid UTF-8, so we fall back to latin-1 rather than crash.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep="\t", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8/latin-1", b"", 0, 1, f"Could not decode {path}")


def load_user_artists(path: Path | str) -> pd.DataFrame:
    """Load user-artist listening counts.

    Parameters
    ----------
    path : Path or str
        Path to ``user_artists.dat``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[user_id, artist_id, weight]``, one row per interaction.
    """
    df = _read_tsv(path)
    df = df.rename(
        columns={"userID": USER_COL, "artistID": ITEM_COL, "weight": WEIGHT_COL}
    )
    expected = {USER_COL, ITEM_COL, WEIGHT_COL}
    if not expected.issubset(df.columns):
        raise ValueError(f"user_artists is missing columns; got {list(df.columns)}")
    if (df[WEIGHT_COL] <= 0).any():
        raise ValueError("Found non-positive listening weights; expected counts > 0.")
    return df[[USER_COL, ITEM_COL, WEIGHT_COL]].reset_index(drop=True)


def load_artists(path: Path | str) -> pd.DataFrame:
    """Load the artist catalogue (id -> name).

    Parameters
    ----------
    path : Path or str
        Path to ``artists.dat``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[artist_id, name]``.
    """
    df = _read_tsv(path)
    df = df.rename(columns={"id": ITEM_COL, "name": "name"})
    if ITEM_COL not in df.columns or "name" not in df.columns:
        raise ValueError(f"artists is missing columns; got {list(df.columns)}")
    return df[[ITEM_COL, "name"]].reset_index(drop=True)
