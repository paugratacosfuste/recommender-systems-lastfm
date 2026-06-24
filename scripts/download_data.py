"""Download and unpack the Last.fm HetRec 2011 dataset into data/raw/.

Source: https://grouplens.org/datasets/hetrec-2011/
Uses only the standard library so it runs before the conda env is built.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Files we expect inside the archive (used to verify a complete download).
EXPECTED_FILES = (
    "artists.dat",
    "tags.dat",
    "user_artists.dat",
    "user_friends.dat",
    "user_taggedartists.dat",
)


def download_and_extract(url: str = DATASET_URL, dest: Path = RAW_DIR) -> None:
    """Download the dataset zip and extract its files into ``dest``.

    Parameters
    ----------
    url : str
        URL of the HetRec 2011 Last.fm zip archive.
    dest : Path
        Directory the ``.dat`` files are extracted into (created if missing).
    """
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (trusted host)
        payload = response.read()
    print(f"Downloaded {len(payload) / 1e6:.1f} MB. Extracting into {dest} ...")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest)

    missing = [name for name in EXPECTED_FILES if not (dest / name).exists()]
    if missing:
        raise RuntimeError(f"Download incomplete; missing files: {missing}")
    print("Done. Files:")
    for name in sorted(p.name for p in dest.glob("*.dat")):
        print(f"  - {name}")


if __name__ == "__main__":
    try:
        download_and_extract()
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user clearly
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
