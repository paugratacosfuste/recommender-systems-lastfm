"""Tests for the static-site build.

The deployed demo is only as correct as this build script: if it emits a bad index or a
stale BUILD tag, the site serves wrong data from a year-long immutable cache. The schema
assertions run against the committed ``data/sample/`` fixture, so they need no download.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys

import pytest

from recsys.config import PROJECT_ROOT, SAMPLE_DIR

_SCRIPT = PROJECT_ROOT / "scripts" / "build_static_site.py"
_SPEC = importlib.util.spec_from_file_location("build_static_site", _SCRIPT)
bss = importlib.util.module_from_spec(_SPEC)
sys.modules["build_static_site"] = bss
_SPEC.loader.exec_module(bss)

HASH = "caea45732bb52679494602c60430435a"
MEDIUM = f"https://cdn-images.dzcdn.net/images/artist/{HASH}/250x250-000000-80-0-0.jpg"


# ------------------------------------------------------------------------ encode_image


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (MEDIUM, HASH),
        # Deezer's "no photo" placeholder is the MD5 of the empty string; showing it
        # would render a blank grey square instead of the coloured-initial tile.
        (MEDIUM.replace(HASH, bss.DEEZER_BLANK), None),
        # Occasionally the hash segment comes back empty entirely.
        ("https://cdn-images.dzcdn.net/images/artist//250x250-000000-80-0-0.jpg", None),
        (None, None),
        ("", None),
    ],
)
def test_encode_image_known_cases(url: str | None, expected: str | None) -> None:
    assert bss.encode_image(url) == expected


@pytest.mark.unit
def test_encode_image_keeps_off_pattern_urls_whole() -> None:
    """Anything we cannot compress is stored verbatim behind a '!' marker."""
    other = "https://example.com/artist.png"
    assert bss.encode_image(other) == "!" + other

    # Right host, unexpected size - keep it rather than silently dropping the photo.
    odd = f"https://cdn-images.dzcdn.net/images/artist/{HASH}/1000x1000-000000-80-0-0.jpg"
    assert bss.encode_image(odd) == "!" + odd


@pytest.mark.unit
def test_marker_can_never_collide_with_a_hash() -> None:
    """The '!' prefix is only unambiguous because a hex digest cannot start with it."""
    assert bss.encode_image(MEDIUM)[0] != "!"


# ----------------------------------------------------------------------------- sharding


@pytest.mark.unit
@pytest.mark.parametrize(("user_id", "shard"), [(2, 0), (99, 0), (100, 1), (2100, 21)])
def test_shard_of(user_id: int, shard: int) -> None:
    assert bss.shard_of(user_id) == shard


# ------------------------------------------------------------------------- build tagging


@pytest.mark.unit
def test_build_tag_matches_the_frontend() -> None:
    """web/app.js appends ?v=BUILD to every data request, which is the only thing making
    the year-long immutable cache header safe. Drift here serves stale JSON for a year."""
    app_js = (PROJECT_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    match = re.search(r'const BUILD = "([^"]+)"', app_js)
    assert match is not None, "web/app.js must declare a BUILD constant"
    assert match.group(1) == bss.BUILD


# ------------------------------------------------------------------------ full schema


@pytest.fixture
def sample_site(tmp_path, monkeypatch):
    """Build the whole bundle from the committed fixture and return (dir, service)."""
    monkeypatch.setattr("app.service.RAW_DIR", SAMPLE_DIR)
    from app.service import RecommenderService

    service = RecommenderService()
    bss.write_site(service, tmp_path, build="test")
    return tmp_path, service


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_site_files_exist(sample_site) -> None:
    out, _ = sample_site
    for name in ("meta.json", "artists.json", "images.json"):
        assert (out / "data" / name).is_file()
    assert list((out / "data" / "users").glob("*.json"))


@pytest.mark.integration
def test_artist_columns_are_index_aligned(sample_site) -> None:
    out, _ = sample_site
    artists = _read(out / "data" / "artists.json")
    lengths = {col: len(artists[col]) for col in ("id", "name", "tag", "hue", "init")}
    assert len(set(lengths.values())) == 1, lengths
    assert len(_read(out / "data" / "images.json")["img"]) == lengths["id"]


@pytest.mark.integration
def test_every_tag_index_resolves(sample_site) -> None:
    out, _ = sample_site
    artists = _read(out / "data" / "artists.json")
    vocab = len(artists["tags"])
    for tags in artists["tag"]:
        assert len(tags) <= 3
        assert all(0 <= t < vocab for t in tags)


@pytest.mark.integration
def test_hues_are_in_range(sample_site) -> None:
    out, _ = sample_site
    assert all(0 <= h < 360 for h in _read(out / "data" / "artists.json")["hue"])


@pytest.mark.integration
def test_meta_matches_the_service(sample_site) -> None:
    out, service = sample_site
    meta = _read(out / "data" / "meta.json")
    # Method order is the contract: shards index recommendation lists by position.
    assert [m["label"] for m in meta["methods"]] == service.methods
    assert meta["users"] == service.user_ids
    assert meta["default_user"] == service.user_ids[0]
    assert meta["build"] == "test"
    assert len(meta["comparison"]) == len(service.methods)
    for method in meta["methods"]:
        assert len(method["accuracy_cards"]) == 4
        assert all(card["hint"] for card in method["accuracy_cards"])


@pytest.mark.integration
def test_shard_payloads_are_well_formed(sample_site) -> None:
    out, service = sample_site
    meta = _read(out / "data" / "meta.json")
    n_artists = len(_read(out / "data" / "artists.json")["id"])
    n_tags = len(_read(out / "data" / "artists.json")["tags"])

    seen = []
    for path in (out / "data" / "users").glob("*.json"):
        shard_no = int(path.stem)
        for raw_uid, entry in _read(path).items():
            uid = int(raw_uid)
            seen.append(uid)
            # The browser derives the filename from the user id alone, so this must hold.
            assert bss.shard_of(uid) == shard_no
            assert len(entry["r"]) == len(service.methods)
            for recs in entry["r"]:
                assert len(recs) <= meta["k"]
                assert all(0 <= i < n_artists for i in recs)
            assert all(0 <= i < n_artists for i, _ in entry["t"])
            assert all(0 <= t < n_tags for t, _ in entry["g"])
    assert sorted(seen) == service.user_ids


@pytest.mark.integration
def test_short_recommendation_lists_survive(sample_site) -> None:
    """Some users get fewer than k recommendations - the payload must not pad or crash."""
    out, _ = sample_site
    lengths = {
        len(recs)
        for path in (out / "data" / "users").glob("*.json")
        for entry in _read(path).values()
        for recs in entry["r"]
    }
    assert lengths, "expected at least one recommendation list"
    assert min(lengths) < _read(out / "data" / "meta.json")["k"]
