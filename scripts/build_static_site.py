"""Precompute the whole demo into static JSON so the deployed site needs no server.

The Flask app builds ``RecommenderService`` at import time: it loads the raw dataset,
fits all eight methods, and evaluates each one over the held-out users before serving a
single byte (~13 s locally, minutes on a small shared CPU). On a free container that
spins down when idle, that boot cost lands on the *visitor*, and the request times out.

Every one of those outputs is finite and deterministic (the ALS seed is fixed), so this
script computes them once - 1,892 users x 8 methods x top-10 - and writes a static
bundle that any CDN can serve with no runtime at all.

Usage
-----
    python scripts/build_static_site.py                # full build, fetches artwork
    python scripts/build_static_site.py --skip-images  # offline, uses the cache only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make both the project root (for the `app` package) and src/ (for `recsys`) importable,
# mirroring app/app.py so this script runs from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from app.service import Recommendation, RecommenderService  # noqa: E402

# Bump on every rebuild. web/app.js appends this as ?v=<BUILD> to every data request,
# which is what makes the year-long `immutable` cache header in vercel.json safe.
# tests/test_build_static_site.py asserts the two stay in sync.
BUILD = "20260727-1"

SHARD_SIZE = 100
TOP_ARTISTS = 6
GENRE_MIX = 6
DEFAULT_OUT = _PROJECT_ROOT / "web"
IMAGE_CACHE = _PROJECT_ROOT / "data" / "deezer_images.json"

# Deezer artwork. `picture_medium` URLs look like:
#   https://cdn-images.dzcdn.net/images/artist/<32-hex>/250x250-000000-80-0-0.jpg
# Only the hash varies, so we store that and rebuild the URL in the browser.
DEEZER_API = "https://api.deezer.com/search/artist?q={q}&limit=1"
DZ_ARTIST_RE = re.compile(r"^https://cdn-images\.dzcdn\.net/images/artist/([0-9a-f]*)/")
DZ_MEDIUM_RE = re.compile(
    r"^https://cdn-images\.dzcdn\.net/images/artist/([0-9a-f]{32})/"
    r"250x250-000000-80-0-0\.jpg$"
)
# MD5 of the empty string: Deezer's "artist has no photo" placeholder, a blank grey
# square. Treated as "no image" so the coloured-initial tile shows instead.
DEEZER_BLANK = "d41d8cd98f00b204e9800998ecf8427e"
# Distinguishes a transient failure (retry on the next run) from a genuine no-hit
# (cached as None forever). Without this, one flaky minute poisons the cache.
FAILED = "__failed__"
IMAGE_WORKERS = 5
IMAGE_TIMEOUT = 8
IMAGE_ATTEMPTS = 4
SAVE_EVERY = 200
# Deezer allows ~50 requests per 5 seconds and signals breaches with an `error` object
# inside a 200 response, not an HTTP status. Pace request starts across all workers to
# stay under the ceiling; going faster just converts artwork into false "no photo".
MIN_REQUEST_INTERVAL = 0.11

# Metric cards, mirroring app/app.py:43-83 - label, metric key, format, plain-language
# hint. Formatting happens here in Python so the static page cannot drift from Flask.
ACCURACY_FIELDS = (
    (
        "Precision@{k}",
        "precision_at_k",
        ".3f",
        "of the picks shown, the share the user actually liked",
    ),
    (
        "Recall@{k}",
        "recall_at_k",
        ".3f",
        "of everything the user liked, the share we surfaced",
    ),
    ("MAP@{k}", "map_at_k", ".3f", "rewards putting the right items near the top"),
    ("NDCG@{k}", "ndcg_at_k", ".3f", "overall ranking quality (top hits count more)"),
)
BEYOND_FIELDS = (
    (
        "Coverage",
        "coverage",
        ".3f",
        "share of the whole catalogue this method ever recommends",
    ),
    ("Diversity", "diversity", ".3f", "how varied the artists within one list are"),
    (
        "Novelty",
        "novelty",
        ".2f",
        "how non-mainstream the picks are (higher = deeper cuts)",
    ),
)


@dataclass(frozen=True)
class Collected:
    """Everything the site needs, straight off the fitted service."""

    recs: dict[str, dict[int, list[Recommendation]]]  # method -> user -> top-N
    top: dict[int, list[Recommendation]]  # user -> most-played
    genres: dict[int, list[dict]]  # user -> [{tag, count}]


@dataclass(frozen=True)
class ArtistIndex:
    """Position-based index over the artists that actually appear in the output."""

    order: list[int]  # artist_id, in output order
    pos: dict[int, int]  # artist_id -> position (what the shards store)
    meta: dict[int, Recommendation]  # artist_id -> name/tags/hue/initial
    tag_pos: dict[str, int]  # tag label -> position in the vocabulary


# --------------------------------------------------------------------------- artwork


def encode_image(url: str | None) -> str | None:
    """Compress a Deezer artwork URL to the part that varies.

    Returns the 32-character hash for a standard ``picture_medium`` URL, ``None`` when
    there is no usable artwork, and ``"!" + url`` for an off-pattern URL (a leading
    ``!`` can never start a hex hash, so the marker is unambiguous).
    """
    if not url:
        return None
    dz = DZ_ARTIST_RE.match(url)
    if dz is None:
        return "!" + url  # some other host entirely - keep it whole
    digest = dz.group(1)
    if len(digest) != 32 or digest == DEEZER_BLANK:
        return None  # empty hash or the blank-placeholder square
    return digest if DZ_MEDIUM_RE.match(url) else "!" + url


def load_cache(path: Path) -> dict[str, str | None]:
    """Read the artist-name -> encoded-artwork cache, tolerating a corrupt file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def save_cache(path: Path, cache: dict[str, str | None]) -> None:
    """Write the cache atomically so an interrupted run never truncates it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8"
    )
    os.replace(tmp, path)


_RATE_LOCK = threading.Lock()
_next_slot = [0.0]


def _throttle() -> None:
    """Space request starts evenly across worker threads."""
    with _RATE_LOCK:
        now = time.monotonic()
        start = max(now, _next_slot[0])
        _next_slot[0] = start + MIN_REQUEST_INTERVAL
    delay = start - now
    if delay > 0:
        time.sleep(delay)


def _probe(name: str) -> tuple[str, str | None, bool]:
    """Look one artist up on Deezer. Returns (name, url, was_transient_failure).

    Deezer reports quota breaches as ``{"error": {...}}`` inside an HTTP 200, so a naive
    ``payload["data"] or []`` reads a throttled request as "this artist has no photo"
    and caches that permanently. Anything that is not a clean response is retried with
    backoff, then handed back as transient so the next run picks it up again.
    """
    api = DEEZER_API.format(q=urllib.parse.quote(name))
    delay = 1.0
    for _ in range(IMAGE_ATTEMPTS):
        _throttle()
        try:
            with urllib.request.urlopen(api, timeout=IMAGE_TIMEOUT) as resp:  # noqa: S310
                payload = json.loads(resp.read())
        except Exception:  # noqa: BLE001 - network/JSON trouble is retryable, not fatal
            payload = None
        if isinstance(payload, dict) and "error" not in payload:
            hits = payload.get("data") or []
            return name, (hits[0].get("picture_medium") if hits else None), False
        time.sleep(delay + random.random() * 0.5)
        delay *= 2
    return name, None, True


def fetch_images(
    names: list[str],
    cache: dict[str, str | None],
    *,
    cache_path: Path | None = None,
    workers: int = IMAGE_WORKERS,
) -> dict[str, str | None]:
    """Resolve artwork for ``names``, reusing and updating ``cache`` in place.

    Only names that are missing or previously failed are re-fetched, so a second run
    costs nothing. Progress is checkpointed, including on Ctrl-C.
    """
    todo = [n for n in dict.fromkeys(names) if cache.get(n, FAILED) == FAILED]
    if not todo:
        print(f"  artwork: all {len(set(names))} artists already cached")
        return cache
    print(f"  artwork: resolving {len(todo)} artists via Deezer...", flush=True)
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, url, transient in pool.map(_probe, todo):
                cache[name] = FAILED if transient else encode_image(url)
                done += 1
                if cache_path and done % SAVE_EVERY == 0:
                    save_cache(cache_path, cache)
                    print(f"    {done}/{len(todo)}", flush=True)
    finally:
        if cache_path:
            save_cache(cache_path, cache)
    return cache


# ----------------------------------------------------------------------- collect/index


def collect(service: RecommenderService) -> Collected:
    """Run every method for every user, plus each user's taste profile."""
    recs = {
        label: {uid: service.recommend(label, uid) for uid in service.user_ids}
        for label in service.methods
    }
    return Collected(
        recs=recs,
        top={
            uid: service.user_top_artists(uid, n=TOP_ARTISTS) for uid in service.user_ids
        },
        genres={
            uid: service.user_genre_mix(uid, n=GENRE_MIX) for uid in service.user_ids
        },
    )


def build_artist_index(collected: Collected) -> ArtistIndex:
    """Index the artists that appear anywhere in the output, and intern their tags."""
    meta: dict[int, Recommendation] = {}
    for by_user in collected.recs.values():
        for items in by_user.values():
            for rec in items:
                meta.setdefault(rec.artist_id, rec)
    for items in collected.top.values():
        for rec in items:
            meta.setdefault(rec.artist_id, rec)

    order = sorted(meta)
    tag_pos: dict[str, int] = {}
    for artist_id in order:
        for tag in meta[artist_id].tags:
            tag_pos.setdefault(tag, len(tag_pos))
    for rows in collected.genres.values():  # genre-mix tags need vocabulary slots too
        for row in rows:
            tag_pos.setdefault(row["tag"], len(tag_pos))

    return ArtistIndex(
        order=order,
        pos={artist_id: i for i, artist_id in enumerate(order)},
        meta=meta,
        tag_pos=tag_pos,
    )


def shard_of(user_id: int, shard_size: int = SHARD_SIZE) -> int:
    """Shard number for a user.

    Keyed on the user id itself rather than its position in the sorted list, so the
    browser can compute the filename straight from the query string and fetch the
    shard in parallel with meta.json instead of waiting on it.
    """
    return user_id // shard_size


# ------------------------------------------------------------------------- JSON payloads


def _cards(fields: tuple, metrics: dict, k: int) -> list[dict]:
    return [
        {
            "label": label.format(k=k),
            "value": format(float(metrics[key]), fmt),
            "hint": hint,
        }
        for label, key, fmt, hint in fields
    ]


def build_meta(service: RecommenderService, build: str) -> dict:
    """Method labels, descriptions, pre-formatted metric cards and the comparison chart."""
    methods = []
    for label in service.methods:
        metrics = service.metrics(label)
        k = int(metrics["k"])
        has_beyond = metrics.get("coverage") is not None
        methods.append(
            {
                "label": label,
                "description": service.description(label),
                "personalised": "Popularity" not in label,
                "metrics": {
                    key: (None if value is None else float(value))
                    for key, value in metrics.items()
                },
                "accuracy_cards": _cards(ACCURACY_FIELDS, metrics, k),
                "beyond_cards": _cards(BEYOND_FIELDS, metrics, k) if has_beyond else [],
            }
        )

    rows = service.comparison("precision_at_k")
    ceiling = max((row["value"] for row in rows), default=1.0) or 1.0
    comparison = [
        {
            "label": row["label"],
            "value": format(row["value"], ".3f"),
            "pct": round(100 * row["value"] / ceiling, 1),
        }
        for row in rows
    ]

    first = service.metrics(service.methods[0])
    return {
        "build": build,
        "k": int(first["k"]),
        "n_eval_users": int(first["n_users"]),
        "users": [int(u) for u in service.user_ids],
        "default_user": int(service.user_ids[0]),
        "shard_size": SHARD_SIZE,
        "methods": methods,
        "comparison": comparison,
    }


def build_artists(index: ArtistIndex, build: str) -> dict:
    """Columnar artist metadata - parallel arrays are far smaller than objects."""
    tags = [tag for tag, _ in sorted(index.tag_pos.items(), key=lambda kv: kv[1])]
    payload: dict = {
        "build": build,
        "tags": tags,
        "id": [],
        "name": [],
        "tag": [],
        "hue": [],
        "init": [],
    }
    for artist_id in index.order:
        rec = index.meta[artist_id]
        payload["id"].append(int(artist_id))
        payload["name"].append(rec.name)
        payload["tag"].append([index.tag_pos[t] for t in rec.tags])
        payload["hue"].append(int(rec.hue))
        # Shipped rather than derived in JS: Python's str.isalnum() is Unicode-aware
        # over the full category set and JS \p{L}|\p{N} is close but not identical,
        # and this catalogue is full of non-ASCII names.
        payload["init"].append(rec.initial)
    return payload


def build_images(index: ArtistIndex, images: dict[str, str | None], build: str) -> dict:
    """Artwork hashes, index-aligned with artists.json. Kept in its own file so it
    stays off the first-paint critical path."""
    return {
        "build": build,
        "img": [
            None if (v := images.get(index.meta[a].name)) == FAILED else v
            for a in index.order
        ],
    }


def build_shards(
    collected: Collected,
    index: ArtistIndex,
    methods: list[str],
    shard_size: int,
) -> dict[int, dict]:
    """Per-user payloads, grouped into shards. Artists are stored as integer positions."""
    shards: dict[int, dict] = {}
    for uid in collected.top:
        shards.setdefault(shard_of(uid, shard_size), {})[str(uid)] = {
            "t": [
                [index.pos[r.artist_id], int(r.plays or 0)] for r in collected.top[uid]
            ],
            "g": [
                [index.tag_pos[g["tag"]], int(g["count"])] for g in collected.genres[uid]
            ],
            "r": [
                [index.pos[r.artist_id] for r in collected.recs[label][uid]]
                for label in methods
            ],
        }
    return shards


def _dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def write_site(
    service: RecommenderService,
    out_dir: Path,
    resolve_images: Callable[[list[str]], dict[str, str | None]] | None = None,
    build: str = BUILD,
    shard_size: int = SHARD_SIZE,
) -> ArtistIndex:
    """Write the full static data bundle under ``out_dir/data``.

    ``resolve_images`` receives the artist names that actually appear and returns a
    name -> encoded-artwork mapping. Omit it to build without artwork (the UI falls
    back to coloured initials), which is what the tests do.
    """
    out_dir = Path(out_dir)
    data_dir = out_dir / "data"

    collected = collect(service)
    index = build_artist_index(collected)
    names = [index.meta[a].name for a in index.order]
    # Resolved before anything is deleted, so a long or failed artwork pass never leaves
    # the currently deployed bundle half-written.
    images = resolve_images(names) if resolve_images is not None else {}

    if data_dir.exists():
        shutil.rmtree(data_dir)  # never let stale shards survive a smaller run
    (data_dir / "users").mkdir(parents=True, exist_ok=True)

    _dump(data_dir / "meta.json", build_meta(service, build))
    _dump(data_dir / "artists.json", build_artists(index, build))
    _dump(data_dir / "images.json", build_images(index, images, build))
    for shard, payload in build_shards(
        collected, index, service.methods, shard_size
    ).items():
        _dump(data_dir / "users" / f"{shard:02d}.json", payload)
    return index


def _report(out_dir: Path) -> None:
    # Skip tooling dotfiles (.vercel/, .env.local) - they are not part of the bundle.
    files = sorted(
        p
        for p in out_dir.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(out_dir).parts)
    )
    total = sum(p.stat().st_size for p in files)
    print(f"\n{len(files)} files, {total / 1024:.0f} KB total")
    for path in files:
        if path.parent.name != "users":
            print(
                f"  {path.relative_to(out_dir)!s:<24} {path.stat().st_size / 1024:7.1f} KB"
            )
    shards = [p for p in files if p.parent.name == "users"]
    if shards:
        size = sum(p.stat().st_size for p in shards) / 1024
        print(f"  {'data/users/*.json':<24} {size:7.1f} KB  ({len(shards)} shards)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="do not hit Deezer; use whatever the cache already holds",
    )
    parser.add_argument(
        "--refresh-artwork",
        action="store_true",
        help="re-check artists currently cached as having no photo (use after a run "
        "that was throttled, which records false negatives)",
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    print("Fitting and evaluating all 8 methods (this is the cost the app used to")
    print("pay on every cold start - we pay it once, here)...", flush=True)
    service = RecommenderService()
    print(f"  service ready in {time.perf_counter() - start:.1f}s")

    cache = load_cache(IMAGE_CACHE)
    if args.refresh_artwork:
        stale = [name for name, value in cache.items() if value is None]
        for name in stale:
            del cache[name]
        print(f"  artwork: re-checking {len(stale)} artists cached as having no photo")

    def resolve(names: list[str]) -> dict[str, str | None]:
        if args.skip_images:
            print(f"  artwork: skipped ({len(cache)} cached entries reused)")
            return cache
        return fetch_images(names, cache, cache_path=IMAGE_CACHE)

    index = write_site(service, args.out, resolve_images=resolve)

    _report(args.out)
    print(
        f"\n{len(index.order)} artists, {len(index.tag_pos)} tags, "
        f"{len(service.user_ids)} users x {len(service.methods)} methods"
    )
    print(f"build={BUILD}  (web/app.js must declare the same BUILD)")
    print(f"done in {time.perf_counter() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
