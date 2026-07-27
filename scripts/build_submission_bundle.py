"""Regenerate Final_Submission/ - the clean hand-in copy of the project.

The bundle is the committed tree minus the internal working documents (the AI working
instructions, the build plan, the assignment brief, the superseded deploy config). It was
previously assembled by hand, which is how it drifted: by June it was missing the static
site entirely and still shipped a README pointing at a dead URL.

Because it is built from ``git ls-files``, it always reflects what is actually committed -
so commit first, then run this.

Usage:
    python scripts/build_submission_bundle.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Final_Submission"

# Internal working documents - useful to us, noise (or confusing) to a marker.
EXCLUDE_FILES = {
    "CLAUDE.md",  # AI working instructions
    "PLAN.md",  # internal phased build plan
    "IndividualProject.pdf",  # the assignment brief itself
    "render.yaml",  # superseded deployment config
}
EXCLUDE_PREFIXES = ("web/.",)  # Vercel tooling dotfiles


def tracked_files() -> list[str]:
    """Every file git currently tracks, relative to the project root."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def wanted(path: str) -> bool:
    if path in EXCLUDE_FILES:
        return False
    return not path.startswith(EXCLUDE_PREFIXES)


def build() -> None:
    """Rebuild the bundle from scratch so deletions propagate too."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print(
            "! working tree has uncommitted changes; the bundle mirrors HEAD's file list"
        )

    paths = [p for p in tracked_files() if wanted(p)]
    if OUT.exists():
        shutil.rmtree(OUT)

    for rel in paths:
        src = ROOT / rel
        if not src.exists():  # tracked but deleted locally
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(
        f"Wrote {OUT.relative_to(ROOT)}: {len(paths)} files, {total / 1024 / 1024:.1f} MB"
    )
    for name in sorted(p.name for p in OUT.iterdir()):
        print(f"  {name}")


if __name__ == "__main__":
    build()
