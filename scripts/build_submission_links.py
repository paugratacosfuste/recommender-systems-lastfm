"""Generate the one-page submission sheet (docs/Submission_Links.pdf).

The links handed in have to survive independently of this repository, so they live in one
small generated document rather than being retyped. Previously this page was produced
ad hoc, which is how it ended up still advertising a dead deployment; generating it from
a script keeps the URLs in exactly one place.

Usage:
    python scripts/build_submission_links.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "Submission_Links.pdf"

TITLE = "Music Recommender System"
SUBTITLE = "Individual Project - ESADE MSc, Recommender Systems (Prof. Marc Torrens)"
AUTHOR = "Pau Gratacós"
DATE = "27/07/2026"

APP_URL = "https://music-recommender-lastfm.vercel.app"
REPO_URL = "https://github.com/paugratacosfuste/recommender-systems-lastfm"

INK = HexColor("#1a2b4a")
LINK = HexColor("#1f5fa8")
MUTED = HexColor("#444444")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    return {
        "title": ParagraphStyle(
            "title",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=1,
            textColor=INK,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base,
            fontSize=10.5,
            leading=14,
            alignment=1,
            textColor=MUTED,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=18,
            textColor=INK,
        ),
        "label": ParagraphStyle(
            "label",
            parent=base,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            spaceBefore=10,
            textColor=INK,
        ),
        "link": ParagraphStyle(
            "link", parent=base, fontSize=11, leading=15, textColor=LINK
        ),
        "note": ParagraphStyle(
            "note", parent=base, fontSize=9, leading=12.5, textColor=MUTED
        ),
    }


def build() -> None:
    """Write the one-page link sheet."""
    style = _styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        topMargin=5 * cm,
        bottomMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        title=f"{TITLE} - submission links",
        author=AUTHOR,
    )

    link = lambda url: f'<link href="{url}" color="#1f5fa8">{url}</link>'  # noqa: E731

    story = [
        Paragraph(TITLE, style["title"]),
        Paragraph(SUBTITLE, style["subtitle"]),
        Paragraph(f"{AUTHOR} · {DATE}", style["subtitle"]),
        Spacer(1, 0.9 * cm),
        Paragraph("Submission links", style["section"]),
        Paragraph("Live app:", style["label"]),
        Paragraph(link(APP_URL), style["link"]),
        Paragraph("GitHub repository (code, report, notebooks, slides):", style["label"]),
        Paragraph(link(REPO_URL), style["link"]),
        Spacer(1, 0.7 * cm),
        Paragraph(
            "The app is a static build: all 1,892 listeners x 8 recommendation methods were "
            "scored offline in Python and shipped as pre-rendered data, so the page loads "
            "immediately and needs no server to start up. Pick a listener and a method to "
            "compare what each approach recommends and how it scores.",
            style["note"],
        ),
    ]
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
