"""Generate the final PDF report from the project's results and figures.

Assembles a multi-page report (narrative + comparison table + figures) with reportlab.
Run after `scripts/evaluate_baselines.py` and the EDA/evaluation notebooks have produced
`docs/method_comparison.csv` and `docs/figures/*.png`.

    python scripts/build_report.py   ->   docs/Recommender_Systems_Report.pdf
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
OUT = DOCS / "Recommender_Systems_Report.pdf"

NAVY = colors.HexColor("#0f1f3d")
ACCENT = colors.HexColor("#3a5bbf")
GREY = colors.HexColor("#555a66")

# ----------------------------------------------------------------------------- styles
_ss = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "title", parent=_ss["Title"], fontSize=24, textColor=NAVY, spaceAfter=6, leading=28
)
SUBTITLE = ParagraphStyle(
    "subtitle",
    parent=_ss["Normal"],
    fontSize=12,
    textColor=GREY,
    alignment=TA_CENTER,
    spaceAfter=18,
)
H1 = ParagraphStyle(
    "h1",
    parent=_ss["Heading1"],
    fontSize=15,
    textColor=NAVY,
    spaceBefore=14,
    spaceAfter=6,
)
H2 = ParagraphStyle(
    "h2",
    parent=_ss["Heading2"],
    fontSize=12,
    textColor=ACCENT,
    spaceBefore=10,
    spaceAfter=4,
)
BODY = ParagraphStyle(
    "body",
    parent=_ss["Normal"],
    fontSize=10,
    leading=15,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)
BULLET = ParagraphStyle(
    "bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3
)
CAPTION = ParagraphStyle(
    "caption",
    parent=_ss["Normal"],
    fontSize=8.5,
    textColor=GREY,
    alignment=TA_CENTER,
    spaceBefore=3,
    spaceAfter=12,
)

story: list = []


def p(text: str) -> None:
    story.append(Paragraph(text, BODY))


def h1(text: str) -> None:
    story.append(Paragraph(text, H1))


def h2(text: str) -> None:
    story.append(Paragraph(text, H2))


def bullets(items: list[str]) -> None:
    for it in items:
        story.append(Paragraph(it, BULLET, bulletText="-"))
    story.append(Spacer(1, 4))


def figure(name: str, caption: str, max_w: float = 16 * cm) -> None:
    path = FIG / name
    if not path.exists():
        return
    iw, ih = ImageReader(str(path)).getSize()
    w = max_w
    h = w * ih / iw
    story.append(Image(str(path), width=w, height=h))
    story.append(Paragraph(caption, CAPTION))


def comparison_table() -> None:
    df = pd.read_csv(DOCS / "method_comparison.csv", index_col=0)
    label = {
        "item_knn": "Item-item CF",
        "als_mf": "ALS (MF)",
        "user_knn": "User-user CF",
        "content_based": "Content-based",
        "social": "Social (friends)",
        "popularity_listeners": "Pop (listeners)",
        "popularity_plays": "Pop (plays)",
        "popularity_damped": "Pop (damped)",
    }
    header = ["Method", "P@10", "NDCG", "Cov", "Div", "Nov", "PopBias", "fit s"]
    rows = [header]
    for idx, r in df.iterrows():
        rows.append(
            [
                label.get(idx, idx),
                f"{r['precision_at_k']:.3f}",
                f"{r['ndcg_at_k']:.3f}",
                f"{r['coverage']:.3f}",
                f"{r['diversity']:.3f}",
                f"{r['novelty']:.2f}",
                f"{r['popularity_bias']:.3f}",
                f"{r['fit_seconds']:.2f}",
            ]
        )
    t = Table(rows, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#eef1f8")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ced8")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(
        Paragraph(
            "Table 1. All methods, mean over 5 random splits (k=10, ~1,884 users; "
            "per-seed std was 0.001-0.003, so rankings are stable). Cov = catalogue "
            "coverage, Div = intra-list diversity, Nov = novelty (bits), PopBias = mean "
            "popularity of recommended artists.",
            CAPTION,
        )
    )


def _content_variant_sentence() -> str:
    """One-line TF-IDF vs raw ablation result, read from the generated CSV."""
    df = pd.read_csv(DOCS / "content_tfidf_vs_raw.csv", index_col=0)
    tf = df.loc["TF-IDF"]
    raw = df.loc["raw counts"]
    return (
        "An ablation comparing TF-IDF tag weighting against raw L2-normalised tag counts "
        f"found them essentially equivalent on this data (Precision@10 {tf['precision_at_k']:.3f} "
        f"vs {raw['precision_at_k']:.3f}; NDCG {tf['ndcg_at_k']:.3f} vs {raw['ndcg_at_k']:.3f}), "
        "indicating the tag co-occurrence signal is already informative without inverse-"
        "document weighting."
    )


def examples_section() -> None:
    """Render side-by-side recommendation examples from the generated CSV."""
    df = pd.read_csv(DOCS / "recommendation_examples.csv")
    p(
        "To show how the algorithms differ in practice, the table below gives the top-5 "
        "recommendations for three users from a representative set of methods. Each user's "
        "actual most-played artists are listed so the fit can be judged directly."
    )
    for user in df["user_id"].unique():
        sub = df[df["user_id"] == user]
        story.append(
            Paragraph(
                f"<b>User {user}</b> - listens to: {sub['user_taste'].iloc[0]}", BODY
            )
        )
        rows = [["Method", "Top-5 recommendations"]]
        rows.extend([[r["method"], r["recommendations"]] for _, r in sub.iterrows()])
        t = Table(rows, colWidths=[3.4 * cm, 12.6 * cm], hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#eef1f8")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ced8")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))
    p(
        "The contrast is clearest for the niche user: popularity recommends mainstream pop "
        "(Lady Gaga, Britney Spears) to an ambient / modern-classical listener, while the "
        "personalised methods correctly surface ambient and IDM artists. Item-item CF stays "
        "closest to the exact niche, content-based pulls tag-similar artists, social surfaces "
        "what friends play, and ALS generalises - occasionally drifting toward the mainstream "
        "for very niche tastes."
    )


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm, "Recommender Systems - Individual Project")
    canvas.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------- content
def build() -> None:
    # Title block
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Music Recommender Systems", TITLE))
    story.append(Paragraph("Eight Methods, One Honest Comparison", TITLE))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Individual Project - ESADE MSc - Recommender Systems (Prof. Marc Torrens)<br/>"
            "Dataset: Last.fm HetRec 2011 (implicit feedback)",
            SUBTITLE,
        )
    )
    story.append(Spacer(1, 0.6 * cm))

    h1("1. Executive summary")
    p(
        "This project builds a music recommender prototype on the Last.fm HetRec 2011 "
        "implicit-feedback dataset and compares eight methods - three non-personalised "
        "popularity baselines, item-item and user-user collaborative filtering, content-based "
        "filtering, matrix factorisation, and a friendship-based social recommender - using "
        "both accuracy and beyond-accuracy metrics, averaged over five random splits. The "
        "central finding is that personalisation more than doubles baseline accuracy "
        "(item-item CF reaches Precision@10 = 0.177 versus 0.069 for the best popularity "
        "baseline), but no single method wins on every objective. Accuracy, diversity, "
        "novelty, catalogue coverage, popularity bias, and scalability pull in different "
        "directions, so the recommended production design is a portfolio rather than one "
        "model."
    )

    h1("2. Problem and data")
    p(
        "The music track uses implicit feedback only: the signal is how many times each user "
        "played each artist, with no explicit ratings and therefore no observed dislikes. This "
        "shapes every downstream decision - preference must be inferred as confidence, and "
        "evaluation is a ranking problem (did we surface the held-out artists?) rather than "
        "rating prediction."
    )
    bullets(
        [
            "<b>Scale:</b> 1,892 users x 17,632 artists, 92,834 interactions.",
            "<b>Sparsity:</b> 99.72% of the user-artist matrix is empty.",
            "<b>Side data:</b> 11,946 tags with 186,479 (user, artist, tag) assignments "
            "(used by content-based filtering), and a 25,434-edge user friendship graph "
            "(used by the social recommender).",
        ]
    )

    h1("3. Exploratory analysis: know your data")
    p(
        "Listening is extremely concentrated. The Gini coefficient of plays-per-artist is "
        "0.893; the top 100 artists account for 43.7% of all listening, while the median artist "
        "has just a single listener. This long tail is the empirical basis for the project's "
        "thesis: a recommender optimised only for accuracy can lean on a tiny set of famous "
        "artists and still look respectable, while ignoring almost the entire catalogue."
    )
    figure(
        "long_tail_lorenz.png",
        "Figure 1. Artist popularity long tail (left) and the Lorenz curve of play "
        "concentration (right, Gini = 0.893).",
    )
    p(
        "Raw play counts span several orders of magnitude, which motivates log-scaling and "
        "confidence weighting before any model sees the data (Figure 2)."
    )
    figure(
        "play_count_distribution.png",
        "Figure 2. Raw play counts (left) versus log(1 + plays) (right).",
    )
    p(
        "User activity, by contrast, is remarkably uniform: HetRec keeps roughly the top 50 "
        "artists per user, so the most active users differ little from the median (the "
        "EDA notebook lists the top 10). This near-absence of low-activity users is why the "
        "cold-start analysis later finds essentially no cold users to stress-test."
    )

    story.append(PageBreak())
    h1("4. System design and engineering decisions")
    p(
        "The codebase is organised as a reusable <b>src/recsys</b> package (data, models, "
        "eval, utils) with a thin Flask prototype on top and notebooks reserved for "
        "exploration. Several deliberate decisions shaped the work:"
    )
    bullets(
        [
            "<b>One model interface.</b> Every recommender implements the same fit / recommend "
            "contract, so the evaluation harness and the app treat all methods interchangeably. "
            "This is what makes a fair, apples-to-apples comparison possible.",
            "<b>One shared split and harness.</b> A single per-user leave-out split and a single "
            "scoring loop are reused by every method; swapping the model keeps everything else "
            "constant.",
            "<b>Hybrid implementation.</b> Core algorithms (cosine kNN, TF-IDF profiles, ALS) "
            "are hand-implemented to demonstrate understanding, while numpy / scipy / sklearn "
            "provide the heavy linear algebra for speed.",
            "<b>Test-driven development.</b> 81 tests at 94% coverage, including closed-form and "
            "known-answer checks for the numerically tricky pieces (metrics, the ALS update).",
            "<b>Agile, vertical-slice first.</b> A thin end-to-end path (load -> popularity -> "
            "one screen -> one metric) was built before deepening any module, de-risking "
            "integration early.",
        ]
    )
    p(
        "Implicit feedback is handled with the Hu, Koren and Volinsky (2008) confidence "
        "formulation, c = 1 + alpha * log(1 + plays): every observed play is a weak-to-strong "
        "vote of confidence, sub-linear in the count so that a few superfans cannot dominate."
    )

    h1("5. The eight methods and their reasoning")
    h2("5.1 Popularity baselines (non-personalised)")
    p(
        "Three definitions of popular - total plays, distinct listeners, and a damped variant "
        "that suppresses mega-hits. These are the comparison floor and the cold-user fallback. "
        "Counting distinct listeners beats counting total plays, because total plays let a "
        "handful of superfans distort the ranking. Complexity is trivial (a single sorted "
        "aggregate)."
    )
    h2("5.2 Collaborative filtering (item-item and user-user kNN)")
    p(
        "Item-item CF scores an artist by its cosine similarity to the artists a user already "
        "plays; user-user CF scores by what similar-taste users play. Cosine similarity is "
        "hand-implemented as row-normalisation followed by a sparse matrix product, with "
        "optional top-k neighbour pruning to bound memory. Item-item is the stronger and more "
        "scalable choice under extreme sparsity because artist-to-artist co-occurrence is a "
        "more stable signal than user-to-user overlap."
    )
    h2("5.3 Content-based filtering (artist tags)")
    p(
        "Each artist is represented as a TF-IDF vector over its tags, so distinctive tags "
        "outweigh generic ones. A user profile is the play-weighted average of the tag vectors "
        "of the artists they play, and recommendations are the nearest artists in tag space. "
        "Its structural advantage is cold-start: it can recommend an artist with no listening "
        "history at all, which neither CF nor matrix factorisation can do."
    )
    p(_content_variant_sentence())
    h2("5.4 Matrix factorisation (implicit ALS)")
    p(
        "Implicit Alternating Least Squares (Hu, Koren and Volinsky 2008) is implemented from "
        "scratch. It learns a short latent factor vector per user and per artist whose dot "
        "product reconstructs the confidence-weighted preferences. ALS alternates between two "
        "closed-form ridge-regression half-steps (fix items, solve users; fix users, solve "
        "items), using the YtY + Yt(C-I)Y trick so the shared term is computed once per "
        "iteration. Correctness is verified against a closed-form single-step solution and a "
        "synthetic two-cluster dataset, not only against a library. Hyperparameters were "
        "chosen by a grid search on a validation slice carved from training (never the test "
        "set); the validation winner beat the shipped configuration by only about 2% at "
        "twice the fit time, so the cheaper configuration was kept."
    )
    h2("5.5 Social recommender (friendship graph)")
    p(
        "No other method uses the dataset's friendship graph, so a social recommender was "
        "added: a user's score for an artist is the log-scaled amount their friends play it. "
        "Structurally it is user-user collaborative filtering with the neighbour set fixed to "
        "declared friends rather than taste-similar users - a useful contrast, since social "
        "ties and statistical taste similarity are not the same thing."
    )

    h1("6. Evaluation methodology")
    p(
        "All methods are scored on the same per-user leave-out split across three metric "
        "families, and every result is averaged over five random splits to remove "
        "single-split fragility; the per-seed standard deviation was only 0.001 to 0.003, so "
        "the rankings are stable:"
    )
    bullets(
        [
            "<b>Accuracy:</b> Precision@10, Recall@10, MAP@10, NDCG@10.",
            "<b>Beyond-accuracy:</b> catalogue coverage, intra-list diversity (1 - mean pairwise "
            "tag cosine), novelty (mean self-information), and popularity bias (mean recommended "
            "popularity plus recommendation exposure Gini).",
            "<b>Operational:</b> training time and per-request serving latency.",
        ]
    )

    h1("7. Results")
    p(
        "Personalisation clearly works: every personalised method beats every popularity "
        "baseline on accuracy, and item-item CF roughly doubles to triples the baseline across "
        "ranking metrics (Figure 3). Notably, the simpler item-item CF beats the more complex "
        "ALS on this small, dense dataset - a reminder that sophistication is not automatically "
        "valuable."
    )
    figure("eval_accuracy.png", "Figure 3. Accuracy at k=10 by method.")
    p(
        "But accuracy is not the whole story. Figure 4 shows that no method wins on coverage, "
        "diversity, and novelty simultaneously, and Figure 5 makes the accuracy-versus-"
        "diversity tension explicit."
    )
    figure(
        "eval_beyond_accuracy.png",
        "Figure 4. Beyond-accuracy metrics: coverage, diversity, novelty.",
    )
    figure("eval_tradeoff.png", "Figure 5. The accuracy versus diversity trade-off.")

    story.append(PageBreak())
    p(
        "Popularity bias is measurable and large. Popularity recommenders show the same handful "
        "of artists to everyone (catalogue coverage near 0.001 and the highest exposure Gini), "
        "while content-based filtering is the least biased toward popular artists. Even "
        "user-user CF, though personalised, leans noticeably on crowd-pleasers. Training cost "
        "and serving cost also differ sharply by method family (Figure 6)."
    )
    figure(
        "eval_scalability_bias.png",
        "Figure 6. Popularity bias via exposure Gini (left) and training cost versus "
        "accuracy (right).",
    )
    comparison_table()

    h2("7.1 The social recommender")
    p(
        "The friendship-based model is a strong, cheap addition: Precision@10 of 0.123 nearly "
        "matches user-user CF (0.131) but with far broader catalogue coverage (0.143 versus "
        "0.018) and higher diversity. In other words, who your friends are is almost as "
        "predictive as who is statistically similar to you, and it spreads recommendations "
        "across a much wider slice of the catalogue."
    )
    h2("7.2 Cold-start")
    p(
        "A cold-start analysis exposes a structural divide. Collaborative filtering, ALS, the "
        "social model, and popularity can only rank artists seen in training, so of the 2,211 "
        "artists with zero training interactions they recommend none; content-based filtering, "
        "needing only tags, can recommend 1,227 of those cold artists - its defining "
        "advantage. On the user side this dataset is a weak stress test: HetRec keeps roughly "
        "the top 50 artists per user, so almost every user has a similar amount of history and "
        "accuracy is essentially flat across history quartiles (Figure 7); genuine cold users "
        "barely exist here."
    )
    figure("cold_start.png", "Figure 7. Cold-user accuracy by training-history quartile.")

    story.append(PageBreak())
    h1("8. Recommendation examples")
    examples_section()

    h1("9. Critical analysis")
    bullets(
        [
            "<b>No universal winner.</b> Item-item CF leads accuracy and coverage; ALS gives the "
            "most diverse of the accurate lists and the lowest popularity bias among strong "
            "methods; content-based is the most novel but the least diverse (a tag filter "
            "bubble); popularity is a cheap floor that is heavily biased and repetitive.",
            "<b>Bias hides inside 'personalised'.</b> User-user CF has higher popularity bias "
            "(0.140) than item-item CF (0.084); personalised does not automatically mean fair.",
            "<b>Cold-start, measured.</b> Of 2,211 artists with no training plays, the "
            "collaborative, ALS, social, and popularity methods can recommend none, while "
            "content-based can recommend 1,227 of them - a concrete, quantified advantage.",
            "<b>Social ties are predictive.</b> The friendship recommender nearly matches "
            "user-user CF on accuracy (0.123 vs 0.131) with far higher coverage and "
            "diversity; declared friends are almost as useful as statistical neighbours.",
            "<b>Scalability is a trade, not a winner.</b> Memory-based CF trains in about 0.02s "
            "but stores an item-item similarity matrix that grows with the catalogue squared; "
            "ALS pays about 6.3s of training but serves from compact factors in about 0.2ms.",
            "<b>Honesty about metrics.</b> Exposure Gini is high for every method because ten "
            "slots across 1,884 users can only ever touch a fraction of 17,632 artists; it is "
            "read as a relative concentration measure, and intra-list diversity is only "
            "meaningful in tag space.",
        ]
    )

    h1("10. Honest limitations and threats to validity")
    p(
        "Several earlier shortcomings were fixed rather than excused: results are now averaged "
        "over five splits, ALS was tuned on a validation set, cold-start was measured, and the "
        "friendship graph was turned into a working method. The limitations that remain are "
        "largely inherent to the dataset and to offline evaluation, and are stated explicitly "
        "below."
    )
    bullets(
        [
            "<b>No temporal split is possible.</b> The interaction file has no timestamps "
            "(only aggregate play counts), so a time-based split cannot be built. Evaluation "
            "therefore uses a random per-user hold-out, which can leak later listens and "
            "likely overstates real-world performance; this is a dataset constraint, not a "
            "design choice.",
            "<b>Accuracy rewards re-discovery, not discovery.</b> Relevant items are artists "
            "the user already played and we held out, so accuracy favours re-finding known "
            "tastes and is structurally biased toward popular items. This is inherent to "
            "offline evaluation - measuring genuine discovery needs an online A/B test or user "
            "study; the beyond-accuracy metrics soften but cannot remove it.",
            "<b>ALS validated by tests, not by a reference library.</b> Hyperparameters were "
            "grid-searched on a validation split, but the hand-written ALS is checked only "
            "against a closed-form solution and a synthetic dataset, not the implicit library.",
            "<b>Beyond-accuracy blind spots.</b> Intra-list diversity is computed only in tag "
            "space (untagged artists are skipped), novelty is derived from training popularity, "
            "and exposure Gini is near-saturated for every method because ten slots across "
            "17,632 artists can only ever touch a fraction of the catalogue.",
            "<b>Small, dense dataset.</b> HetRec keeps roughly the top 50 artists per user, "
            "which makes the popularity baseline unusually strong, flatters the neighbourhood "
            "methods, and (as the cold-start analysis confirmed) leaves almost no genuine cold "
            "users to test; generalisation to a larger, sparser catalogue is untested.",
            "<b>Scalability is descriptive only.</b> Training and serving costs are reported at "
            "this dataset's scale; none of the methods were stress-tested at production scale.",
            "<b>Prototype UX.</b> The interface is functional and not user-tested.",
        ]
    )

    h1("11. Conclusion and recommendation")
    p(
        "Accuracy is necessary but not sufficient. The right production system for this dataset "
        "is a portfolio: ship item-item CF as the default for its strong accuracy, broad "
        "coverage, and near-instant training; blend in content-based filtering for cold-start "
        "coverage and novelty; use the social recommender to widen coverage where a friendship "
        "graph exists; and keep a popularity model as the fallback for brand-new users. "
        "Natural next steps are hybrid score blending, a learning-to-rank layer, and a "
        "cross-check of the hand-written ALS against the implicit library."
    )
    p(
        "<b>Reproducibility.</b> The dataset is fetched by scripts/download_data.py; "
        "scripts/build_processed.py caches the processed artifacts; scripts/evaluate_baselines."
        "py regenerates the multi-seed comparison; scripts/tune_als.py reproduces the ALS "
        "validation search; scripts/cold_start.py the cold-start analysis; "
        "scripts/recommendation_examples.py the examples; scripts/content_variant.py the "
        "TF-IDF ablation; `python main.py` runs the whole pipeline end to end; the notebooks "
        "regenerate every figure; and the full test suite (81 tests, 94% coverage) guards "
        "correctness."
    )

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Music Recommender Systems - Final Report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
