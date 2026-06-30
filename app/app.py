"""Flask prototype - one screen: pick a user, see recommendations + the offline metric.

Run with:
    flask --app app/app.py run --debug
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, render_template, request

# Make both the project root (for the `app` package) and src/ (for `recsys`)
# importable when running via `flask --app app/app.py` from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from app.service import RecommenderService  # noqa: E402

app = Flask(__name__)

# Build the service once at startup (loads data, fits every method).
service = RecommenderService()


@app.route("/")
def index() -> str:
    """Render the recommendation screen for the selected user and method."""
    raw_user = request.args.get("user_id", "")
    selected_user = int(raw_user) if raw_user.isdigit() else service.user_ids[0]
    selected_method = request.args.get("method", service.methods[0])

    recommendations = service.attach_images(
        service.recommend(selected_method, selected_user)
    )
    top_artists = service.attach_images(service.user_top_artists(selected_user))
    metrics = service.metrics(selected_method)
    k = int(metrics["k"])

    # Metric cards with plain-language hints, so a first-time viewer understands them.
    accuracy_cards = [
        {
            "label": f"Precision@{k}",
            "value": f"{metrics['precision_at_k']:.3f}",
            "hint": "of the picks shown, the share the user actually liked",
        },
        {
            "label": f"Recall@{k}",
            "value": f"{metrics['recall_at_k']:.3f}",
            "hint": "of everything the user liked, the share we surfaced",
        },
        {
            "label": f"MAP@{k}",
            "value": f"{metrics['map_at_k']:.3f}",
            "hint": "rewards putting the right items near the top",
        },
        {
            "label": f"NDCG@{k}",
            "value": f"{metrics['ndcg_at_k']:.3f}",
            "hint": "overall ranking quality (top hits count more)",
        },
    ]
    beyond_cards = []
    if metrics.get("coverage") is not None:
        beyond_cards = [
            {
                "label": "Coverage",
                "value": f"{metrics['coverage']:.3f}",
                "hint": "share of the whole catalogue this method ever recommends",
            },
            {
                "label": "Diversity",
                "value": f"{metrics['diversity']:.3f}",
                "hint": "how varied the artists within one list are",
            },
            {
                "label": "Novelty",
                "value": f"{metrics['novelty']:.2f}",
                "hint": "how non-mainstream the picks are (higher = deeper cuts)",
            },
        ]

    # Visuals (pure-CSS bar charts).
    comparison = service.comparison("precision_at_k")
    comparison_max = max((r["value"] for r in comparison), default=1.0) or 1.0
    genre_mix = service.user_genre_mix(selected_user)
    genre_max = max((g["count"] for g in genre_mix), default=1)

    return render_template(
        "index.html",
        user_ids=service.user_ids,
        selected_user=selected_user,
        recommendations=recommendations,
        top_artists=top_artists,
        method_description=service.description(selected_method),
        methods=service.methods,
        selected_method=selected_method,
        is_personalised="Popularity" not in selected_method,
        accuracy_cards=accuracy_cards,
        beyond_cards=beyond_cards,
        comparison=comparison,
        comparison_max=comparison_max,
        genre_mix=genre_mix,
        genre_max=genre_max,
        k=k,
        n_eval_users=int(metrics["n_users"]),
    )


if __name__ == "__main__":
    app.run(debug=True)
