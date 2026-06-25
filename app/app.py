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

    recommendations = service.recommend(selected_method, selected_user)
    metrics = service.metrics(selected_method)
    return render_template(
        "index.html",
        user_ids=service.user_ids,
        selected_user=selected_user,
        recommendations=recommendations,
        top_artists=service.user_top_artists(selected_user),
        method_description=service.description(selected_method),
        methods=service.methods,
        selected_method=selected_method,
        precision=metrics["precision_at_k"],
        recall=metrics["recall_at_k"],
        mean_ap=metrics["map_at_k"],
        ndcg=metrics["ndcg_at_k"],
        coverage=metrics.get("coverage"),
        diversity=metrics.get("diversity"),
        novelty=metrics.get("novelty"),
        k=int(metrics["k"]),
        n_eval_users=int(metrics["n_users"]),
    )


if __name__ == "__main__":
    app.run(debug=True)
