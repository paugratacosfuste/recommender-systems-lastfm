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

# Build the service once at startup (loads data, fits the model).
service = RecommenderService()

# The method switcher is a stub for Phase 0 (only the popularity baseline exists yet).
AVAILABLE_METHODS = ["Popularity (plays)"]


@app.route("/")
def index() -> str:
    """Render the recommendation screen for the selected (or first) user."""
    raw_user = request.args.get("user_id", "")
    selected_user = int(raw_user) if raw_user.isdigit() else service.user_ids[0]

    recommendations = service.recommend(selected_user)
    return render_template(
        "index.html",
        user_ids=service.user_ids,
        selected_user=selected_user,
        recommendations=recommendations,
        methods=AVAILABLE_METHODS,
        precision=service.precision_at_k,
        k=service.k,
        n_eval_users=service.n_eval_users,
    )


if __name__ == "__main__":
    app.run(debug=True)
