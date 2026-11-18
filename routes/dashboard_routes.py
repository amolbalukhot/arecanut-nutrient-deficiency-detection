from flask import Blueprint, current_app, render_template, request

from services import database_service

bp = Blueprint("dashboard", __name__)


@bp.get("/dashboard")
def dashboard():
    db = current_app.config["DATABASE_PATH"]
    return render_template("dashboard.html", stats=database_service.dashboard_stats(db),
                           recent=database_service.recent_predictions(5, db))


@bp.get("/history")
def history():
    page = request.args.get("page", 1, type=int)
    q = request.args.get("q", "").strip()[:50]
    return render_template("history.html", h=database_service.get_history(page, 10, q, current_app.config["DATABASE_PATH"]), q=q)
