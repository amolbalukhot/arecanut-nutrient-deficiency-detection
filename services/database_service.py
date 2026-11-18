"""SQLite persistence using parameterised SQL only."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config


class DatabaseError(Exception):
    pass


@contextmanager
def _conn(db_path=None):
    c = sqlite3.connect(str(db_path or config.DATABASE_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except sqlite3.Error as e:
        c.rollback()
        raise DatabaseError("Database operation failed.") from e
    finally:
        c.close()


def init_db(db_path=None):
    path = Path(db_path or config.DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = (config.BASE_DIR / "schema.sql").read_text()
    with _conn(path) as c:
        c.executescript(schema)  


def insert_prediction(rec: dict, db_path=None) -> int:
    with _conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO predictions (image_path, verification_class, verification_confidence,
               verification_accepted, nutrient_class, nutrient_confidence, heatmap_path, recommendation, status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (rec["image_path"], rec.get("verification_class"), rec.get("verification_confidence"),
             int(bool(rec.get("verification_accepted"))), rec.get("nutrient_class"), rec.get("nutrient_confidence"),
             rec.get("heatmap_path"), rec.get("recommendation"), rec.get("status", "ok")))
        return cur.lastrowid


def update_heatmap(pid: int, heatmap_path: str | None, db_path=None) -> None:
    """Attach an asynchronously generated heatmap to an existing prediction."""
    with _conn(db_path) as c:
        c.execute("UPDATE predictions SET heatmap_path = ? WHERE id = ?", (heatmap_path, pid))


def get_prediction(pid: int, db_path=None):
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone()
        return dict(row) if row else None


def get_history(page=1, per_page=10, search="", db_path=None):
    page = max(1, page)
    like = f"%{search}%"
    where, params = ("WHERE nutrient_class LIKE ? OR verification_class LIKE ? OR CAST(id AS TEXT) = ?",
                     (like, like, search)) if search else ("", ())
    with _conn(db_path) as c:
        total = c.execute(f"SELECT COUNT(*) FROM predictions {where}", params).fetchone()[0]
        rows = c.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                         (*params, per_page, (page - 1) * per_page)).fetchall()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "per_page": per_page,
            "pages": max(1, -(-total // per_page))}


def recent_predictions(limit=5, db_path=None):
    with _conn(db_path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]


def dashboard_stats(db_path=None):
    with _conn(db_path) as c:
        classified = c.execute("SELECT COUNT(*) FROM predictions WHERE nutrient_class IS NOT NULL").fetchone()[0]
        rejected = c.execute("SELECT COUNT(*) FROM predictions WHERE verification_accepted = 0").fetchone()[0]
        counts = {r[0]: r[1] for r in c.execute(
            "SELECT nutrient_class, COUNT(*) FROM predictions WHERE nutrient_class IS NOT NULL GROUP BY nutrient_class")}
        trend = [{"date": r[0], "count": r[1]} for r in c.execute(
            """SELECT substr(created_at,1,10) d, COUNT(*) FROM predictions WHERE nutrient_class IS NOT NULL
               GROUP BY d ORDER BY d""")]
    per_class = {k: counts.get(k, 0) for k in config.CLASS_NAMES}
    healthy = per_class["Healthy"]
    return {"total": classified, "rejected": rejected, "per_class": per_class, "healthy": healthy,
            "healthy_pct": round(100 * healthy / classified, 1) if classified else 0.0, "trend": trend}
