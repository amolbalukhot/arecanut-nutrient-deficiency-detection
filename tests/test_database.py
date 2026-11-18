from services import database_service as db


def rec(**kw):
    base = dict(image_path="a.png", verification_class="Arecanut", verification_confidence=0.9,
                verification_accepted=True, nutrient_class="Healthy", nutrient_confidence=0.8,
                heatmap_path=None, recommendation="r")
    return {**base, **kw}


def test_empty_db(tmp_path):
    p = tmp_path / "e.db"; db.init_db(p)
    s = db.dashboard_stats(p)
    assert s["total"] == 0 and s["healthy_pct"] == 0.0 and db.get_history(db_path=p)["total"] == 0


def test_insert_fetch_stats_history(tmp_path):
    p = tmp_path / "d.db"; db.init_db(p)
    ids = [db.insert_prediction(rec(), p), db.insert_prediction(rec(nutrient_class="Nitrogen_Deficiency"), p),
           db.insert_prediction(rec(verification_accepted=False, nutrient_class=None, nutrient_confidence=None,
                                    status="rejected"), p)]
    assert db.get_prediction(ids[1], p)["nutrient_class"] == "Nitrogen_Deficiency"
    assert db.get_prediction(999, p) is None
    s = db.dashboard_stats(p)
    assert s["total"] == 2 and s["rejected"] == 1 and s["healthy"] == 1 and s["healthy_pct"] == 50.0
    h = db.get_history(1, 2, db_path=p)
    assert h["total"] == 3 and h["pages"] == 2 and h["items"][0]["id"] == ids[2]
    assert db.get_history(search="Nitrogen", db_path=p)["total"] == 1
    db.init_db(p)  # re-init keeps data
    assert db.get_history(db_path=p)["total"] == 3
    assert len(db.recent_predictions(2, p)) == 2


def test_sql_injection_search_is_safe(tmp_path):
    p = tmp_path / "s.db"; db.init_db(p); db.insert_prediction(rec(), p)
    assert db.get_history(search="'; DROP TABLE predictions;--", db_path=p)["total"] == 0
    assert db.get_history(db_path=p)["total"] == 1
