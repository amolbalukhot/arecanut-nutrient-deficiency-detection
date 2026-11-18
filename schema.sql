CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    verification_class TEXT,
    verification_confidence REAL,
    verification_accepted INTEGER NOT NULL DEFAULT 0,
    nutrient_class TEXT,
    nutrient_confidence REAL,
    heatmap_path TEXT,
    recommendation TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now'))
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
