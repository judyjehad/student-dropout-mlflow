"""
Student Dropout — Early Warning Model
monitoring.py

Monitors every prediction made by the API:
  - Logs input + output to SQLite database
  - Tracks confidence scores over time
  - Alerts when average confidence drops below threshold
  - Exposes /monitoring endpoint for live stats
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH             = "data/monitoring.db"
CONFIDENCE_THRESHOLD = 60.0  # alert if avg confidence drops below this

# ── 1. Setup database ─────────────────────────────────────────────────────────
def init_db():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            input_data      TEXT    NOT NULL,
            prediction      TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            prob_dropout    REAL,
            prob_enrolled   REAL,
            prob_graduate   REAL
        )
    """)
    conn.commit()
    conn.close()

# ── 2. Log a prediction ───────────────────────────────────────────────────────
def log_prediction(input_data: dict, result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
            (timestamp, input_data, prediction, confidence,
             prob_dropout, prob_enrolled, prob_graduate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        json.dumps(input_data),
        result["prediction"],
        result["confidence"],
        result["probabilities"].get("Dropout",  0.0),
        result["probabilities"].get("Enrolled", 0.0),
        result["probabilities"].get("Graduate", 0.0),
    ))
    conn.commit()
    conn.close()

# ── 3. Get monitoring stats ───────────────────────────────────────────────────
def get_stats() -> dict:
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total predictions
    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    if total == 0:
        conn.close()
        return {
            "total_predictions": 0,
            "message": "No predictions logged yet.",
        }

    # Average confidence
    avg_confidence = cursor.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]

    # Prediction distribution
    distribution = cursor.execute("""
        SELECT prediction, COUNT(*) as count
        FROM predictions
        GROUP BY prediction
        ORDER BY count DESC
    """).fetchall()

    # Last 10 confidence scores (for trend)
    recent = cursor.execute("""
        SELECT timestamp, prediction, confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    # Drift alert: avg confidence of last 20 predictions
    recent_avg = cursor.execute("""
        SELECT AVG(confidence)
        FROM (
            SELECT confidence FROM predictions
            ORDER BY id DESC LIMIT 20
        )
    """).fetchone()[0]

    conn.close()

    drift_alert = (
        recent_avg is not None and recent_avg < CONFIDENCE_THRESHOLD
    )

    return {
        "total_predictions":    total,
        "average_confidence":   round(avg_confidence, 1),
        "recent_avg_confidence": round(recent_avg, 1) if recent_avg else None,
        "drift_alert":          drift_alert,
        "drift_threshold":      CONFIDENCE_THRESHOLD,
        "alert_message":        (
            f"⚠️  Average confidence dropped to {round(recent_avg, 1)}% "
            f"(threshold: {CONFIDENCE_THRESHOLD}%)"
            if drift_alert else "✅ Model confidence is stable"
        ),
        "prediction_distribution": {
            row[0]: row[1] for row in distribution
        },
        "recent_predictions": [
            {
                "timestamp":  row[0],
                "prediction": row[1],
                "confidence": row[2],
            }
            for row in recent
        ],
    }

# ── 4. Initialize on import ───────────────────────────────────────────────────
init_db()