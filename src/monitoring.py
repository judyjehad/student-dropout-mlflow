"""
Student Dropout — Early Warning Model
monitoring.py

Monitors every prediction made by the API:
  - Logs input + output to SQLite database
  - Tracks confidence scores over time
  - Alerts when average confidence drops below threshold
  - Stores feedback (actual outcomes) for real accuracy tracking
  - Exposes stats, alerts, and feedback via serve_model.py
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH              = "data/monitoring.db"
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
            prob_graduate   REAL,
            actual_outcome  TEXT    DEFAULT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            alert_type  TEXT    NOT NULL,
            message     TEXT    NOT NULL,
            severity    TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# ── 2. Log a prediction ───────────────────────────────────────────────────────
def log_prediction(input_data: dict, result: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO predictions
            (timestamp, input_data, prediction, confidence,
             prob_dropout, prob_enrolled, prob_graduate)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        json.dumps(input_data),
        result["prediction"],
        result["confidence"],
        result["probabilities"].get("Dropout",  0.0),
        result["probabilities"].get("Enrolled", 0.0),
        result["probabilities"].get("Graduate", 0.0),
    ))
    prediction_id = cursor.lastrowid
    conn.commit()

    # Check drift and log alert if needed
    recent_avg = conn.execute("""
        SELECT AVG(confidence)
        FROM (SELECT confidence FROM predictions ORDER BY id DESC LIMIT 20)
    """).fetchone()[0]

    if recent_avg is not None and recent_avg < CONFIDENCE_THRESHOLD:
        conn.execute("""
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            "confidence_drift",
            f"Average confidence dropped to {round(recent_avg, 1)}% — below threshold of {CONFIDENCE_THRESHOLD}%",
            "warning",
        ))
        conn.commit()

    conn.close()
    return prediction_id

# ── 3. Log feedback (actual outcome) ─────────────────────────────────────────
def log_feedback(prediction_id: int, actual_outcome: str) -> dict:
    conn = sqlite3.connect(DB_PATH)

    # Check prediction exists
    row = conn.execute(
        "SELECT prediction, actual_outcome FROM predictions WHERE id = ?",
        (prediction_id,)
    ).fetchone()

    if not row:
        conn.close()
        return {"error": f"Prediction ID {prediction_id} not found"}

    predicted, existing_outcome = row

    if existing_outcome:
        conn.close()
        return {"error": f"Feedback already submitted for prediction {prediction_id}"}

    conn.execute(
        "UPDATE predictions SET actual_outcome = ? WHERE id = ?",
        (actual_outcome, prediction_id)
    )
    conn.commit()
    conn.close()

    correct = predicted == actual_outcome
    return {
        "prediction_id":  prediction_id,
        "predicted":      predicted,
        "actual_outcome": actual_outcome,
        "correct":        correct,
        "message":        "✅ Correct prediction" if correct else "❌ Incorrect prediction",
    }

# ── 4. Get monitoring stats ───────────────────────────────────────────────────
def get_stats() -> dict:
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    if total == 0:
        conn.close()
        return {"total_predictions": 0, "message": "No predictions logged yet."}

    avg_confidence = cursor.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]

    distribution = cursor.execute("""
        SELECT prediction, COUNT(*) as count
        FROM predictions
        GROUP BY prediction
        ORDER BY count DESC
    """).fetchall()

    recent = cursor.execute("""
        SELECT id, timestamp, prediction, confidence, actual_outcome
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    recent_avg = cursor.execute("""
        SELECT AVG(confidence)
        FROM (SELECT confidence FROM predictions ORDER BY id DESC LIMIT 20)
    """).fetchone()[0]

    # Real accuracy from feedback
    feedback_rows = cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN prediction = actual_outcome THEN 1 ELSE 0 END) as correct
        FROM predictions
        WHERE actual_outcome IS NOT NULL
    """).fetchone()

    conn.close()

    drift_alert = recent_avg is not None and recent_avg < CONFIDENCE_THRESHOLD

    feedback_total   = feedback_rows[0] or 0
    feedback_correct = feedback_rows[1] or 0
    real_accuracy    = round(feedback_correct / feedback_total * 100, 1) if feedback_total > 0 else None

    return {
        "total_predictions":     total,
        "average_confidence":    round(avg_confidence, 1),
        "recent_avg_confidence": round(recent_avg, 1) if recent_avg else None,
        "drift_alert":           drift_alert,
        "drift_threshold":       CONFIDENCE_THRESHOLD,
        "alert_message":         (
            f"⚠️  Average confidence dropped to {round(recent_avg, 1)}% "
            f"(threshold: {CONFIDENCE_THRESHOLD}%)"
            if drift_alert else "✅ Model confidence is stable"
        ),
        "prediction_distribution": {row[0]: row[1] for row in distribution},
        "feedback_summary": {
            "total_feedback":  feedback_total,
            "correct":         feedback_correct,
            "real_accuracy":   f"{real_accuracy}%" if real_accuracy else "No feedback yet",
        },
        "recent_predictions": [
            {
                "id":             row[0],
                "timestamp":      row[1],
                "prediction":     row[2],
                "confidence":     row[3],
                "actual_outcome": row[4],
            }
            for row in recent
        ],
    }

# ── 5. Get alerts ─────────────────────────────────────────────────────────────
def get_alerts(limit: int = 20) -> dict:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, alert_type, message, severity
        FROM alerts
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return {
        "total_alerts": len(rows),
        "alerts": [
            {
                "timestamp":  row[0],
                "alert_type": row[1],
                "message":    row[2],
                "severity":   row[3],
            }
            for row in rows
        ] if rows else [],
        "status": "⚠️  Active alerts found" if rows else "✅ No alerts",
    }

# ── 6. Initialize on import ───────────────────────────────────────────────────
init_db()