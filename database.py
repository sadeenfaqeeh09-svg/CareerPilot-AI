"""
database.py — SQLite Progress Tracker for CareerPilot AI
=========================================================
This module handles all database operations using Python's
built-in sqlite3 library. No extra installation needed.

Schema:
    sessions table stores one row per completed mock interview.
"""

import sqlite3
import os
from datetime import datetime

# Path to the SQLite database file
DB_PATH = "careerpilot_progress.db"


def init_db():
    """
    Create the database and sessions table if they don't exist yet.
    Called once when the app starts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL,
            interview_type  TEXT NOT NULL,
            target_role     TEXT NOT NULL,
            score           REAL NOT NULL,
            weakness        TEXT,
            improvement_tip TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at: {DB_PATH}")


def save_session(date, interview_type, target_role, score, weakness, improvement_tip):
    """
    Save a completed interview session to the database.

    Args:
        date (str): Timestamp string, e.g. "2024-01-15 14:30"
        interview_type (str): e.g. "Technical Interview"
        target_role (str): e.g. "Data Scientist"
        score (float): Average score out of 10
        weakness (str): Main weakness detected
        improvement_tip (str): Suggested improvement action
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sessions (date, interview_type, target_role, score, weakness, improvement_tip)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (date, interview_type, target_role, score, weakness, improvement_tip))

    conn.commit()
    conn.close()
    print(f"[DB] Session saved — {interview_type} for {target_role}, score: {score}")


def get_all_sessions():
    """
    Retrieve all interview sessions ordered by most recent first.

    Returns:
        list of tuples: (id, date, interview_type, target_role, score, weakness, improvement_tip)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, date, interview_type, target_role, score, weakness, improvement_tip
        FROM sessions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_summary_stats():
    """
    Get aggregate statistics across all sessions.

    Returns:
        dict: {total_sessions, avg_score, best_score, worst_score, most_practiced_type}
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(score), MAX(score), MIN(score) FROM sessions")
    row = cursor.fetchone()

    # Find most practiced interview type
    cursor.execute("""
        SELECT interview_type, COUNT(*) as cnt
        FROM sessions
        GROUP BY interview_type
        ORDER BY cnt DESC
        LIMIT 1
    """)
    top_type = cursor.fetchone()

    conn.close()

    return {
        "total_sessions": row[0] or 0,
        "avg_score": round(row[1], 1) if row[1] else 0,
        "best_score": row[2] or 0,
        "worst_score": row[3] or 0,
        "most_practiced_type": top_type[0] if top_type else "None yet"
    }


def clear_all_sessions():
    """
    Delete all sessions (useful for demo resets).
    Returns number of rows deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
