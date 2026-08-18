from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "drift"
        / "skills"
        / "feedback-preference-context"
        / "scripts"
        / "sample_feedback_context.py"
    )
    spec = importlib.util.spec_from_file_location("feedback_context_script", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_drift_db(path: Path, cursor_json: str = "{}") -> None:
    conn = sqlite3.connect(path)
    try:
        _ = conn.execute("""
            CREATE TABLE skill_continuum (
                skill_name TEXT PRIMARY KEY,
                cursor_json TEXT NOT NULL
            )
            """)
        _ = conn.execute(
            "INSERT INTO skill_continuum(skill_name, cursor_json) VALUES (?, ?)",
            ("feedback-preference-context", cursor_json),
        )
        conn.commit()
    finally:
        conn.close()


def _write_feedback_db(path: Path, rows: list[tuple[int, str]]) -> None:
    conn = sqlite3.connect(path)
    try:
        _ = conn.execute("""
            CREATE TABLE proactive_feedback_events (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                session_key TEXT NOT NULL,
                user_message_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                proactive_message_id TEXT,
                feedback_type TEXT NOT NULL,
                confidence TEXT NOT NULL,
                pa_score REAL,
                pua_score REAL,
                lag_seconds INTEGER,
                candidate_count INTEGER NOT NULL,
                matched_by TEXT NOT NULL,
                reason TEXT NOT NULL
            )
            """)
        for row_id, feedback_type in rows:
            _ = conn.execute(
                """
                INSERT INTO proactive_feedback_events(
                    id, created_at, session_key, user_message_id,
                    assistant_message_id, proactive_message_id, feedback_type,
                    confidence, pa_score, pua_score, lag_seconds, candidate_count,
                    matched_by, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    "2026-08-18T10:00:00+00:00",
                    "web:test",
                    f"u{row_id}",
                    f"a{row_id}",
                    f"p{row_id}",
                    feedback_type,
                    "high",
                    0.9,
                    0.8,
                    1,
                    1,
                    "test",
                    "test",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _workspace(tmp_path: Path) -> Path:
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    _write_drift_db(drift_dir / "drift.db")
    return tmp_path


def test_sample_distinguishes_missing_database(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    payload = module.sample(workspace / "drift", 50, 10, 0)

    assert payload["status"] == "db_missing"
    assert payload["found"] is False


def test_sample_distinguishes_empty_database(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    feedback_dir = workspace / "proactive_feedback"
    feedback_dir.mkdir()
    _write_feedback_db(feedback_dir / "proactive_feedback.db", [])

    payload = module.sample(workspace / "drift", 50, 10, 0)

    assert payload["status"] == "db_empty"
    assert payload["found"] is False


def test_sample_distinguishes_invalid_cursor(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    drift_db = workspace / "drift" / "drift.db"
    drift_db.unlink()
    _write_drift_db(drift_db, "not-json")

    payload = module.sample(workspace / "drift", 50, 10, 0)

    assert payload["status"] == "cursor_invalid"
    assert payload["reason"] == "drift_cursor_invalid"


def test_sample_returns_ready_batch_and_stable_chunk_keys(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    feedback_dir = workspace / "proactive_feedback"
    feedback_dir.mkdir()
    _write_feedback_db(
        feedback_dir / "proactive_feedback.db",
        [(1, "topic_follow"), (2, "explicit_quote"), (3, "topic_follow")],
    )

    first = module.sample(workspace / "drift", 50, 2, 0)
    replay = module.sample(workspace / "drift", 50, 2, 0)

    assert first["status"] == "ready"
    assert first["batch_key"] == "feedback#3-feedback#1"
    assert first["chunk_key"] == "feedback#3-feedback#2"
    assert first["has_more"] is True
    assert replay["batch_key"] == first["batch_key"]
    assert replay["chunk_key"] == first["chunk_key"]
    assert replay["chunk_feedback_ids"] == first["chunk_feedback_ids"]


def test_active_cursor_freezes_batch_when_new_feedback_arrives(tmp_path: Path) -> None:
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    _write_drift_db(
        drift_dir / "drift.db",
        json.dumps(
            {
                "active_feedback_ids": [3, 2, 1],
                "active_cursor_tail_feedback_id": 3,
                "active_chunk_index": 1,
            }
        ),
    )
    feedback_dir = tmp_path / "proactive_feedback"
    feedback_dir.mkdir()
    _write_feedback_db(
        feedback_dir / "proactive_feedback.db",
        [
            (1, "topic_follow"),
            (2, "explicit_quote"),
            (3, "topic_follow"),
            (4, "explicit_quote"),
        ],
    )

    payload = module.sample(drift_dir, 50, 2, 0)

    assert payload["status"] == "ready"
    assert payload["active_batch"] is True
    assert payload["batch_key"] == "feedback#3-feedback#1"
    assert payload["chunk_index"] == 1
    assert payload["chunk_feedback_ids"] == [1]
    assert 4 not in payload["feedback_ids"]


def test_active_cursor_past_batch_is_invalid(tmp_path: Path) -> None:
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    _write_drift_db(
        drift_dir / "drift.db",
        json.dumps(
            {
                "active_feedback_ids": [1],
                "active_cursor_tail_feedback_id": 1,
                "active_chunk_index": 1,
            }
        ),
    )
    feedback_dir = tmp_path / "proactive_feedback"
    feedback_dir.mkdir()
    _write_feedback_db(feedback_dir / "proactive_feedback.db", [(1, "topic_follow")])

    payload = module.sample(drift_dir, 50, 10, 0)

    assert payload["status"] == "cursor_invalid"
    assert payload["reason"] == "drift_cursor_invalid"


def test_sample_reports_invalid_schema(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    feedback_dir = workspace / "proactive_feedback"
    feedback_dir.mkdir()
    conn = sqlite3.connect(feedback_dir / "proactive_feedback.db")
    try:
        _ = conn.execute("CREATE TABLE proactive_feedback_events(id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    payload = module.sample(workspace / "drift", 50, 10, 0)

    assert payload["status"] == "schema_invalid"
    assert payload["found"] is False


def test_validate_pending_accepts_candidate_and_rejects_duplicate_chunk(
    tmp_path: Path,
) -> None:
    pending = tmp_path / "proactive_pending.md"
    pending.write_text(
        """# Proactive Pending

## Batch feedback#3-feedback#1

### Chunk feedback#3-feedback#2

- [ ] effect=verify confidence=low topic="AI Agent Runtime" granularity="仅限 Runtime，不扩大到所有 AI 新闻" inference="用户连续追问 Runtime 结构" action="后续候选先核验是否属于 Agent Runtime" evidence=feedback#3 user_message_id=u3
""",
        encoding="utf-8",
    )

    valid = module.validate_pending(pending)
    assert valid["valid"] is True
    assert valid["candidate_count"] == 1

    pending.write_text(
        pending.read_text(encoding="utf-8")
        + "\n### Chunk feedback#3-feedback#2\n"
        + "- [ ] no_candidate evidence=feedback#3 reason=duplicate\n",
        encoding="utf-8",
    )
    invalid = module.validate_pending(pending)
    assert invalid["valid"] is False
    assert any("duplicate chunk key" in item for item in invalid["errors"])


def test_crash_replay_keeps_pending_marker_idempotent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    feedback_dir = workspace / "proactive_feedback"
    feedback_dir.mkdir()
    _write_feedback_db(feedback_dir / "proactive_feedback.db", [(7, "topic_follow")])
    first = module.sample(workspace / "drift", 50, 10, 0)
    pending = workspace / "proactive_pending.md"
    pending.write_text(
        f"# Proactive Pending\n\n## Batch {first['batch_key']}\n\n"
        f"### Chunk {first['chunk_key']}\n\n"
        "- [ ] no_candidate evidence=feedback#7 reason=crash_replay\n",
        encoding="utf-8",
    )

    replay = module.sample(workspace / "drift", 50, 10, 0)
    validation = module.validate_pending(pending)

    assert replay["chunk_key"] == first["chunk_key"]
    assert validation["valid"] is True
    assert (
        pending.read_text(encoding="utf-8").count(f"### Chunk {first['chunk_key']}")
        == 1
    )
