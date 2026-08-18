from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

SKILL_NAME = "feedback-preference-context"
FEEDBACK_TABLE = "proactive_feedback_events"
PROACTIVE_TEXT_LIMIT = 100
QUESTION_MARKERS = ("吗", "么", "为什么", "怎么", "谁", "哪")
_FEEDBACK_COLUMNS = frozenset(
    {
        "id",
        "created_at",
        "session_key",
        "user_message_id",
        "assistant_message_id",
        "proactive_message_id",
        "feedback_type",
        "confidence",
        "pa_score",
        "pua_score",
        "lag_seconds",
        "candidate_count",
        "matched_by",
        "reason",
    }
)
_EFFECTS = frozenset({"block", "boost", "verify", "timing", "tone"})
_CONFIDENCES = frozenset({"low", "medium", "high"})
_FEEDBACK_ID = re.compile(r"feedback#\d+")
_RANGE_KEY = re.compile(r"^feedback#\d+-feedback#\d+$")
_PENDING_ITEM = re.compile(r"^- \[([ xX])\] (.+)$")
_PENDING_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\"(?:[^\"\\]|\\.)*\"|[^\s]+)")


def _valid_evidence(value: str) -> bool:
    parts = [item for item in value.split(",") if item]
    return bool(parts) and all(
        _FEEDBACK_ID.fullmatch(item) or _RANGE_KEY.fullmatch(item) for item in parts
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_feedback_schema(conn: sqlite3.Connection) -> None:
    """拒绝把缺列或错误数据库误判为空反馈源。"""

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (FEEDBACK_TABLE,),
    ).fetchone()
    if row is None:
        raise ValueError(f"反馈库缺少表: {FEEDBACK_TABLE}")
    columns = {
        str(item[1])
        for item in conn.execute(f"PRAGMA table_info({FEEDBACK_TABLE})").fetchall()
    }
    missing = sorted(_FEEDBACK_COLUMNS - columns)
    if missing:
        raise ValueError("反馈库缺少列: " + ", ".join(missing))


def _load_cursor(drift_dir: Path) -> dict[str, Any]:
    db_path = drift_dir / "drift.db"
    if not db_path.exists():
        return {}
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT cursor_json
            FROM skill_continuum
            WHERE skill_name = ?
            """,
            (SKILL_NAME,),
        ).fetchone()
    if row is None:
        return {}
    try:
        data = json.loads(str(row["cursor_json"] or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("drift cursor JSON is invalid") from exc
    if not isinstance(data, dict):
        raise ValueError("drift cursor must be a JSON object")
    return cast(dict[str, Any], data)


def _active_feedback_ids(cursor: dict[str, Any]) -> list[int]:
    raw = cursor.get("active_feedback_ids")
    if raw in (None, ""):
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("drift cursor active_feedback_ids must be a non-empty list")
    ids: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "drift cursor active_feedback_ids must contain positive integers"
            )
        ids.append(int(value))
    if len(ids) > 50 or len(set(ids)) != len(ids):
        raise ValueError(
            "drift cursor active_feedback_ids must contain 1-50 unique IDs"
        )
    return ids


def _message_previews(workspace: Path, ids: list[str]) -> dict[str, str]:
    db_path = workspace / "sessions.db"
    if not ids or not db_path.exists():
        return {}
    unique_ids = list(dict.fromkeys(text for text in ids if text))
    placeholders = ",".join("?" for _ in unique_ids)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, content
            FROM messages
            WHERE id IN ({placeholders})
            """,
            unique_ids,
        ).fetchall()
    return {str(row["id"]): str(row["content"] or "") for row in rows}


def _clip_text(text: str, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _range_key(ids: list[int]) -> str:
    if not ids:
        return ""
    return f"feedback#{max(ids)}-feedback#{min(ids)}"


def _empty_result(
    *,
    status: str,
    reason: str,
    last_feedback_id: int,
    chunk_size: int,
    chunk_index: int,
    active_batch: bool = False,
) -> dict[str, Any]:
    return {
        "found": False,
        "status": status,
        "reason": reason,
        "last_feedback_id": last_feedback_id,
        "latest_processed_feedback_id": last_feedback_id,
        "cursor_tail_feedback_id": last_feedback_id,
        "count": 0,
        "chunk_index": max(0, int(chunk_index)),
        "chunk_size": max(1, min(int(chunk_size), 10)),
        "has_more": False,
        "next_chunk_index": None,
        "batch_key": None,
        "chunk_key": None,
        "active_batch": active_batch,
    }


def _cursor_invalid_result(
    *,
    error: str,
    last_feedback_id: int = 0,
    chunk_size: int = 10,
    chunk_index: int = 0,
) -> dict[str, Any]:
    """返回需要人工维护 cursor 的暂停结果。"""

    return {
        "found": False,
        "status": "cursor_invalid",
        "reason": "drift_cursor_invalid",
        "error": error,
        "last_feedback_id": last_feedback_id,
        "latest_processed_feedback_id": last_feedback_id,
        "cursor_tail_feedback_id": last_feedback_id,
        "count": 0,
        "chunk_index": max(0, int(chunk_index)),
        "chunk_size": max(1, min(int(chunk_size), 10)),
        "has_more": False,
        "next_chunk_index": None,
        "batch_key": None,
        "chunk_key": None,
        "active_batch": True,
    }


def _signal_hints(user: str, feedback_type: str) -> list[str]:
    text = _clean_text(user)
    hints: list[str] = []
    if feedback_type == "explicit_quote":
        hints.append("explicit_quote")
    if any(marker in text for marker in QUESTION_MARKERS):
        hints.append("question")
    if len(text) >= 8:
        hints.append("substantive_reply")
    return hints


def sample(
    drift_dir: Path,
    limit: int,
    chunk_size: int,
    chunk_index: int,
) -> dict[str, Any]:
    workspace = drift_dir.parent
    feedback_db = workspace / "proactive_feedback" / "proactive_feedback.db"
    try:
        cursor = _load_cursor(drift_dir)
    except ValueError as exc:
        return _cursor_invalid_result(error=str(exc), chunk_size=chunk_size)
    raw_last_feedback_id = cursor.get(
        "latest_processed_feedback_id",
        cursor.get("last_feedback_id", 0),
    )
    if (
        isinstance(raw_last_feedback_id, bool)
        or not isinstance(raw_last_feedback_id, int)
        or raw_last_feedback_id < 0
    ):
        return _cursor_invalid_result(
            error="latest_processed_feedback_id must be a non-negative integer",
            chunk_size=chunk_size,
        )
    last_feedback_id = int(raw_last_feedback_id)
    try:
        active_ids = _active_feedback_ids(cursor)
    except ValueError as exc:
        return _cursor_invalid_result(
            error=str(exc),
            last_feedback_id=last_feedback_id,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )
    raw_active_tail = cursor.get("active_cursor_tail_feedback_id", 0)
    if (
        isinstance(raw_active_tail, bool)
        or not isinstance(raw_active_tail, int)
        or raw_active_tail < 0
    ):
        return _cursor_invalid_result(
            error="active_cursor_tail_feedback_id must be a non-negative integer",
            last_feedback_id=last_feedback_id,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )
    active_tail = int(raw_active_tail)
    if active_ids and active_tail != max(active_ids):
        return {
            "found": False,
            "status": "cursor_invalid",
            "reason": "drift_cursor_invalid",
            "error": "active_cursor_tail_feedback_id does not match active_feedback_ids",
            "last_feedback_id": last_feedback_id,
            "latest_processed_feedback_id": last_feedback_id,
            "cursor_tail_feedback_id": active_tail,
            "count": 0,
            "chunk_index": max(0, int(chunk_index)),
            "chunk_size": max(1, min(int(chunk_size), 10)),
            "has_more": False,
            "next_chunk_index": None,
            "batch_key": _range_key(active_ids),
            "chunk_key": None,
            "active_batch": True,
        }
    if not feedback_db.exists():
        return _empty_result(
            status="cursor_invalid" if active_ids else "db_missing",
            reason="feedback_db_missing",
            last_feedback_id=last_feedback_id,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            active_batch=bool(active_ids),
        )
    safe_limit = max(1, min(int(limit), 50))
    try:
        with _connect(feedback_db) as conn:
            _validate_feedback_schema(conn)
            total_rows = int(
                conn.execute(f"SELECT count(*) FROM {FEEDBACK_TABLE}").fetchone()[0]
            )
            if total_rows == 0:
                return _empty_result(
                    status="cursor_invalid" if active_ids else "db_empty",
                    reason="feedback_db_empty",
                    last_feedback_id=last_feedback_id,
                    chunk_size=chunk_size,
                    chunk_index=chunk_index,
                    active_batch=bool(active_ids),
                )
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                rows = conn.execute(
                    f"""
                    SELECT
                        id,
                        created_at,
                        session_key,
                        user_message_id,
                        proactive_message_id,
                        feedback_type,
                        confidence,
                        pa_score,
                        pua_score,
                        lag_seconds,
                        candidate_count,
                        matched_by,
                        reason
                    FROM {FEEDBACK_TABLE}
                    WHERE id IN ({placeholders})
                      AND feedback_type IN ('topic_follow', 'explicit_quote')
                    ORDER BY id DESC
                    """,
                    tuple(active_ids),
                ).fetchall()
                if len(rows) != len(active_ids):
                    found_ids = {int(row["id"]) for row in rows}
                    missing_ids = sorted(set(active_ids) - found_ids)
                    raise ValueError(
                        "active feedback IDs missing: "
                        + ",".join(str(value) for value in missing_ids)
                    )
            else:
                rows = conn.execute(
                    f"""
                    SELECT
                        id,
                        created_at,
                        session_key,
                        proactive_message_id,
                        user_message_id,
                        feedback_type,
                        confidence,
                        pa_score,
                        pua_score,
                        lag_seconds,
                        candidate_count,
                        matched_by,
                        reason
                    FROM {FEEDBACK_TABLE}
                    WHERE id > ?
                      AND feedback_type IN ('topic_follow', 'explicit_quote')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (last_feedback_id, safe_limit),
                ).fetchall()
    except (sqlite3.DatabaseError, ValueError) as exc:
        cursor_error = isinstance(exc, ValueError) and str(exc).startswith(
            "active feedback IDs missing:"
        )
        return {
            "found": False,
            "status": "cursor_invalid" if cursor_error else "schema_invalid",
            "reason": (
                "drift_cursor_invalid" if cursor_error else "feedback_db_schema_invalid"
            ),
            "error": str(exc),
            "last_feedback_id": last_feedback_id,
            "latest_processed_feedback_id": last_feedback_id,
            "cursor_tail_feedback_id": last_feedback_id,
            "count": 0,
            "chunk_index": max(0, int(chunk_index)),
            "chunk_size": max(1, min(int(chunk_size), 10)),
            "has_more": False,
            "next_chunk_index": None,
            "batch_key": None,
            "chunk_key": None,
            "active_batch": bool(active_ids),
        }

    if not rows:
        return _empty_result(
            status="cursor_invalid" if active_ids else "no_eligible_rows",
            reason=(
                "active_feedback_rows_not_eligible"
                if active_ids
                else "no_unprocessed_eligible_feedback"
            ),
            last_feedback_id=last_feedback_id,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            active_batch=bool(active_ids),
        )

    safe_chunk_size = max(1, min(int(chunk_size), 10))
    raw_active_index = cursor.get("active_chunk_index", chunk_index)
    if active_ids and (
        isinstance(raw_active_index, bool)
        or not isinstance(raw_active_index, int)
        or raw_active_index < 0
    ):
        return _cursor_invalid_result(
            error="active_chunk_index must be a non-negative integer",
            last_feedback_id=last_feedback_id,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
        )
    safe_chunk_index = max(0, int(raw_active_index))
    chunk_start = safe_chunk_index * safe_chunk_size
    chunk_end = chunk_start + safe_chunk_size
    chunk_rows = rows[chunk_start:chunk_end]
    if not chunk_rows:
        return (
            _cursor_invalid_result(
                error="active_chunk_index points past the frozen feedback batch",
                last_feedback_id=last_feedback_id,
                chunk_size=chunk_size,
                chunk_index=safe_chunk_index,
            )
            if active_ids
            else _empty_result(
                status="no_eligible_rows",
                reason="chunk_index_out_of_range",
                last_feedback_id=last_feedback_id,
                chunk_size=chunk_size,
                chunk_index=safe_chunk_index,
            )
        )
    feedback_ids = [int(row["id"]) for row in rows]
    chunk_feedback_ids = [int(row["id"]) for row in chunk_rows]
    has_more = chunk_end < len(rows)

    message_ids: list[str] = []
    for row in chunk_rows:
        message_ids.extend(
            str(row[key] or "")
            for key in (
                "proactive_message_id",
                "user_message_id",
            )
            if row[key]
        )
    previews = _message_previews(workspace, message_ids)
    events: list[dict[str, Any]] = []
    for row in chunk_rows:
        proactive_id = str(row["proactive_message_id"] or "")
        user_id = str(row["user_message_id"] or "")
        user_text = _clean_text(previews.get(user_id, ""))
        events.append(
            {
                "id": int(row["id"]),
                "created_at": str(row["created_at"] or ""),
                "session_key": str(row["session_key"] or ""),
                "feedback_type": str(row["feedback_type"] or ""),
                "confidence": str(row["confidence"] or ""),
                "pa_score": row["pa_score"],
                "pua_score": row["pua_score"],
                "lag_seconds": row["lag_seconds"],
                "candidate_count": int(row["candidate_count"] or 0),
                "matched_by": str(row["matched_by"] or ""),
                "reason": str(row["reason"] or ""),
                "message_ids": {
                    "proactive": proactive_id,
                    "user": user_id,
                },
                "signal_hints": _signal_hints(
                    user_text,
                    str(row["feedback_type"] or ""),
                ),
                "texts": {
                    "proactive": _clip_text(
                        previews.get(proactive_id, ""),
                        PROACTIVE_TEXT_LIMIT,
                    ),
                    "user": user_text,
                },
            }
        )

    cursor_tail = active_tail or max(int(row["id"]) for row in rows)
    batch_feedback_ids = active_ids or feedback_ids
    return {
        "found": True,
        "status": "ready",
        "last_feedback_id": last_feedback_id,
        "latest_processed_feedback_id": last_feedback_id,
        "count": len(rows),
        "cursor_tail_feedback_id": cursor_tail,
        "feedback_ids": batch_feedback_ids,
        "batch_key": _range_key(batch_feedback_ids),
        "chunk_index": safe_chunk_index,
        "chunk_size": safe_chunk_size,
        "chunk_count": len(events),
        "chunk_feedback_ids": chunk_feedback_ids,
        "chunk_key": _range_key(chunk_feedback_ids),
        "active_batch": bool(active_ids),
        "active_feedback_ids": batch_feedback_ids,
        "active_cursor_tail_feedback_id": cursor_tail,
        "has_more": has_more,
        "next_chunk_index": safe_chunk_index + 1 if has_more else None,
        "text_limits": {
            "proactive": PROACTIVE_TEXT_LIMIT,
            "user": None,
        },
        "events": events,
    }


def evidence_bundle(
    drift_dir: Path,
    limit: int,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    chunk_index = 0
    while True:
        payload = sample(drift_dir, limit, 10, chunk_index)
        chunks.append(payload)
        if not payload.get("found"):
            return payload
        events.extend(payload.get("events", []))
        if payload.get("active_batch"):
            break
        if not payload.get("has_more"):
            break
        chunk_index = int(payload.get("next_chunk_index") or chunk_index + 1)

    first = chunks[0]
    compact_events: list[dict[str, Any]] = []
    for event in events:
        texts = event.get("texts", {})
        message_ids = event.get("message_ids", {})
        compact_events.append(
            {
                "fid": int(event["id"]),
                "type": str(event.get("feedback_type") or ""),
                "conf": str(event.get("confidence") or ""),
                "uid": str(message_ids.get("user") or ""),
                "pid": str(message_ids.get("proactive") or ""),
                "hints": list(event.get("signal_hints") or []),
                "p": str(texts.get("proactive") or ""),
                "u": str(texts.get("user") or ""),
            }
        )

    return {
        "found": True,
        "status": "ready",
        "last_feedback_id": first["last_feedback_id"],
        "count": first["count"],
        "cursor_tail_feedback_id": first["cursor_tail_feedback_id"],
        "feedback_ids": first["feedback_ids"],
        "batch_key": first["batch_key"],
        "events": compact_events,
    }


def _parse_pending_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _PENDING_FIELD.finditer(raw):
        key = str(match.group(1))
        value = str(match.group(2))
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        fields[key] = value
    return fields


def validate_pending(path: Path) -> dict[str, Any]:
    """校验待审核队列的候选字段、effect 和稳定 chunk 标记。"""

    if not path.exists():
        return {
            "valid": True,
            "status": "empty",
            "path": str(path),
            "candidate_count": 0,
            "chunk_keys": [],
            "errors": [],
        }
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {
            "valid": True,
            "status": "empty",
            "path": str(path),
            "candidate_count": 0,
            "chunk_keys": [],
            "errors": [],
        }

    errors: list[str] = []
    chunk_keys: list[str] = []
    current_batch = ""
    current_chunk = ""
    candidate_count = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.startswith("## Batch "):
            current_batch = line.removeprefix("## Batch ").strip()
            if not _RANGE_KEY.fullmatch(current_batch):
                errors.append(f"line {line_number}: invalid batch key")
            current_chunk = ""
            continue
        if line.startswith("### Chunk "):
            current_chunk = line.removeprefix("### Chunk ").strip()
            if not _RANGE_KEY.fullmatch(current_chunk):
                errors.append(f"line {line_number}: invalid chunk key")
            elif current_chunk in chunk_keys:
                errors.append(
                    f"line {line_number}: duplicate chunk key {current_chunk}"
                )
            else:
                chunk_keys.append(current_chunk)
            if not current_batch:
                errors.append(f"line {line_number}: chunk without batch")
            continue
        match = _PENDING_ITEM.fullmatch(line)
        if match is None:
            if line.startswith("- ["):
                errors.append(f"line {line_number}: malformed pending item")
            continue
        if not current_batch or not current_chunk:
            errors.append(f"line {line_number}: item without batch/chunk")
            continue
        fields = _parse_pending_fields(match.group(2))
        if "no_candidate" in match.group(2):
            if not fields.get("evidence") or not fields.get("reason"):
                errors.append(
                    f"line {line_number}: no_candidate requires evidence and reason"
                )
            evidence = fields.get("evidence", "")
            if not _valid_evidence(evidence):
                errors.append(f"line {line_number}: invalid no_candidate evidence")
            continue
        candidate_count += 1
        required = {
            "effect",
            "confidence",
            "topic",
            "granularity",
            "inference",
            "action",
            "evidence",
            "user_message_id",
        }
        missing = sorted(key for key in required if not fields.get(key))
        if missing:
            errors.append(f"line {line_number}: missing fields {','.join(missing)}")
        if fields.get("effect") not in _EFFECTS:
            errors.append(f"line {line_number}: invalid effect")
        if fields.get("confidence") not in _CONFIDENCES:
            errors.append(f"line {line_number}: invalid confidence")
        evidence = fields.get("evidence", "")
        if not _valid_evidence(evidence):
            errors.append(f"line {line_number}: invalid evidence")
        if "/" in fields.get("topic", ""):
            errors.append(f"line {line_number}: topic must not combine objects")

    return {
        "valid": not errors,
        "status": "ready" if not errors else "invalid",
        "path": str(path),
        "candidate_count": candidate_count,
        "chunk_keys": chunk_keys,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sample_cmd = sub.add_parser("sample")
    _ = sample_cmd.add_argument("--drift-dir", default=".")
    _ = sample_cmd.add_argument("--limit", type=int, default=50)
    _ = sample_cmd.add_argument("--chunk-size", type=int, default=10)
    _ = sample_cmd.add_argument("--chunk-index", type=int, default=0)
    validate_cmd = sub.add_parser("validate")
    _ = validate_cmd.add_argument("--drift-dir", default=".")
    _ = validate_cmd.add_argument("--pending-path", default="")
    bundle_cmd = sub.add_parser("evidence-bundle")
    _ = bundle_cmd.add_argument("--drift-dir", default=".")
    _ = bundle_cmd.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if args.command == "sample":
        payload = sample(
            Path(args.drift_dir).expanduser().resolve(),
            args.limit,
            args.chunk_size,
            args.chunk_index,
        )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if payload.get("status") == "schema_invalid":
            raise SystemExit(2)
    elif args.command == "evidence-bundle":
        payload = evidence_bundle(
            Path(args.drift_dir).expanduser().resolve(),
            args.limit,
        )
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if payload.get("status") == "schema_invalid":
            raise SystemExit(2)
    elif args.command == "validate":
        drift_dir = Path(args.drift_dir).expanduser().resolve()
        pending_path = (
            Path(args.pending_path).expanduser().resolve()
            if args.pending_path
            else drift_dir.parent / "proactive_pending.md"
        )
        payload = validate_pending(pending_path)
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if not bool(payload.get("valid")):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
