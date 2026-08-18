from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.plugins import (
    MobileUiContribution,
    MobileUiNavigation,
    Plugin,
    tool,
)
from agent.plugins.mobile_ui import MobileUiRpcInvalidRequest
from bus.events_proactive import ProactiveFeedbackRecorded
from proactive_v2.frame import ProactiveFrame

from .db import apply_feedback, build_effect, get_state, open_db
from .dashboard import EmotionDashboardReader

logger = logging.getLogger("plugin.emotion")


class EmotionProactivePromptModule:
    slot = "proactive.prompt.emotion"
    produces = (
        "proactive:prompt:system_bottom:emotion",
        "proactive:effect:emotion",
    )

    def __init__(self, plugin: "EmotionPlugin") -> None:
        self._plugin = plugin

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        effect = self._plugin.build_proactive_prompt_effect(frame)
        if effect is None:
            return frame
        frame.slots["proactive:prompt:system_bottom:emotion"] = str(
            effect.get("prompt_section") or ""
        )
        frame.slots["proactive:effect:emotion"] = effect
        return frame


class EmotionPlugin(Plugin):
    api_version = 2

    @classmethod
    def dashboard_module(cls) -> str | None:
        return "dashboard.py"

    @classmethod
    def mobile_ui(cls) -> MobileUiContribution:
        return MobileUiContribution(
            module="mobile_panel.js",
            stylesheet="mobile_panel.css",
            navigation=MobileUiNavigation(
                label="主动状态",
                description="反馈如何改变 Agent 的语气和主动发送把握",
            ),
        )

    name = "emotion"
    version = "1.1.0"

    @classmethod
    def drift_skill_roots(cls) -> tuple[str, ...]:
        return ("drift/skills",)

    def activate(self) -> None:
        workspace = self.context.workspace
        if workspace is None:
            logger.warning("emotion 插件缺少 workspace，跳过加载")
            return
        self._db_path = workspace / "emotion" / "emotion.db"
        conn = open_db(self._db_path)
        conn.close()
        self.context.event_bus.on(ProactiveFeedbackRecorded, self._on_feedback_recorded)

    async def terminate(self) -> None:
        return None

    def mobile_ui_query(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """返回反馈如何调节主动状态的移动投影。"""

        # 1. 在插件 RPC 边界校验方法与列表上限
        _ = session_id, turn_id
        if method != "emotion.bootstrap":
            raise MobileUiRpcInvalidRequest(f"未知 emotion 移动方法: {method}")
        workspace = self.context.workspace
        if workspace is None:
            raise RuntimeError("emotion 移动看板缺少 workspace")
        limit = payload.get("limit", 30)
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 50
        ):
            raise MobileUiRpcInvalidRequest("limit 必须是 1 到 50 的整数")

        # 2. 单个 SQLite 快照返回首屏全部区域
        return EmotionDashboardReader(workspace).get_mobile_bootstrap(limit=limit)

    def proactive_modules(self) -> list[object]:
        return [EmotionProactivePromptModule(self)]

    def build_proactive_prompt_effect(
        self,
        frame: ProactiveFrame,
    ) -> dict[str, Any] | None:
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return None
        conn = open_db(Path(db_path))
        try:
            return build_effect(
                conn,
                tick_id=f"frame:{frame.input.started_at.isoformat()}",
                session_key=str(
                    frame.slots.get("proactive:session_key") or frame.input.session_key
                ),
                now_utc=frame.input.started_at,
                last_user_at=frame.slots.get("proactive:last_user_at"),
                base_threshold=float(
                    frame.slots.get("proactive:base_judge_send_threshold") or 0.60
                ),
            )
        finally:
            conn.close()

    def _on_feedback_recorded(self, event: ProactiveFeedbackRecorded) -> None:
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return
        payload: dict[str, Any] = {
            "feedback_event_id": event.event_id,
            "user_message_id": event.user_message_id,
            "assistant_message_id": event.assistant_message_id,
            "proactive_message_id": event.proactive_message_id,
            "feedback_type": event.feedback_type,
            "confidence": event.confidence,
            "pua_score": event.pua_score,
            "lag_seconds": event.lag_seconds,
            "matched_by": event.matched_by,
        }
        conn = open_db(Path(db_path))
        try:
            _ = apply_feedback(
                conn,
                source_event_id=f"proactive_feedback:{event.event_id}",
                session_key=event.session_key,
                feedback_type=event.feedback_type,
                confidence=event.confidence,
                payload=payload,
            )
        finally:
            conn.close()

    @tool(
        "get_emotion_state",
        risk="read-only",
        search_hint="查询 proactive VAD 情绪状态",
    )
    async def get_emotion_state(self, event: Any) -> dict[str, Any]:
        """查询 proactive VAD 情绪状态。"""
        _ = event
        db_path = getattr(self, "_db_path", None)
        if db_path is None:
            return {"available": False}
        conn = open_db(Path(db_path))
        try:
            state = get_state(conn)
        finally:
            conn.close()
        return {
            "available": True,
            "valence": state.valence,
            "arousal": state.arousal,
            "dominance": state.dominance,
            "updated_at": state.updated_at,
        }
