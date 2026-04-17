"""
Conversation context budgeting for long-running LangGraph sessions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

from gustobot.application.services.llm_client import LLMClient
from gustobot.config import settings

try:  # pragma: no cover - optional dependency in some test environments
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - fallback estimator is used
    tiktoken = None

try:  # pragma: no cover - runtime dependency
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - lightweight fallbacks for unit tests
    @dataclass
    class BaseMessage:  # type: ignore[override]
        content: Any
        id: Optional[str] = None
        type: str = "base"
        additional_kwargs: Dict[str, Any] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.additional_kwargs is None:
                self.additional_kwargs = {}

    @dataclass
    class HumanMessage(BaseMessage):  # type: ignore[override]
        type: str = "human"

    @dataclass
    class SystemMessage(BaseMessage):  # type: ignore[override]
        type: str = "system"

try:  # pragma: no cover - runtime dependency
    from langgraph.graph.message import RemoveMessage
except ImportError:  # pragma: no cover - lightweight fallback for unit tests
    @dataclass
    class RemoveMessage:  # type: ignore[override]
        id: str


SUMMARY_PROMPT = """你是一个对话上下文压缩器，负责把长会话整理成供后续问答使用的稳定摘要。

请遵守以下规则：
1. 只保留后续回答真正需要的事实、约束、偏好、未完成任务和关键结论。
2. 保留用户口味偏好、禁忌、文件/图片上下文、系统已得出的中间结论。
3. 删除寒暄、重复表达、无关展开和已经失效的细节。
4. 输出使用简洁中文，优先项目符号。
5. 不要编造任何新信息，不要解释你正在总结。
6. 严格控制长度，目标不超过 {summary_token_limit} tokens。
"""


@dataclass
class BudgetApplicationResult:
    messages: List[Any]
    estimated_tokens_before: int
    estimated_tokens_after: int
    compressed: bool
    summary_updated: bool
    pruned_messages: int

    def as_metadata(self) -> Dict[str, Any]:
        return asdict(self)


class TokenEstimator:
    """Best-effort token estimator with a tiktoken fast path."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.LLM_MODEL
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except Exception:
                try:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoding = None

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass
        # Conservative fallback for mixed Chinese/English content.
        return max(1, math.ceil(len(text) / 3.2))

    def estimate_messages(self, messages: Sequence[Any]) -> int:
        total = 0
        for message in messages:
            content = self._message_content(message)
            total += self.estimate_text(content) + 8
        return total

    def _message_content(self, message: Any) -> str:
        return _stringify_content(getattr(message, "content", ""))


class ContextBudgetManager:
    """Applies a sliding window plus dynamic summary to long conversations."""

    SUMMARY_MESSAGE_TEMPLATE = "context-summary:{session_id}"

    def __init__(
        self,
        *,
        llm_client: Optional[LLMClient] = None,
        estimator: Optional[TokenEstimator] = None,
        max_context_tokens: Optional[int] = None,
        response_reserve_tokens: Optional[int] = None,
        system_reserve_tokens: Optional[int] = None,
        trigger_ratio: Optional[float] = None,
        keep_last_turns: Optional[int] = None,
        sliding_window_tokens: Optional[int] = None,
        summary_max_tokens: Optional[int] = None,
        summary_max_chars: Optional[int] = None,
    ) -> None:
        self._llm_client = llm_client if llm_client is not None else (
            LLMClient(temperature=0.2) if settings.OPENAI_API_KEY else None
        )
        self._estimator = estimator or TokenEstimator()
        self.trigger_ratio = (
            trigger_ratio if trigger_ratio is not None else settings.CONTEXT_TRIGGER_RATIO
        )
        self.max_context_tokens = (
            max_context_tokens if max_context_tokens is not None else settings.CONTEXT_MAX_TOKENS
        )
        self.response_reserve_tokens = (
            response_reserve_tokens
            if response_reserve_tokens is not None
            else settings.CONTEXT_RESPONSE_RESERVE_TOKENS
        )
        self.system_reserve_tokens = (
            system_reserve_tokens
            if system_reserve_tokens is not None
            else settings.CONTEXT_SYSTEM_RESERVE_TOKENS
        )
        self.keep_last_turns = (
            keep_last_turns if keep_last_turns is not None else settings.CONTEXT_KEEP_LAST_TURNS
        )
        self.sliding_window_tokens = (
            sliding_window_tokens
            if sliding_window_tokens is not None
            else settings.CONTEXT_SLIDING_WINDOW_TOKENS
        )
        self.summary_max_tokens = (
            summary_max_tokens
            if summary_max_tokens is not None
            else settings.CONTEXT_SUMMARY_MAX_TOKENS
        )
        self.summary_max_chars = (
            summary_max_chars
            if summary_max_chars is not None
            else settings.CONTEXT_SUMMARY_MAX_CHARS
        )

    @property
    def effective_history_budget(self) -> int:
        budget = (
            self.max_context_tokens
            - self.response_reserve_tokens
            - self.system_reserve_tokens
        )
        return max(budget, self.sliding_window_tokens + self.summary_max_tokens)

    async def prepare_input(
        self,
        graph: Any,
        *,
        session_id: str,
        user_message: str,
        config: Dict[str, Any],
    ) -> BudgetApplicationResult:
        new_message = HumanMessage(content=user_message)
        existing_messages = self._load_existing_messages(graph, config)
        if not existing_messages:
            initial_tokens = self._estimator.estimate_messages([new_message])
            return BudgetApplicationResult(
                messages=[new_message],
                estimated_tokens_before=initial_tokens,
                estimated_tokens_after=initial_tokens,
                compressed=False,
                summary_updated=False,
                pruned_messages=0,
            )

        summary_id = self.SUMMARY_MESSAGE_TEMPLATE.format(session_id=session_id)
        summary_message = next(
            (msg for msg in existing_messages if getattr(msg, "id", None) == summary_id),
            None,
        )
        transcript_messages = [
            msg for msg in existing_messages if getattr(msg, "id", None) != summary_id
        ]

        estimated_before = self._estimator.estimate_messages([*existing_messages, new_message])
        compression_limit = int(self.effective_history_budget * self.trigger_ratio)
        if estimated_before <= compression_limit:
            return BudgetApplicationResult(
                messages=[new_message],
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
                compressed=False,
                summary_updated=False,
                pruned_messages=0,
            )

        kept_messages = self._select_recent_window(transcript_messages, new_message)
        kept_ids = {
            getattr(message, "id", None)
            for message in kept_messages
            if getattr(message, "id", None)
        }
        pruned_messages = [
            message
            for message in transcript_messages
            if getattr(message, "id", None) and getattr(message, "id", None) not in kept_ids
        ]

        if not pruned_messages:
            logger.info(
                "Context budget exceeded but no removable messages found | session_id={}",
                session_id,
            )
            return BudgetApplicationResult(
                messages=[new_message],
                estimated_tokens_before=estimated_before,
                estimated_tokens_after=estimated_before,
                compressed=False,
                summary_updated=False,
                pruned_messages=0,
            )

        previous_summary = self._extract_summary_text(summary_message)
        merged_summary = await self._build_summary(previous_summary, pruned_messages)

        update_messages: List[Any] = [
            RemoveMessage(id=message.id)
            for message in pruned_messages
            if getattr(message, "id", None)
        ]
        summary_updated = False
        if merged_summary:
            update_messages.append(
                SystemMessage(
                    content=self._format_summary(merged_summary),
                    id=summary_id,
                    additional_kwargs={"context_summary": True},
                )
            )
            summary_updated = True
        update_messages.append(new_message)

        estimated_after = self._estimator.estimate_messages(
            [*( [SystemMessage(content=self._format_summary(merged_summary))] if merged_summary else []), *kept_messages, new_message]
        )

        logger.info(
            "Applied context compression | session_id={} before_tokens={} after_tokens={} pruned_messages={} summary_updated={}",
            session_id,
            estimated_before,
            estimated_after,
            len(pruned_messages),
            summary_updated,
        )
        return BudgetApplicationResult(
            messages=update_messages,
            estimated_tokens_before=estimated_before,
            estimated_tokens_after=estimated_after,
            compressed=True,
            summary_updated=summary_updated,
            pruned_messages=len(pruned_messages),
        )

    def _load_existing_messages(self, graph: Any, config: Dict[str, Any]) -> List[Any]:
        try:
            snapshot = graph.get_state(config)
        except Exception as exc:
            logger.debug("Failed to load graph state for budgeting: {}", exc)
            return []

        if snapshot is None:
            return []
        values = getattr(snapshot, "values", {}) or {}
        messages = values.get("messages", [])
        return list(messages) if messages else []

    def _select_recent_window(
        self,
        messages: Sequence[Any],
        pending_user_message: Any,
    ) -> List[Any]:
        selected: List[Any] = []
        running_tokens = self._estimator.estimate_messages([pending_user_message])
        user_turns = 0

        for message in reversed(messages):
            message_tokens = self._estimator.estimate_messages([message])
            message_type = getattr(message, "type", "")
            next_user_turns = user_turns + (1 if message_type == "human" else 0)

            exceeds_turn_budget = bool(selected) and next_user_turns > self.keep_last_turns
            exceeds_token_budget = bool(selected) and (
                running_tokens + message_tokens > self.sliding_window_tokens
            )
            if exceeds_turn_budget or exceeds_token_budget:
                break

            selected.append(message)
            running_tokens += message_tokens
            user_turns = next_user_turns

        selected.reverse()
        return selected

    async def _build_summary(
        self,
        previous_summary: str,
        pruned_messages: Sequence[Any],
    ) -> str:
        transcript = self._render_transcript(pruned_messages)
        if not transcript and not previous_summary:
            return ""

        if self._llm_client is None:
            return self._truncate_summary(self._fallback_summary(previous_summary, transcript))

        summary_payload = "\n\n".join(
            section
            for section in [
                f"已有摘要：\n{previous_summary}" if previous_summary else "",
                f"新增需压缩对话：\n{transcript}" if transcript else "",
            ]
            if section
        )

        try:
            response = await self._llm_client.chat(
                system_prompt=SUMMARY_PROMPT.format(
                    summary_token_limit=self.summary_max_tokens,
                ),
                user_message=summary_payload or "请输出空摘要",
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Failed to generate rolling summary with LLM: {}", exc)
            response = self._fallback_summary(previous_summary, transcript)

        return self._truncate_summary(response)

    def _fallback_summary(self, previous_summary: str, transcript: str) -> str:
        sections: List[str] = []
        if previous_summary:
            sections.append(previous_summary.strip())
        if transcript:
            compressed_lines = [
                line.strip()
                for line in transcript.splitlines()
                if line.strip()
            ]
            if compressed_lines:
                sections.append("近期压缩记录：")
                sections.extend(f"- {line[:160]}" for line in compressed_lines[-12:])
        return "\n".join(sections).strip()

    def _truncate_summary(self, summary: str) -> str:
        if not summary:
            return ""

        trimmed = summary.strip()
        if len(trimmed) > self.summary_max_chars:
            trimmed = trimmed[: self.summary_max_chars].rstrip() + "..."

        estimated_tokens = self._estimator.estimate_text(trimmed)
        if estimated_tokens <= self.summary_max_tokens:
            return trimmed

        ratio = self.summary_max_tokens / max(estimated_tokens, 1)
        target_chars = max(256, int(len(trimmed) * ratio))
        return trimmed[:target_chars].rstrip() + "..."

    def _render_transcript(self, messages: Sequence[Any]) -> str:
        rendered: List[str] = []
        for message in messages:
            role = self._role_name(getattr(message, "type", ""))
            content = _stringify_content(getattr(message, "content", ""))
            if not content:
                continue
            rendered.append(f"{role}: {content}")
        return "\n".join(rendered)

    def _extract_summary_text(self, message: Optional[Any]) -> str:
        if message is None:
            return ""
        content = _stringify_content(getattr(message, "content", ""))
        prefix = "会话摘要（供系统参考）:\n"
        if content.startswith(prefix):
            return content[len(prefix):].strip()
        return content.strip()

    def _format_summary(self, summary: str) -> str:
        return f"会话摘要（供系统参考）:\n{summary.strip()}"

    @staticmethod
    def _role_name(message_type: str) -> str:
        if message_type == "human":
            return "用户"
        if message_type == "ai":
            return "助手"
        return "系统"


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        items: List[str] = []
        for item in content:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    items.append(str(text))
            else:
                items.append(str(item))
        return "".join(items)
    return str(content)


@lru_cache(maxsize=1)
def get_context_budget_manager() -> ContextBudgetManager:
    return ContextBudgetManager()
