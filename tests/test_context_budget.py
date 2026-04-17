import asyncio
from types import SimpleNamespace

from gustobot.application.services.context_budget import (
    ContextBudgetManager,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)


class FakeEstimator:
    def estimate_text(self, text: str) -> int:
        return len(text)

    def estimate_messages(self, messages):
        total = 0
        for message in messages:
            total += self.estimate_text(str(getattr(message, "content", ""))) + 1
        return total


class FakeLLMClient:
    async def chat(self, *, system_prompt, user_message, context=None, temperature=None):
        return "偏好：清淡\n未完成：继续推荐低脂菜谱"


class FakeGraph:
    def __init__(self, messages):
        self._messages = list(messages)

    def get_state(self, config):
        return SimpleNamespace(values={"messages": list(self._messages)})


def _message(message_cls, content: str, message_id: str):
    return message_cls(content=content, id=message_id)


def test_context_budget_compresses_old_messages():
    messages = [
        _message(HumanMessage, "我最近在减脂", "u1"),
        _message(SystemMessage, "你可以优先吃蒸煮类菜", "a1"),
        _message(HumanMessage, "不要太辣", "u2"),
        _message(SystemMessage, "收到，后续推荐会避开重辣", "a2"),
        _message(HumanMessage, "家里还有鸡胸肉", "u3"),
        _message(SystemMessage, "可以做香煎鸡胸肉沙拉", "a3"),
    ]
    manager = ContextBudgetManager(
        llm_client=FakeLLMClient(),
        estimator=FakeEstimator(),
        max_context_tokens=80,
        response_reserve_tokens=20,
        system_reserve_tokens=20,
        trigger_ratio=0.5,
        keep_last_turns=1,
        sliding_window_tokens=20,
        summary_max_tokens=30,
        summary_max_chars=120,
    )

    result = asyncio.run(
        manager.prepare_input(
            FakeGraph(messages),
            session_id="session-1",
            user_message="再给我推荐一道晚餐",
            config={"configurable": {"thread_id": "session-1"}},
        )
    )

    assert result.compressed is True
    assert result.summary_updated is True
    assert result.pruned_messages >= 2
    assert any(isinstance(message, RemoveMessage) for message in result.messages)
    summary_message = next(
        message for message in result.messages if isinstance(message, SystemMessage)
    )
    assert summary_message.id == "context-summary:session-1"
    assert "偏好" in summary_message.content
    assert result.messages[-1].content == "再给我推荐一道晚餐"


def test_context_budget_skips_compression_under_limit():
    messages = [
        _message(HumanMessage, "你好", "u1"),
        _message(SystemMessage, "你好呀", "a1"),
    ]
    manager = ContextBudgetManager(
        llm_client=FakeLLMClient(),
        estimator=FakeEstimator(),
        max_context_tokens=400,
        response_reserve_tokens=40,
        system_reserve_tokens=40,
        trigger_ratio=0.95,
        keep_last_turns=2,
        sliding_window_tokens=100,
        summary_max_tokens=30,
        summary_max_chars=120,
    )

    result = asyncio.run(
        manager.prepare_input(
            FakeGraph(messages),
            session_id="session-2",
            user_message="继续聊聊家常菜",
            config={"configurable": {"thread_id": "session-2"}},
        )
    )

    assert result.compressed is False
    assert result.summary_updated is False
    assert result.pruned_messages == 0
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], HumanMessage)
