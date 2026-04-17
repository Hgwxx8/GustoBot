"""
Redis Stack-backed LangGraph checkpoint persistence with in-memory fallback.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger
from redis import Redis as SyncRedis
from redis.asyncio import Redis as AsyncRedis

from gustobot.config import settings

try:  # pragma: no cover - runtime dependency
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
except ImportError:  # pragma: no cover - lightweight fallbacks for unit tests
    class BaseCheckpointSaver:  # type: ignore[override]
        pass

    @dataclass
    class CheckpointTuple:  # type: ignore[override]
        config: Dict[str, Any]
        checkpoint: Dict[str, Any]
        metadata: Dict[str, Any]
        parent_config: Optional[Dict[str, Any]]
        pending_writes: List[Any]

    class JsonPlusSerializer:  # type: ignore[override]
        def dumps_typed(self, value: Any) -> Sequence[Any]:
            return (type(value).__name__, json.dumps(value, default=str).encode("utf-8"))

        def loads_typed(self, payload: Sequence[Any]) -> Any:
            _, raw = payload
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)

    class MemorySaver:  # type: ignore[override]
        def __init__(self) -> None:
            self._store: Dict[str, Any] = {}


def _utc_now_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def _decode_response(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class RedisStackCheckpointSaver(BaseCheckpointSaver):
    """Persist LangGraph checkpoints via RedisJSON while keeping a memory fallback."""

    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        prefix: Optional[str] = None,
        ttl: Optional[int] = None,
        sync_redis: Optional[SyncRedis] = None,
        async_redis: Optional[AsyncRedis] = None,
        memory_fallback: Optional[MemorySaver] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix or settings.REDIS_CHECKPOINT_PREFIX
        self.ttl = ttl if ttl is not None else settings.REDIS_CHECKPOINT_TTL
        resolved_url = redis_url or settings.REDIS_URL
        self._redis = sync_redis or SyncRedis.from_url(resolved_url, decode_responses=True)
        self._aredis = async_redis or AsyncRedis.from_url(resolved_url, decode_responses=True)
        self._memory = memory_fallback or MemorySaver()
        self.serde = JsonPlusSerializer()

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        try:
            return self._load_checkpoint_tuple(config, redis_client=self._redis)
        except Exception as exc:
            logger.warning("Redis checkpoint get_tuple failed; falling back to memory saver: {}", exc)
            return self._delegate("get_tuple", config)

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[CheckpointTuple]:
        try:
            for item in self._list_checkpoints(
                config,
                before=before,
                limit=limit,
                redis_client=self._redis,
            ):
                yield item
        except Exception as exc:
            logger.warning("Redis checkpoint list failed; falling back to memory saver: {}", exc)
            fallback = self._delegate("list", config, before=before, limit=limit) or []
            for item in fallback:
                yield item

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Any,
    ) -> Dict[str, Any]:
        try:
            return self._save_checkpoint(
                config,
                checkpoint,
                metadata,
                new_versions=new_versions,
                redis_client=self._redis,
            )
        except Exception as exc:
            logger.warning("Redis checkpoint put failed; falling back to memory saver: {}", exc)
            return self._delegate("put", config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Sequence[Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        try:
            self._append_writes(
                config,
                writes=writes,
                task_id=task_id,
                task_path=task_path,
                redis_client=self._redis,
            )
            return None
        except Exception as exc:
            logger.warning("Redis checkpoint put_writes failed; falling back to memory saver: {}", exc)
            self._delegate("put_writes", config, writes, task_id, task_path)
            return None

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        try:
            return await self._aload_checkpoint_tuple(config, redis_client=self._aredis)
        except Exception as exc:
            logger.warning("Redis checkpoint aget_tuple failed; falling back to memory saver: {}", exc)
            return await self._adelegate("aget_tuple", config)

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[CheckpointTuple]:
        try:
            items = await self._alist_checkpoints(
                config,
                before=before,
                limit=limit,
                redis_client=self._aredis,
            )
            for item in items:
                yield item
        except Exception as exc:
            logger.warning("Redis checkpoint alist failed; falling back to memory saver: {}", exc)
            fallback = await self._adelegate("alist", config, before=before, limit=limit)
            if hasattr(fallback, "__aiter__"):
                async for item in fallback:
                    yield item
                return
            for item in fallback or []:
                yield item

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Any,
    ) -> Dict[str, Any]:
        try:
            return await self._asave_checkpoint(
                config,
                checkpoint,
                metadata,
                new_versions=new_versions,
                redis_client=self._aredis,
            )
        except Exception as exc:
            logger.warning("Redis checkpoint aput failed; falling back to memory saver: {}", exc)
            return await self._adelegate("aput", config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Sequence[Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        try:
            await self._aappend_writes(
                config,
                writes=writes,
                task_id=task_id,
                task_path=task_path,
                redis_client=self._aredis,
            )
            return None
        except Exception as exc:
            logger.warning("Redis checkpoint aput_writes failed; falling back to memory saver: {}", exc)
            await self._adelegate("aput_writes", config, writes, task_id, task_path)
            return None

    def _load_checkpoint_tuple(
        self,
        config: Dict[str, Any],
        *,
        redis_client: SyncRedis,
    ) -> Optional[CheckpointTuple]:
        doc = self._load_checkpoint_document(config, redis_client=redis_client)
        if not doc:
            return None
        return self._document_to_tuple(doc)

    async def _aload_checkpoint_tuple(
        self,
        config: Dict[str, Any],
        *,
        redis_client: AsyncRedis,
    ) -> Optional[CheckpointTuple]:
        doc = await self._aload_checkpoint_document(config, redis_client=redis_client)
        if not doc:
            return None
        return self._document_to_tuple(doc)

    def _list_checkpoints(
        self,
        config: Optional[Dict[str, Any]],
        *,
        before: Optional[Dict[str, Any]],
        limit: Optional[int],
        redis_client: SyncRedis,
    ) -> List[CheckpointTuple]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        index_key = self._index_key(thread_id, checkpoint_ns)
        checkpoint_ids = redis_client.zrevrange(index_key, 0, (limit or 100) - 1)
        tuples: List[CheckpointTuple] = []
        for checkpoint_id in checkpoint_ids:
            doc = self._load_document_by_id(
                thread_id,
                checkpoint_ns,
                _decode_response(checkpoint_id),
                redis_client=redis_client,
            )
            if doc:
                tuples.append(self._document_to_tuple(doc))
        return tuples

    async def _alist_checkpoints(
        self,
        config: Optional[Dict[str, Any]],
        *,
        before: Optional[Dict[str, Any]],
        limit: Optional[int],
        redis_client: AsyncRedis,
    ) -> List[CheckpointTuple]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        index_key = self._index_key(thread_id, checkpoint_ns)
        checkpoint_ids = await redis_client.zrevrange(index_key, 0, (limit or 100) - 1)
        tuples: List[CheckpointTuple] = []
        for checkpoint_id in checkpoint_ids:
            doc = await self._aload_document_by_id(
                thread_id,
                checkpoint_ns,
                _decode_response(checkpoint_id),
                redis_client=redis_client,
            )
            if doc:
                tuples.append(self._document_to_tuple(doc))
        return tuples

    def _save_checkpoint(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        *,
        new_versions: Any,
        redis_client: SyncRedis,
    ) -> Dict[str, Any]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = str(
            checkpoint.get("id")
            or self._configurable(config).get("checkpoint_id")
            or f"cp-{int(_utc_now_timestamp() * 1000)}"
        )
        parent_checkpoint_id = self._configurable(config).get("checkpoint_id")
        key = self._doc_key(thread_id, checkpoint_ns, checkpoint_id)
        current_doc = self._load_document_by_id(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        ) or {}
        document = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "config": self._build_config(thread_id, checkpoint_ns, checkpoint_id),
            "checkpoint": self._encode_value(checkpoint),
            "metadata": self._encode_value(metadata or {}),
            "pending_writes": current_doc.get("pending_writes", []),
            "created_at": current_doc.get("created_at") or _utc_now_timestamp(),
            "updated_at": _utc_now_timestamp(),
            "new_versions": self._encode_value(new_versions),
        }
        self._json_set(redis_client, key, document)
        self._touch_indices(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        )
        return self._build_config(thread_id, checkpoint_ns, checkpoint_id)

    async def _asave_checkpoint(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        *,
        new_versions: Any,
        redis_client: AsyncRedis,
    ) -> Dict[str, Any]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = str(
            checkpoint.get("id")
            or self._configurable(config).get("checkpoint_id")
            or f"cp-{int(_utc_now_timestamp() * 1000)}"
        )
        parent_checkpoint_id = self._configurable(config).get("checkpoint_id")
        key = self._doc_key(thread_id, checkpoint_ns, checkpoint_id)
        current_doc = await self._aload_document_by_id(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        ) or {}
        document = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "config": self._build_config(thread_id, checkpoint_ns, checkpoint_id),
            "checkpoint": self._encode_value(checkpoint),
            "metadata": self._encode_value(metadata or {}),
            "pending_writes": current_doc.get("pending_writes", []),
            "created_at": current_doc.get("created_at") or _utc_now_timestamp(),
            "updated_at": _utc_now_timestamp(),
            "new_versions": self._encode_value(new_versions),
        }
        await self._ajson_set(redis_client, key, document)
        await self._atouch_indices(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        )
        return self._build_config(thread_id, checkpoint_ns, checkpoint_id)

    def _append_writes(
        self,
        config: Dict[str, Any],
        *,
        writes: Sequence[Sequence[Any]],
        task_id: str,
        task_path: str,
        redis_client: SyncRedis,
    ) -> None:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = self._resolve_checkpoint_id(config, thread_id, checkpoint_ns, redis_client)
        if not checkpoint_id:
            return None
        doc = self._load_document_by_id(thread_id, checkpoint_ns, checkpoint_id, redis_client=redis_client)
        if not doc:
            return None
        pending_writes = list(doc.get("pending_writes", []))
        for write in writes:
            if len(write) < 2:
                continue
            pending_writes.append(
                {
                    "task_id": task_id,
                    "task_path": task_path,
                    "channel": write[0],
                    "value": self._encode_value(write[1]),
                }
            )
        doc["pending_writes"] = pending_writes
        self._json_set(redis_client, self._doc_key(thread_id, checkpoint_ns, checkpoint_id), doc)
        return None

    async def _aappend_writes(
        self,
        config: Dict[str, Any],
        *,
        writes: Sequence[Sequence[Any]],
        task_id: str,
        task_path: str,
        redis_client: AsyncRedis,
    ) -> None:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = await self._aresolve_checkpoint_id(
            config,
            thread_id,
            checkpoint_ns,
            redis_client,
        )
        if not checkpoint_id:
            return None
        doc = await self._aload_document_by_id(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        )
        if not doc:
            return None
        pending_writes = list(doc.get("pending_writes", []))
        for write in writes:
            if len(write) < 2:
                continue
            pending_writes.append(
                {
                    "task_id": task_id,
                    "task_path": task_path,
                    "channel": write[0],
                    "value": self._encode_value(write[1]),
                }
            )
        doc["pending_writes"] = pending_writes
        await self._ajson_set(redis_client, self._doc_key(thread_id, checkpoint_ns, checkpoint_id), doc)
        return None

    def _touch_indices(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        *,
        redis_client: SyncRedis,
    ) -> None:
        latest_key = self._latest_key(thread_id, checkpoint_ns)
        index_key = self._index_key(thread_id, checkpoint_ns)
        redis_client.set(latest_key, checkpoint_id, ex=self.ttl)
        redis_client.zadd(index_key, {checkpoint_id: _utc_now_timestamp()})
        redis_client.expire(index_key, self.ttl)

    async def _atouch_indices(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        *,
        redis_client: AsyncRedis,
    ) -> None:
        latest_key = self._latest_key(thread_id, checkpoint_ns)
        index_key = self._index_key(thread_id, checkpoint_ns)
        await redis_client.set(latest_key, checkpoint_id, ex=self.ttl)
        await redis_client.zadd(index_key, {checkpoint_id: _utc_now_timestamp()})
        await redis_client.expire(index_key, self.ttl)

    def _load_checkpoint_document(
        self,
        config: Dict[str, Any],
        *,
        redis_client: SyncRedis,
    ) -> Optional[Dict[str, Any]]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = self._resolve_checkpoint_id(config, thread_id, checkpoint_ns, redis_client)
        if not checkpoint_id:
            return None
        return self._load_document_by_id(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        )

    async def _aload_checkpoint_document(
        self,
        config: Dict[str, Any],
        *,
        redis_client: AsyncRedis,
    ) -> Optional[Dict[str, Any]]:
        thread_id, checkpoint_ns = self._resolve_scope(config)
        checkpoint_id = await self._aresolve_checkpoint_id(
            config,
            thread_id,
            checkpoint_ns,
            redis_client,
        )
        if not checkpoint_id:
            return None
        return await self._aload_document_by_id(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            redis_client=redis_client,
        )

    def _load_document_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        *,
        redis_client: SyncRedis,
    ) -> Optional[Dict[str, Any]]:
        key = self._doc_key(thread_id, checkpoint_ns, checkpoint_id)
        raw = redis_client.execute_command("JSON.GET", key, "$")
        if not raw:
            return None
        payload = _decode_response(raw)
        data = json.loads(payload)
        if isinstance(data, list):
            return data[0]
        return data

    async def _aload_document_by_id(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        *,
        redis_client: AsyncRedis,
    ) -> Optional[Dict[str, Any]]:
        key = self._doc_key(thread_id, checkpoint_ns, checkpoint_id)
        raw = await redis_client.execute_command("JSON.GET", key, "$")
        if not raw:
            return None
        payload = _decode_response(raw)
        data = json.loads(payload)
        if isinstance(data, list):
            return data[0]
        return data

    def _json_set(self, redis_client: SyncRedis, key: str, payload: Dict[str, Any]) -> None:
        redis_client.execute_command("JSON.SET", key, "$", json.dumps(payload))
        redis_client.expire(key, self.ttl)

    async def _ajson_set(
        self,
        redis_client: AsyncRedis,
        key: str,
        payload: Dict[str, Any],
    ) -> None:
        await redis_client.execute_command("JSON.SET", key, "$", json.dumps(payload))
        await redis_client.expire(key, self.ttl)

    def _document_to_tuple(self, document: Dict[str, Any]) -> CheckpointTuple:
        pending_writes = [
            (
                item.get("task_id"),
                item.get("channel"),
                self._decode_value(item.get("value")),
            )
            for item in document.get("pending_writes", [])
        ]
        parent_checkpoint_id = document.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = self._build_config(
                document["thread_id"],
                document.get("checkpoint_ns", ""),
                parent_checkpoint_id,
            )
        return CheckpointTuple(
            config=document.get("config") or self._build_config(
                document["thread_id"],
                document.get("checkpoint_ns", ""),
                document["checkpoint_id"],
            ),
            checkpoint=self._decode_value(document.get("checkpoint")),
            metadata=self._decode_value(document.get("metadata")) or {},
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _resolve_scope(self, config: Optional[Dict[str, Any]]) -> Sequence[str]:
        configurable = self._configurable(config)
        thread_id = str(configurable.get("thread_id") or "default")
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        return thread_id, checkpoint_ns

    def _resolve_checkpoint_id(
        self,
        config: Dict[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        redis_client: SyncRedis,
    ) -> Optional[str]:
        configurable = self._configurable(config)
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id:
            return str(checkpoint_id)
        latest = redis_client.get(self._latest_key(thread_id, checkpoint_ns))
        return str(_decode_response(latest)) if latest else None

    async def _aresolve_checkpoint_id(
        self,
        config: Dict[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        redis_client: AsyncRedis,
    ) -> Optional[str]:
        configurable = self._configurable(config)
        checkpoint_id = configurable.get("checkpoint_id")
        if checkpoint_id:
            return str(checkpoint_id)
        latest = await redis_client.get(self._latest_key(thread_id, checkpoint_ns))
        return str(_decode_response(latest)) if latest else None

    @staticmethod
    def _configurable(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {}
        configurable = config.get("configurable", {})
        return configurable if isinstance(configurable, dict) else {}

    def _doc_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self.prefix}:doc:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _latest_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self.prefix}:latest:{thread_id}:{checkpoint_ns}"

    def _index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self.prefix}:index:{thread_id}:{checkpoint_ns}"

    @staticmethod
    def _build_config(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _encode_value(self, value: Any) -> Dict[str, str]:
        payload_type, raw = self.serde.dumps_typed(value)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return {
            "type": payload_type,
            "payload": base64.b64encode(raw).decode("ascii"),
        }

    def _decode_value(self, payload: Any) -> Any:
        if payload is None:
            return None
        raw = base64.b64decode(payload["payload"])
        return self.serde.loads_typed((payload["type"], raw))

    def _delegate(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fallback_method = getattr(self._memory, method, None)
        if callable(fallback_method):
            return fallback_method(*args, **kwargs)
        return None

    async def _adelegate(self, method: str, *args: Any, **kwargs: Any) -> Any:
        fallback_method = getattr(self._memory, method, None)
        if callable(fallback_method):
            result = fallback_method(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result

        sync_method = getattr(self._memory, method[1:] if method.startswith("a") else method, None)
        if callable(sync_method):
            return sync_method(*args, **kwargs)
        return None


@lru_cache(maxsize=1)
def build_graph_checkpointer() -> Any:
    backend = (settings.LANGGRAPH_CHECKPOINTER_BACKEND or "redis").strip().lower()
    if backend == "memory":
        return MemorySaver()
    return RedisStackCheckpointSaver()
