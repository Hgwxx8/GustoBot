import asyncio
import json

from gustobot.application.services.redis_checkpoint import RedisStackCheckpointSaver


class FakeRedisJSON:
    def __init__(self):
        self.values = {}
        self.expiry = {}
        self.sorted_sets = {}

    def execute_command(self, command, key, *args):
        if command == "JSON.SET":
            self.values[key] = json.loads(args[1])
            return "OK"
        if command == "JSON.GET":
            if key not in self.values:
                return None
            return json.dumps([self.values[key]])
        raise AssertionError(f"Unsupported command: {command}")

    def expire(self, key, ttl):
        self.expiry[key] = ttl

    def set(self, key, value, ex=None):
        self.values[key] = value
        if ex is not None:
            self.expiry[key] = ex

    def get(self, key):
        return self.values.get(key)

    def zadd(self, key, mapping):
        bucket = self.sorted_sets.setdefault(key, {})
        bucket.update(mapping)

    def zrevrange(self, key, start, end):
        bucket = self.sorted_sets.get(key, {})
        ordered = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        if end == -1:
            selected = ordered[start:]
        else:
            selected = ordered[start : end + 1]
        return [item[0] for item in selected]


class FakeAsyncRedisJSON(FakeRedisJSON):
    async def execute_command(self, command, key, *args):
        return super().execute_command(command, key, *args)

    async def expire(self, key, ttl):
        super().expire(key, ttl)

    async def set(self, key, value, ex=None):
        super().set(key, value, ex=ex)

    async def get(self, key):
        return super().get(key)

    async def zadd(self, key, mapping):
        super().zadd(key, mapping)

    async def zrevrange(self, key, start, end):
        return super().zrevrange(key, start, end)


def test_redis_stack_checkpoint_roundtrip_and_pending_writes():
    sync_redis = FakeRedisJSON()
    async_redis = FakeAsyncRedisJSON()
    saver = RedisStackCheckpointSaver(
        sync_redis=sync_redis,
        async_redis=async_redis,
        ttl=300,
        prefix="test:checkpoint",
    )

    base_config = {"configurable": {"thread_id": "thread-1"}}
    returned_config = saver.put(
        base_config,
        checkpoint={"id": "cp-1", "channel_values": {"messages": ["hello"]}},
        metadata={"source": "unit-test"},
        new_versions={},
    )
    saver.put_writes(
        returned_config,
        writes=[("messages", {"content": "pending"})],
        task_id="task-1",
        task_path="chat",
    )

    loaded = saver.get_tuple(base_config)

    assert loaded is not None
    assert loaded.config["configurable"]["checkpoint_id"] == "cp-1"
    assert loaded.checkpoint["channel_values"]["messages"] == ["hello"]
    assert loaded.metadata["source"] == "unit-test"
    assert loaded.pending_writes == [
        ("task-1", "messages", {"content": "pending"})
    ]


def test_redis_stack_checkpoint_async_roundtrip():
    sync_redis = FakeRedisJSON()
    async_redis = FakeAsyncRedisJSON()
    saver = RedisStackCheckpointSaver(
        sync_redis=sync_redis,
        async_redis=async_redis,
        ttl=300,
        prefix="test:checkpoint",
    )

    async def _roundtrip():
        config = {"configurable": {"thread_id": "thread-2"}}
        await saver.aput(
            config,
            checkpoint={"id": "cp-async", "channel_values": {"messages": ["hi"]}},
            metadata={"source": "async-test"},
            new_versions={},
        )
        return await saver.aget_tuple(config)

    loaded = asyncio.run(_roundtrip())

    assert loaded is not None
    assert loaded.config["configurable"]["checkpoint_id"] == "cp-async"
    assert loaded.metadata["source"] == "async-test"
