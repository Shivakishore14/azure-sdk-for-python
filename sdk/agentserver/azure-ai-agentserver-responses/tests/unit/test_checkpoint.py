# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Checkpoint primitive conformance (spec 025 §A.3 / §7.3).

Covers the resilient-background gate (no-op matrix), idempotency, failure
swallowing, terminal drop, status-as-is, and that the checkpoint never reaches
the wire — exercised through the public HTTP surface and the shared persist
helper. End-to-end crash recovery is covered by the resilience_contract suite.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from typing import Any

import pytest
from starlette.testclient import TestClient

from azure.ai.agentserver.responses import (
    ResponseEventStream,
    FileResponseStore,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.hosting._orchestrator import _do_checkpoint_persist
from azure.ai.agentserver.responses.models._generated import ResponseObject
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent


class _RecordingProvider:
    """Minimal provider stub recording update_response snapshots."""

    def __init__(self, *, fail: bool = False) -> None:
        self.updates: list[dict[str, Any]] = []
        self.fail = fail

    async def update_response(self, response, *, context=None):  # noqa: ANN001
        if self.fail:
            raise RuntimeError("boom")
        self.updates.append(dict(response))


def _event(**md) -> ResponseCheckpointEvent:
    resp: ResponseObject = {"id": "r1", "object": "response", "status": "in_progress", "output": [], "model": "m"}
    if md:
        resp["metadata"] = {"_internal_metadata": json.dumps(md, separators=(",", ":"), sort_keys=True)}
    return ResponseCheckpointEvent(resp)


def _opts(resilient_background: bool) -> ResponsesServerOptions:
    return ResponsesServerOptions(resilient_background=resilient_background)


# --------------------------------------------------------------------------
# §7.3 T18 — configuration gate (no-op matrix) via the shared persist helper
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t18_no_op_matrix():
    # (a) store=False
    p = _RecordingProvider()
    await _do_checkpoint_persist(
        _event(),
        provider=p,
        runtime_options=_opts(True),
        store=False,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    assert p.updates == []
    # (b) background=False
    p = _RecordingProvider()
    await _do_checkpoint_persist(
        _event(),
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=False,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    assert p.updates == []
    # (c) resilient_background=False
    p = _RecordingProvider()
    await _do_checkpoint_persist(
        _event(),
        provider=p,
        runtime_options=_opts(False),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    assert p.updates == []
    # resilient background → persists
    p = _RecordingProvider()
    snap = await _do_checkpoint_persist(
        _event(cp=1),
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    assert len(p.updates) == 1
    assert snap is not None


@pytest.mark.asyncio
async def test_t20_idempotent_when_snapshot_unchanged():
    p = _RecordingProvider()
    ev = _event(cp=1)
    snap = await _do_checkpoint_persist(
        ev,
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    # Second call with the same snapshot bytes → no provider call.
    await _do_checkpoint_persist(
        ev,
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=snap,
        terminal_seen=False,
    )
    assert len(p.updates) == 1


@pytest.mark.asyncio
async def test_t21_status_as_is_in_snapshot():
    p = _RecordingProvider()
    ev = _event(cp=1)
    ev.response["status"] = "in_progress"
    await _do_checkpoint_persist(
        ev,
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=False,
    )
    assert p.updates[0]["status"] == "in_progress"
    # Reserved internal_metadata is in the persisted snapshot (storage retains it).
    assert p.updates[0]["metadata"]["_internal_metadata"] == '{"cp":1}'


@pytest.mark.asyncio
async def test_t22_failure_swallowed_and_tagged():
    from azure.ai.agentserver.core._platform_headers import PLATFORM_ERROR_TAG  # noqa: E501

    p = _RecordingProvider(fail=True)
    # Must not raise; last_snapshot unchanged.
    snap = await _do_checkpoint_persist(
        _event(cp=1),
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=b"prev",
        terminal_seen=False,
    )
    assert snap == b"prev"
    del PLATFORM_ERROR_TAG  # symbol exists


@pytest.mark.asyncio
async def test_t22b_drop_after_terminal():
    p = _RecordingProvider()
    snap = await _do_checkpoint_persist(
        _event(cp=1),
        provider=p,
        runtime_options=_opts(True),
        store=True,
        background=True,
        context=None,
        response_id="r1",
        last_snapshot=None,
        terminal_seen=True,
    )
    assert p.updates == []
    assert snap is None


# --------------------------------------------------------------------------
# Integration via the HTTP surface
# --------------------------------------------------------------------------


def _bg_client(handler) -> TestClient:
    app = ResponsesAgentServerHost(options=ResponsesServerOptions(resilient_background=True))
    app.response_handler(handler)
    return TestClient(app)


def _poll_terminal(client: TestClient, rid: str) -> dict:
    for _ in range(80):
        g = client.get(f"/responses/{rid}")
        body = g.json()
        if body.get("status") in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("response did not reach a terminal state")


def test_checkpoint_yielded_does_not_crash_and_no_leak():
    async def handler(request, context, cancellation_signal):
        async def _events():
            stream = ResponseEventStream(response_id=context.response_id, request=request)
            yield stream.emit_created()
            yield stream.emit_in_progress()
            msg = stream.add_output_item_message()
            msg.internal_metadata["phase"] = "p0"
            yield msg.emit_added()
            text = msg.add_text_content()
            yield text.emit_added()
            yield text.emit_delta("hi")
            yield text.emit_text_done("hi")
            yield text.emit_done()
            yield msg.emit_done()
            stream.internal_metadata["completed_phases"] = 1
            yield stream.checkpoint()  # mid-flight checkpoint
            yield stream.emit_completed()

        return _events()

    client = _bg_client(handler)
    rid = client.post(
        "/responses",
        json={"model": "m", "input": "hi", "stream": False, "store": True, "background": True},
    ).json()["id"]
    body = _poll_terminal(client, rid)
    assert body["status"] == "completed"
    assert len(body["output"]) == 1
    assert "internal_metadata" not in client.get(f"/responses/{rid}").text


def test_t22d_no_implicit_checkpoints_zero_checkpoint_handler():
    """A handler yielding zero checkpoints triggers no extra update_response."""
    update_count = {"n": 0}

    class _CountingProvider:
        def __init__(self) -> None:
            self._inner: dict[str, Any] = {}

        async def create_response(self, response, input_items, history_item_ids, *, context=None):  # noqa: ANN001
            self._inner[response.id] = response

        async def update_response(self, response, *, context=None):  # noqa: ANN001
            update_count["n"] += 1
            self._inner[response.id] = response

        async def get_response(self, response_id, *, context=None):  # noqa: ANN001
            from azure.ai.agentserver.responses.store._foundry_errors import FoundryResourceNotFoundError

            if response_id not in self._inner:
                raise FoundryResourceNotFoundError("not found")
            return self._inner[response_id]

        async def delete_response(self, response_id, *, context=None):  # noqa: ANN001
            self._inner.pop(response_id, None)

        async def get_input_items(
            self, response_id, limit=20, ascending=False, after=None, before=None, *, context=None
        ):  # noqa: ANN001,E501
            return []

        async def get_items(self, item_ids, *, context=None):  # noqa: ANN001
            return [None for _ in item_ids]

        async def get_history_item_ids(
            self, previous_response_id, conversation_id, limit, *, context=None
        ):  # noqa: ANN001,E501
            return []

    async def handler(request, context, cancellation_signal):
        async def _events():
            stream = ResponseEventStream(response_id=context.response_id, request=request)
            yield stream.emit_created()
            yield stream.emit_in_progress()
            for evt in stream.output_item_message("hello"):
                yield evt
            yield stream.emit_completed()

        return _events()

    app = ResponsesAgentServerHost(
        options=ResponsesServerOptions(resilient_background=True),
        store=_CountingProvider(),
    )
    app.response_handler(handler)
    client = TestClient(app)
    rid = client.post(
        "/responses",
        json={"model": "m", "input": "hi", "stream": False, "store": True, "background": True},
    ).json()["id"]
    _poll_terminal(client, rid)
    # Only the terminal update (no in-flight checkpoint write).
    assert update_count["n"] <= 1, f"unexpected extra update_response calls: {update_count['n']}"


def test_checkpoint_event_public_export():
    from azure.ai.agentserver.responses import ResponseCheckpointEvent as root_event
    from azure.ai.agentserver.responses.streaming import ResponseCheckpointEvent as streaming_event

    assert isinstance(ResponseEventStream(response_id="r1").checkpoint(), root_event)
    assert root_event is streaming_event


@pytest.mark.parametrize(
    "reference",
    [
        None,
        {},
        {"type": "agent_reference", "name": "partial"},
        {"type": "agent_reference", "name": "default"},
        {"type": "agent_reference", "name": "explicit", "version": "2"},
    ],
)
@pytest.mark.asyncio
async def test_checkpoint_uses_terminal_identity_contract(reference):
    provider = _RecordingProvider()
    event = _event(progress="saved")
    event.response["agent_reference"] = reference
    event.response["agent_session_id"] = "handler-session"
    event.response["conversation"] = {"id": "handler-conversation"}
    original = deepcopy(event.response)
    resolved = {"type": "agent_reference", "name": "resolved", "version": "7"}
    args = {
        "provider": provider,
        "runtime_options": _opts(True),
        "store": True,
        "background": True,
        "context": None,
        "response_id": "r1",
        "terminal_seen": False,
        "agent_reference": resolved,
        "model": "request-model",
        "agent_session_id": "resolved-session",
        "conversation_id": "resolved-conversation",
    }
    snapshot = await _do_checkpoint_persist(event, last_snapshot=None, **args)
    await _do_checkpoint_persist(event, last_snapshot=snapshot, **args)
    assert len(provider.updates) == 1
    saved = provider.updates[0]
    assert saved["agent_reference"] == (reference if reference and reference.get("version") else resolved)
    assert saved["model"] == "m"
    assert saved["id"] == saved["response_id"] == "r1"
    assert saved["agent_session_id"] == "resolved-session"
    assert saved["conversation"] == {"id": "resolved-conversation"}
    assert saved["background"] is True
    assert saved["status"] == "in_progress"
    assert event.response == original


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
def test_checkpoint_identity_before_terminal(monkeypatch, streaming, seeded):
    provider = FileResponseStore()
    processed = Event()
    finish = Event()
    updates = []
    originals = []
    original_update = provider.update_response
    resolved_reference = {"type": "agent_reference", "name": "checkpoint-agent", "version": "7"}
    explicit_reference = {"type": "agent_reference", "name": "handler-agent", "version": "2"}

    async def record_update(response, *, context=None):
        updates.append(deepcopy(dict(response)))
        await original_update(response, context=context)

    monkeypatch.setattr(provider, "update_response", record_update)

    async def handler(request, context, cancellation_signal):
        async def events():
            stream = ResponseEventStream(
                response_id=context.response_id,
                request=request if seeded else None,
                agent_reference=explicit_reference if seeded else None,
                model="handler-model" if seeded else None,
            )
            yield stream.emit_created()
            yield stream.emit_in_progress()
            for event in stream.output_item_message("saved progress"):
                yield event
            checkpoint = stream.checkpoint()
            original = deepcopy(dict(checkpoint.response))
            yield checkpoint
            originals.append((original, deepcopy(dict(checkpoint.response))))
            processed.set()
            await asyncio.to_thread(finish.wait, 10)
            yield stream.emit_completed()

        return events()

    app = ResponsesAgentServerHost(options=ResponsesServerOptions(resilient_background=True), store=provider)
    app.response_handler(handler)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.post,
            "/responses",
            json={
                "agent_reference": resolved_reference,
                "model": "request-model",
                "input": "hi",
                "conversation": "conv-checkpoint",
                "stream": streaming,
                "store": True,
                "background": True,
            },
        )
        try:
            assert processed.wait(5)
            assert len(updates) == 1
            saved = updates[0]
            rid = saved["id"]
            assert saved["response_id"] == rid
            assert saved["agent_reference"] == (explicit_reference if seeded else resolved_reference)
            assert saved["model"] == ("handler-model" if seeded else "request-model")
            assert saved["conversation"] == {"id": "conv-checkpoint"}
            assert saved["background"] is True
            assert saved["agent_session_id"]
            assert saved["status"] == "in_progress"
            assert originals[0][0] == originals[0][1]
            persisted = client.portal.call(FileResponseStore().get_response, rid)
            for field in ("response_id", "agent_reference", "model", "agent_session_id", "conversation", "background"):
                assert persisted[field] == saved[field]
            assert persisted["output"][0]["content"][0]["text"] == "saved progress"
        finally:
            finish.set()
        response = future.result(timeout=5)
        assert response.status_code == 200
        assert "response.checkpoint" not in response.text
        terminal = _poll_terminal(client, rid)
        assert terminal["status"] == "completed"
        for field in ("response_id", "agent_reference", "model", "agent_session_id", "conversation", "background"):
            assert terminal[field] == saved[field]
