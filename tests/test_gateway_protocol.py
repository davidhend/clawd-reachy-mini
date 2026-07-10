"""Protocol-focused tests for GatewayClient."""

from __future__ import annotations

import asyncio

import pytest

import clawd_reachy_mini.gateway as gateway_mod
from clawd_reachy_mini.config import Config
from clawd_reachy_mini.gateway import GatewayClient, ReplyPending


@pytest.mark.asyncio
async def test_handle_connect_challenge_sends_connect_request():
    client = GatewayClient(Config(gateway_token="secret-token"))
    client._auth_event = asyncio.Event()
    sent: list[dict] = []

    async def fake_send_raw(data: dict) -> None:
        sent.append(data)

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    await client._handle_event("connect.challenge", {"payload": {"nonce": "n", "ts": "t"}})

    assert client._authenticated is True
    assert client._auth_event.is_set()
    assert sent
    assert sent[0]["type"] == "req"
    assert sent[0]["method"] == "connect"
    assert sent[0]["params"]["auth"]["token"] == "secret-token"
    # chat.send requires operator.write; the gateway only grants scopes that
    # the connect request asks for.
    assert "operator.write" in sent[0]["params"]["scopes"]


@pytest.mark.asyncio
async def test_handle_res_hello_ok_sets_register_event():
    client = GatewayClient(Config())
    client._register_event = asyncio.Event()

    await client._handle_message(
        {
            "type": "res",
            "ok": True,
            "payload": {"type": "hello-ok"},
        }
    )

    assert client._register_event.is_set()


@pytest.mark.asyncio
async def test_send_message_uses_chat_send_protocol_and_returns_text():
    client = GatewayClient(Config())
    client._connected = True
    client._ws = object()  # type: ignore[assignment]
    sent: list[dict] = []

    async def fake_send_raw(data: dict) -> None:
        sent.append(data)
        if data.get("method") == "chat.send":
            message_id = data["id"]
            future = client._response_handlers[message_id]
            future.set_result({"text": "pong"})

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    response = await client.send_message("ping")

    assert response == "pong"
    assert sent
    assert sent[0]["type"] == "req"
    assert sent[0]["method"] == "chat.send"
    params = sent[0]["params"]
    assert params["message"] == "ping"
    assert params["sessionKey"].startswith("reachy-mini:")
    assert params["idempotencyKey"] == sent[0]["id"]


@pytest.mark.asyncio
async def test_slow_run_raises_reply_pending_and_delivers_late_result(monkeypatch):
    """A run that outlives the reply timeout is not abandoned: send_message
    raises ReplyPending and the final text arrives via the late callback."""
    monkeypatch.setattr(gateway_mod, "REPLY_TIMEOUT_S", 0.05)
    client = GatewayClient(Config())
    client._connected = True
    client._ws = object()  # type: ignore[assignment]

    late: list[str] = []

    async def on_late(text: str) -> None:
        late.append(text)

    client.register_late_result_callback(on_late)

    async def fake_send_raw(data: dict) -> None:
        if data.get("method") == "chat.send":
            client._response_handlers[data["id"]].set_result({"runId": "run-1"})

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    with pytest.raises(ReplyPending):
        await client.send_message("install updates on dns01")

    # The run handler must survive the timeout, marked as late.
    assert client._response_handlers["run-1"]["late"] is True

    # Text keeps accumulating after the timeout, then the run ends.
    await client._handle_event(
        "agent",
        {"payload": {"runId": "run-1", "stream": "assistant", "data": {"text": "Updates done."}}},
    )
    await client._handle_event(
        "agent",
        {"payload": {"runId": "run-1", "stream": "lifecycle", "data": {"phase": "end"}}},
    )
    await asyncio.sleep(0)  # let the callback task run

    assert late == ["Updates done."]
    assert "run-1" not in client._response_handlers


@pytest.mark.asyncio
async def test_slow_run_late_result_via_chat_complete(monkeypatch):
    """Same as above but the completion arrives as a chat-complete event."""
    monkeypatch.setattr(gateway_mod, "REPLY_TIMEOUT_S", 0.05)
    client = GatewayClient(Config())
    client._connected = True
    client._ws = object()  # type: ignore[assignment]

    late: list[str] = []

    async def on_late(text: str) -> None:
        late.append(text)

    client.register_late_result_callback(on_late)

    async def fake_send_raw(data: dict) -> None:
        if data.get("method") == "chat.send":
            client._response_handlers[data["id"]].set_result({"runId": "run-2"})

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    with pytest.raises(ReplyPending):
        await client.send_message("long task")

    await client._handle_event(
        "chat",
        {
            "payload": {
                "runId": "run-2",
                "state": "complete",
                "message": {"content": [{"type": "text", "text": "All patched."}]},
            }
        },
    )
    await asyncio.sleep(0)

    assert late == ["All patched."]
    assert "run-2" not in client._response_handlers


@pytest.mark.asyncio
async def test_slow_run_late_result_when_run_id_equals_message_id(monkeypatch):
    """The real OpenClaw gateway uses the chat.send idempotencyKey as the
    runId, so run_id == message_id. The send_message cleanup must not pop the
    still-pending run handler through the message_id alias (the bug that ate
    the Solbox announcement)."""
    monkeypatch.setattr(gateway_mod, "REPLY_TIMEOUT_S", 0.05)
    client = GatewayClient(Config())
    client._connected = True
    client._ws = object()  # type: ignore[assignment]

    late: list[str] = []

    async def on_late(text: str) -> None:
        late.append(text)

    client.register_late_result_callback(on_late)

    sent_ids: list[str] = []

    async def fake_send_raw(data: dict) -> None:
        if data.get("method") == "chat.send":
            sent_ids.append(data["id"])
            # runId == the request's own id, like the real gateway.
            client._response_handlers[data["id"]].set_result({"runId": data["id"]})

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    with pytest.raises(ReplyPending):
        await client.send_message("install updates on solbox")

    run_id = sent_ids[0]
    assert client._response_handlers[run_id]["late"] is True

    await client._handle_event(
        "agent",
        {"payload": {"runId": run_id, "stream": "assistant", "data": {"text": "Updates done."}}},
    )
    await client._handle_event(
        "agent",
        {"payload": {"runId": run_id, "stream": "lifecycle", "data": {"phase": "end"}}},
    )
    await asyncio.sleep(0)

    assert late == ["Updates done."]
    assert run_id not in client._response_handlers


@pytest.mark.asyncio
async def test_timeout_without_late_callback_still_raises_and_cleans_up(monkeypatch):
    """Without a registered callback the old behavior stands: timeout + cleanup."""
    monkeypatch.setattr(gateway_mod, "REPLY_TIMEOUT_S", 0.05)
    client = GatewayClient(Config())
    client._connected = True
    client._ws = object()  # type: ignore[assignment]

    async def fake_send_raw(data: dict) -> None:
        if data.get("method") == "chat.send":
            client._response_handlers[data["id"]].set_result({"runId": "run-3"})

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    with pytest.raises(asyncio.TimeoutError):
        await client.send_message("long task")

    assert "run-3" not in client._response_handlers


@pytest.mark.asyncio
async def test_tool_request_placeholder_returns_error_response():
    client = GatewayClient(Config())
    sent: list[dict] = []

    async def fake_send_raw(data: dict) -> None:
        sent.append(data)

    client._send_raw = fake_send_raw  # type: ignore[method-assign]

    await client._handle_tool_request(
        {
            "id": "tool-1",
            "tool": "reachy_move_head",
            "arguments": {"pitch": 10},
        }
    )

    assert sent == [
        {
            "type": "tool.response",
            "id": "tool-1",
            "result": {"status": "error", "message": "Tool handler not registered"},
        }
    ]
