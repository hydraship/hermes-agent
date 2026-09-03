from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.api_passthrough import run_chat_completion


@pytest.mark.asyncio
async def test_run_chat_completion_uses_codex_subscription_client_once():
    response = SimpleNamespace(choices=[], usage=None)
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    body = {
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "List shipments"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "list_shipments",
                "description": "List shipments",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tokens": 2000,
        "temperature": 0.2,
        "stream": False,
    }
    runtime = {
        "provider": "openai-codex",
        "api_key": "subscription-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        return_value=runtime,
    ), patch(
        "gateway.run._resolve_gateway_model",
        return_value="gpt-5.5",
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.5"),
    ) as get_client:
        result = await run_chat_completion(body)

    assert result is response
    get_client.assert_called_once_with(
        "openai-codex",
        model="gpt-5.5",
        async_mode=True,
        main_runtime=runtime,
    )
    create.assert_awaited_once_with(
        model="gpt-5.5",
        messages=body["messages"],
        tools=body["tools"],
        tool_choice="auto",
        parallel_tool_calls=False,
        max_tokens=2000,
        temperature=0.2,
    )


@pytest.mark.asyncio
async def test_run_chat_completion_rejects_non_codex_provider():
    runtime = {"provider": "openai", "api_key": "direct-api-key"}

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs",
        return_value=runtime,
    ):
        with pytest.raises(RuntimeError, match="openai-codex"):
            await run_chat_completion({
                "messages": [{"role": "user", "content": "Hello"}],
            })
