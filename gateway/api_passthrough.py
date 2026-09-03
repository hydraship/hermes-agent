"""Minimal API passthrough for subscription-backed Codex inference.

This path deliberately bypasses ``AIAgent``.  The API caller owns the system
prompt, conversation history, and external tool loop; Hermes only resolves its
configured Codex OAuth client and performs one model request.
"""

from typing import Any, Dict


_FORWARDED_FIELDS = (
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "response_format",
    "seed",
)


async def run_chat_completion(body: Dict[str, Any]) -> Any:
    """Run exactly one completion through the configured Codex OAuth client."""
    from agent.auxiliary_client import _get_cached_client
    from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs

    runtime = _resolve_runtime_agent_kwargs()
    provider = str(runtime.get("provider") or "").strip().lower()
    if provider != "openai-codex":
        raise RuntimeError(
            "API passthrough requires the openai-codex subscription provider."
        )

    configured_model = _resolve_gateway_model()
    client, model = _get_cached_client(
        provider,
        model=configured_model,
        async_mode=True,
        main_runtime=runtime,
    )
    if client is None or not model:
        raise RuntimeError("Unable to resolve the openai-codex subscription client.")

    request = {
        key: body[key]
        for key in _FORWARDED_FIELDS
        if key in body and key != "messages"
    }
    request["model"] = model
    request["messages"] = body["messages"]

    return await client.chat.completions.create(**request)
