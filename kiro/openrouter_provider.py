"""
OpenRouter provider for Open AI Gateway.

Routes models with provider/model format (e.g. openai/gpt-4o, anthropic/claude-sonnet-4)
to the OpenRouter API. Supports both OpenAI and Anthropic client formats with full
tool calling integration.

Endpoint: POST https://openrouter.ai/api/v1/chat/completions
Auth:     Authorization: Bearer {OPENROUTER_API_KEY}
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_ENABLED

_openrouter_client: Optional[httpx.AsyncClient] = None

KNOWN_OPENROUTER_PREFIXES = {
    "openai", "anthropic", "meta-llama", "google", "mistralai",
    "cohere", "deepseek", "qwen", "nvidia", "openrouter",
    "microsoft", "perplexity", "x-ai", "nousresearch",
    "cognitivecomputations", "01-ai", "databricks",
}

OPENROUTER_MODELS: List[Dict[str, str]] = [
    {"id": "openai/gpt-4o", "display_name": "GPT-4o (via OpenRouter)"},
    {"id": "openai/gpt-4.1", "display_name": "GPT-4.1 (via OpenRouter)"},
    {"id": "anthropic/claude-sonnet-4", "display_name": "Claude Sonnet 4 (via OpenRouter)"},
    {"id": "google/gemini-2.5-pro", "display_name": "Gemini 2.5 Pro (via OpenRouter)"},
    {"id": "meta-llama/llama-4-maverick", "display_name": "Llama 4 Maverick (via OpenRouter)"},
    {"id": "deepseek/deepseek-r1", "display_name": "DeepSeek R1 (via OpenRouter)"},
    {"id": "openrouter/auto", "display_name": "OpenRouter Auto (best model)"},
]


def is_openrouter_model(model_name: str) -> bool:
    """Return True if the model should be routed to OpenRouter."""
    if not OPENROUTER_ENABLED or not OPENROUTER_API_KEY:
        return False
    if "/" not in model_name:
        return False
    prefix = model_name.split("/")[0].lower()
    return prefix in KNOWN_OPENROUTER_PREFIXES


def get_openrouter_models() -> List[Dict[str, str]]:
    """Return list of OpenRouter models for /v1/models endpoint."""
    if not OPENROUTER_ENABLED or not OPENROUTER_API_KEY:
        return []
    return OPENROUTER_MODELS


def _get_openrouter_client() -> httpx.AsyncClient:
    global _openrouter_client
    if _openrouter_client is None or _openrouter_client.is_closed:
        _openrouter_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=120,
            ),
        )
        logger.info("Created shared OpenRouter HTTP client (HTTP/2, pooled)")
    return _openrouter_client


async def close_openrouter_client() -> None:
    global _openrouter_client
    if _openrouter_client and not _openrouter_client.is_closed:
        await _openrouter_client.aclose()
        _openrouter_client = None
        logger.info("Closed shared OpenRouter HTTP client")


# ==================================================================================================
# Anthropic → OpenAI conversion helpers
# ==================================================================================================


def _convert_anthropic_tools_to_openai(tools: List[Dict]) -> List[Dict]:
    """Convert Anthropic tool definitions to OpenAI format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


def _convert_anthropic_messages_to_openai(
    messages: List[Dict], system: Optional[str] = None
) -> List[Dict]:
    """Convert Anthropic messages to OpenAI format."""
    openai_messages: List[Dict] = []

    if system:
        openai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": str(content) if content else ""})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "".join(text_parts)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            openai_messages.append(assistant_msg)

        elif role == "user":
            text_parts = []
            tool_results = []
            for block in content:
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "".join(
                            b.get("text", "") for b in result_content if b.get("type") == "text"
                        )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result_content,
                    })

            if tool_results:
                openai_messages.extend(tool_results)
            if text_parts:
                openai_messages.append({"role": "user", "content": "".join(text_parts)})

    return openai_messages


def _build_openai_payload(request_data: Dict, model: str) -> Dict:
    """Build OpenAI-format payload from Anthropic request data."""
    system_text = ""
    system_field = request_data.get("system", "")
    if isinstance(system_field, str):
        system_text = system_field
    elif isinstance(system_field, list):
        system_text = "".join(
            b.get("text", "") for b in system_field if isinstance(b, dict)
        )

    messages = _convert_anthropic_messages_to_openai(
        request_data.get("messages", []), system_text
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    if request_data.get("max_tokens"):
        payload["max_tokens"] = request_data["max_tokens"]
    if request_data.get("temperature") is not None:
        payload["temperature"] = request_data["temperature"]
    if request_data.get("top_p") is not None:
        payload["top_p"] = request_data["top_p"]
    if request_data.get("stop_sequences"):
        payload["stop"] = request_data["stop_sequences"]

    tools = request_data.get("tools")
    if tools:
        payload["tools"] = _convert_anthropic_tools_to_openai(tools)

    return payload


# ==================================================================================================
# SSE event helpers (Anthropic format)
# ==================================================================================================


def _make_message_start_event(model: str, message_id: str) -> str:
    data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    return f"event: message_start\ndata: {json.dumps(data)}\n\n"


def _make_text_block_start_event(index: int) -> str:
    data = {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }
    return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"


def _make_text_delta_event(index: int, text: str) -> str:
    data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"


def _make_tool_use_block_start_event(index: int, tool_id: str, tool_name: str) -> str:
    data = {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {},
        },
    }
    return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"


def _make_tool_input_delta_event(index: int, partial_json: str) -> str:
    data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"


def _make_block_stop_event(index: int) -> str:
    data = {"type": "content_block_stop", "index": index}
    return f"event: content_block_stop\ndata: {json.dumps(data)}\n\n"


def _make_message_delta_event(stop_reason: str = "end_turn") -> str:
    data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }
    return f"event: message_delta\ndata: {json.dumps(data)}\n\n"


def _make_message_stop_event() -> str:
    data = {"type": "message_stop"}
    return f"event: message_stop\ndata: {json.dumps(data)}\n\n"


# ==================================================================================================
# Error handling
# ==================================================================================================


def _raise_openrouter_error(status_code: int, body_text: str = "") -> None:
    if status_code == 401:
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "OpenRouter API key is invalid. Check OPENROUTER_API_KEY.",
                },
            },
        )
    if status_code == 402:
        raise HTTPException(
            status_code=402,
            detail={
                "type": "error",
                "error": {
                    "type": "billing_error",
                    "message": "Insufficient OpenRouter credits. Check your account balance.",
                },
            },
        )
    if status_code == 429:
        raise HTTPException(
            status_code=429,
            detail={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "OpenRouter rate limit reached. Please wait before retrying.",
                },
            },
        )
    raise HTTPException(
        status_code=502,
        detail={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"OpenRouter API error (HTTP {status_code}): {body_text[:200]}",
            },
        },
    )


# ==================================================================================================
# Streaming: Anthropic output (for /v1/messages clients)
# ==================================================================================================


async def stream_openrouter_anthropic(
    request_data: Dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream an OpenRouter response translated to Anthropic SSE format.

    Converts Anthropic request to OpenAI format, sends to OpenRouter,
    and translates the OpenAI SSE response back to Anthropic format.
    """
    payload = _build_openai_payload(request_data, model)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "HTTP-Referer": "https://github.com/openai-gateway",
        "X-Title": "Open AI Gateway",
    }

    logger.info(f"Sending request to OpenRouter (model={model})")

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    block_index = 0
    text_block_started = False
    has_tool_calls = False
    tool_call_states: Dict[int, Dict] = {}

    try:
        client = _get_openrouter_client()
        response_cm = client.stream("POST", url, json=payload, headers=headers)
        response = await response_cm.__aenter__()

        try:
            if response.status_code >= 400:
                try:
                    body = await response.aread()
                    body_text = body.decode("utf-8", errors="replace")
                except Exception:
                    body_text = f"HTTP {response.status_code}"
                logger.error(f"OpenRouter returned HTTP {response.status_code}: {body_text[:200]}")
                _raise_openrouter_error(response.status_code, body_text)

            yield _make_message_start_event(model, message_id)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: "):]
                if raw.strip() in ("", "[DONE]"):
                    continue

                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Text content
                content = delta.get("content")
                if content:
                    if not text_block_started:
                        yield _make_text_block_start_event(block_index)
                        text_block_started = True
                    yield _make_text_delta_event(block_index, content)

                # Tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    has_tool_calls = True

                    if text_block_started:
                        yield _make_block_stop_event(block_index)
                        block_index += 1
                        text_block_started = False

                    for tc in tool_calls:
                        tc_index = tc.get("index", 0)

                        if tc_index not in tool_call_states:
                            tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                            tool_name = tc.get("function", {}).get("name", "")
                            tool_call_states[tc_index] = {
                                "id": tool_id,
                                "name": tool_name,
                                "block_index": block_index,
                            }
                            yield _make_tool_use_block_start_event(
                                block_index, tool_id, tool_name
                            )

                        args_delta = tc.get("function", {}).get("arguments", "")
                        if args_delta:
                            bi = tool_call_states[tc_index]["block_index"]
                            yield _make_tool_input_delta_event(bi, args_delta)

                # Finish
                if finish_reason:
                    for tc_idx in sorted(tool_call_states.keys()):
                        bi = tool_call_states[tc_idx]["block_index"]
                        yield _make_block_stop_event(bi)
                        block_index = bi + 1
                    tool_call_states.clear()

        finally:
            await response_cm.__aexit__(None, None, None)

    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        logger.error(f"OpenRouter request timed out: {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "OpenRouter request timed out. Please try again.",
                },
            },
        ) from e
    except httpx.RequestError as e:
        logger.error(f"OpenRouter network error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Could not reach OpenRouter API: {e}",
                },
            },
        ) from e

    if text_block_started:
        yield _make_block_stop_event(block_index)

    if block_index == 0 and not text_block_started and not has_tool_calls:
        yield _make_text_block_start_event(0)
        yield _make_block_stop_event(0)

    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    yield _make_message_delta_event(stop_reason)
    yield _make_message_stop_event()


# ==================================================================================================
# Streaming: OpenAI output (for /v1/chat/completions clients)
# ==================================================================================================


async def stream_openrouter_openai(
    request_data: Dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream an OpenRouter response in OpenAI SSE format (near pass-through).

    The client already sent an OpenAI-format request, so we forward it
    to OpenRouter and relay the SSE chunks back directly.
    """
    messages = []
    for msg in request_data.get("messages", []):
        if isinstance(msg, dict):
            messages.append(msg)
        else:
            messages.append(msg.model_dump() if hasattr(msg, "model_dump") else dict(msg))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    if request_data.get("max_tokens"):
        payload["max_tokens"] = request_data["max_tokens"]
    if request_data.get("temperature") is not None:
        payload["temperature"] = request_data["temperature"]
    if request_data.get("top_p") is not None:
        payload["top_p"] = request_data["top_p"]
    if request_data.get("stop"):
        payload["stop"] = request_data["stop"]
    if request_data.get("tools"):
        tools = request_data["tools"]
        payload["tools"] = [
            t.model_dump() if hasattr(t, "model_dump") else t for t in tools
        ]
    if request_data.get("tool_choice"):
        tc = request_data["tool_choice"]
        payload["tool_choice"] = tc.model_dump() if hasattr(tc, "model_dump") else tc

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "HTTP-Referer": "https://github.com/openai-gateway",
        "X-Title": "Open AI Gateway",
    }

    logger.info(f"Sending request to OpenRouter (model={model}, format=openai)")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    try:
        client = _get_openrouter_client()
        response_cm = client.stream("POST", url, json=payload, headers=headers)
        response = await response_cm.__aenter__()

        try:
            if response.status_code >= 400:
                try:
                    body = await response.aread()
                    body_text = body.decode("utf-8", errors="replace")
                except Exception:
                    body_text = f"HTTP {response.status_code}"
                logger.error(f"OpenRouter returned HTTP {response.status_code}: {body_text[:200]}")
                _raise_openrouter_error(response.status_code, body_text)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                if not raw.strip():
                    continue
                yield f"data: {raw}\n\n"

        finally:
            await response_cm.__aexit__(None, None, None)

    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        logger.error(f"OpenRouter request timed out: {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "OpenRouter request timed out. Please try again.",
                },
            },
        ) from e
    except httpx.RequestError as e:
        logger.error(f"OpenRouter network error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Could not reach OpenRouter API: {e}",
                },
            },
        ) from e
