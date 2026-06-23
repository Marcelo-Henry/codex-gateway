
"""
Google Gemini provider for Open AI Gateway.

Translates Anthropic Messages API requests to the Google Gemini
generateContent / streamGenerateContent API, streams the response back
as Anthropic-compatible SSE events.

Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse
Auth:      x-goog-api-key header (API key) or Authorization: Bearer (OAuth2)
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.gemini_auth import (
    get_gemini_auth_headers,
    get_auth_type,
    discover_project_id,
    GeminiAuthType,
    CODE_ASSIST_ENDPOINT,
    PUBLIC_GEMINI_ENDPOINT,
)

# Base URL for the Gemini generativelanguage API (kept for backwards compat)
_GEMINI_BASE_URL = PUBLIC_GEMINI_ENDPOINT

# Shared HTTP client with connection pooling and HTTP/2
_gemini_client: Optional[httpx.AsyncClient] = None


def _get_gemini_client() -> httpx.AsyncClient:
    """
    Return a shared httpx.AsyncClient with HTTP/2 and connection pooling.

    Reuses the same client across requests to avoid per-request TCP/TLS
    handshake overhead. The client is lazily created on first use.

    Returns:
        Shared httpx.AsyncClient instance
    """
    global _gemini_client
    if _gemini_client is None or _gemini_client.is_closed:
        _gemini_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=120,
            ),
        )
        logger.info("Created shared Gemini HTTP client (HTTP/2, pooled)")
    return _gemini_client


async def close_gemini_client() -> None:
    """
    Close the shared Gemini HTTP client gracefully.

    Should be called during application shutdown to release connections.
    """
    global _gemini_client
    if _gemini_client and not _gemini_client.is_closed:
        await _gemini_client.aclose()
        _gemini_client = None
        logger.info("Closed shared Gemini HTTP client")

# Models available via Gemini API
GEMINI_MODELS: List[Dict[str, str]] = [
    {"id": "gemini-2.5-pro",                  "display_name": "Gemini 2.5 Pro"},
    {"id": "gemini-2.5-flash",                "display_name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-flash-lite",           "display_name": "Gemini 2.5 Flash Lite"},
    {"id": "gemini-2.0-flash",                "display_name": "Gemini 2.0 Flash"},
    {"id": "gemini-3.1-pro-preview",          "display_name": "Gemini 3.1 Pro (Preview)"},
    {"id": "gemini-3.1-flash-lite-preview",   "display_name": "Gemini 3.1 Flash Lite (Preview)"},
]


def is_gemini_model(model_name: str) -> bool:
    """
    Return True if the model should be routed to the Google Gemini provider.

    Args:
        model_name: Model name from the client request

    Returns:
        True for any model name starting with "gemini-" (case-insensitive)
    """
    return model_name.lower().startswith("gemini-")


# ==================================================================================================
# Tool conversion helpers
# ==================================================================================================


def _sanitize_parameters_for_gemini(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove fields from JSON Schema that Gemini does not accept.

    Gemini's functionDeclarations use a subset of JSON Schema and reject
    unknown fields like $schema, additionalProperties, $defs, etc.

    Args:
        params: JSON Schema dict (e.g. from input_schema)

    Returns:
        Cleaned copy with unsupported fields removed (recursively)
    """
    _BLOCKED_KEYS = {
        "$schema", "$defs", "$ref", "$id", "$comment",
        "additionalProperties", "default", "examples", "const",
        "propertyNames", "patternProperties",
        "if", "then", "else",
        "allOf", "anyOf", "oneOf", "not",
        "contentMediaType", "contentEncoding",
        "deprecated", "readOnly", "writeOnly",
        "uniqueItems", "exclusiveMinimum", "exclusiveMaximum",
        "minProperties", "maxProperties",
        "title",
    }

    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if key in _BLOCKED_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_parameters_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _sanitize_parameters_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _convert_anthropic_tools_to_gemini(
    tools: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic-format tools to Gemini functionDeclarations format.

    Anthropic: {"name": "x", "description": "y", "input_schema": {...}}
    Gemini:    {"name": "x", "description": "y", "parameters": {...}}

    Args:
        tools: List of tools in Anthropic format, or None

    Returns:
        List of Gemini functionDeclaration dicts (may be empty)
    """
    if not tools:
        return []

    declarations: List[Dict[str, Any]] = []

    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("input_schema") or tool.get("parameters") or {}
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            parameters = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "parameters", None)
                or {}
            )

        if not name:
            logger.debug("Skipping Gemini tool with empty name")
            continue

        decl: Dict[str, Any] = {"name": name, "parameters": _sanitize_parameters_for_gemini(parameters)}
        if description:
            decl["description"] = description

        declarations.append(decl)

    if declarations:
        logger.debug(f"Converted {len(declarations)} tool(s) to Gemini functionDeclarations")

    return declarations


# ==================================================================================================
# Message conversion
# ==================================================================================================


def _extract_tool_name_by_id(
    messages: List[Any],
    tool_use_id: str,
) -> str:
    """
    Scan messages backwards to find the tool name for a given tool_use_id.

    Gemini's functionResponse requires the function name, but Anthropic's
    tool_result only carries the tool_use_id. We look up the matching
    tool_use block in the conversation history.

    Args:
        messages: Full list of Anthropic messages (dicts or objects)
        tool_use_id: The tool_use_id from a tool_result block

    Returns:
        Tool name string, or empty string if not found
    """
    for msg in reversed(messages):
        content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                block = block.__dict__ if hasattr(block, "__dict__") else {}
            if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                return block.get("name", "")
    return ""


def _convert_anthropic_messages_to_gemini(
    messages: List[Any],
) -> List[Dict[str, Any]]:
    """
    Convert an Anthropic messages list to Gemini contents format.

    Role mapping:
    - Anthropic "user"      → Gemini "user"
    - Anthropic "assistant" → Gemini "model"

    Content block mapping:
    - text block            → {"text": "..."}
    - tool_use block        → {"functionCall": {"name": ..., "args": ...}}
    - tool_result block     → {"functionResponse": {"name": ..., "response": {"result": ...}}}
    - string content        → [{"text": "..."}]

    Args:
        messages: List of Anthropic message dicts (or Pydantic objects)

    Returns:
        List of Gemini content dicts with "role" and "parts" keys
    """
    gemini_contents: List[Dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")

        gemini_role = "model" if role == "assistant" else "user"

        # String content — wrap in a single text part
        if not isinstance(content, list):
            text = str(content) if content is not None else ""
            if text:
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": text}],
                })
            continue

        # List of content blocks — convert each to a Gemini part
        parts: List[Dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                if hasattr(block, "__dict__"):
                    block = {
                        "type": getattr(block, "type", ""),
                        **{k: v for k, v in block.__dict__.items() if k != "type"},
                    }
                else:
                    continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})

            elif block_type == "tool_use":
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    tool_input = {}
                parts.append({
                    "functionCall": {
                        "name": block.get("name", ""),
                        "args": tool_input,
                    }
                })

            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                tool_name = _extract_tool_name_by_id(messages, tool_use_id)

                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from content blocks
                    text_parts = [
                        b.get("text", "")
                        for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    result_content = "".join(text_parts)
                elif not isinstance(result_content, str):
                    result_content = str(result_content) if result_content else ""

                parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": result_content},
                    }
                })

        if parts:
            gemini_contents.append({"role": gemini_role, "parts": parts})

    return gemini_contents


# ==================================================================================================
# Payload builder
# ==================================================================================================


def _build_gemini_payload(
    request_data: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    Translate an Anthropic Messages request dict to Gemini generateContent format.

    Args:
        request_data: Anthropic request as a plain dict (model_dump() output)
        model: Model name to send to Gemini (e.g. "gemini-2.5-pro")

    Returns:
        Gemini-format request payload dict
    """
    messages = request_data.get("messages", [])
    contents = _convert_anthropic_messages_to_gemini(messages)

    payload: Dict[str, Any] = {"contents": contents}

    # System instruction
    system = request_data.get("system")
    if system:
        if isinstance(system, list):
            system_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in system
            ).strip()
        else:
            system_text = str(system).strip()

        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    # Tools
    anthropic_tools = request_data.get("tools")
    if anthropic_tools:
        declarations = _convert_anthropic_tools_to_gemini(anthropic_tools)
        if declarations:
            payload["tools"] = [{"functionDeclarations": declarations}]

    # Generation config
    gen_config: Dict[str, Any] = {}
    if request_data.get("max_tokens") is not None:
        gen_config["maxOutputTokens"] = request_data["max_tokens"]
    if request_data.get("temperature") is not None:
        gen_config["temperature"] = request_data["temperature"]
    if request_data.get("top_p") is not None:
        gen_config["topP"] = request_data["top_p"]
    if request_data.get("stop_sequences"):
        gen_config["stopSequences"] = request_data["stop_sequences"]

    if gen_config:
        payload["generationConfig"] = gen_config

    return payload


# ==================================================================================================
# SSE event helpers (Anthropic format)
# ==================================================================================================


def _make_message_start_event(model: str, message_id: str) -> str:
    """Build the Anthropic message_start SSE event string."""
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
    """Build Anthropic content_block_start for a text block."""
    data = {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }
    return f"event: content_block_start\ndata: {json.dumps(data)}\n\n"


def _make_text_delta_event(index: int, text: str) -> str:
    """Build Anthropic content_block_delta for text."""
    data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"


def _make_tool_use_block_start_event(index: int, tool_id: str, tool_name: str) -> str:
    """Build Anthropic content_block_start for a tool_use block."""
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
    """Build Anthropic content_block_delta for tool input."""
    data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"


def _make_block_stop_event(index: int) -> str:
    """Build Anthropic content_block_stop at a given index."""
    data = {"type": "content_block_stop", "index": index}
    return f"event: content_block_stop\ndata: {json.dumps(data)}\n\n"


def _make_message_delta_event(stop_reason: str = "end_turn") -> str:
    """Build the Anthropic message_delta SSE event string."""
    data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }
    return f"event: message_delta\ndata: {json.dumps(data)}\n\n"


def _make_message_stop_event() -> str:
    """Build the Anthropic message_stop SSE event string."""
    data = {"type": "message_stop"}
    return f"event: message_stop\ndata: {json.dumps(data)}\n\n"


# ==================================================================================================
# Main streaming function
# ==================================================================================================


async def stream_gemini_response(
    request_data: Dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Stream a Gemini response translated to Anthropic SSE format.

    Fetches valid auth headers, builds the Gemini payload, opens a streaming
    POST to the Gemini SSE endpoint, and translates each SSE chunk to the
    Anthropic streaming format expected by Claude Code.

    Supports both text and tool_use (functionCall) responses.

    Args:
        request_data: Anthropic request as a plain dict (model_dump() output)
        model: Model name (e.g. "gemini-2.5-pro")

    Yields:
        Anthropic-format SSE event strings

    Raises:
        HTTPException: On authentication, rate-limit, or server errors
    """
    # Obtain auth headers (API key or OAuth2 token)
    try:
        auth_headers = await get_gemini_auth_headers()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "type": "error",
                "error": {
                    "type": "service_unavailable",
                    "message": (
                        "Gemini is not authenticated. "
                        "Set GEMINI_API_KEY or run `agy` to authenticate via Antigravity CLI."
                    ),
                },
            },
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": str(e),
                },
            },
        ) from e

    payload = _build_gemini_payload(request_data, model)

    # Route based on auth type: OAuth → Code Assist, API key → public API
    auth_type = get_auth_type()
    if auth_type == GeminiAuthType.OAUTH:
        project_id = await discover_project_id()
        url = f"{CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
        final_payload = {
            "model": model,
            "project": project_id,
            "request": payload,
        }
        logger.info(f"Sending request to Gemini Code Assist (model={model}, project={project_id})")
    else:
        url = f"{_GEMINI_BASE_URL}/models/{model}:streamGenerateContent?alt=sse"
        final_payload = payload
        logger.info(f"Sending request to Gemini public API (model={model})")

    headers = {
        **auth_headers,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    logger.debug(
        f"Gemini payload: contents={len(payload.get('contents', []))} messages, "
        f"tools={len(payload.get('tools', [{'functionDeclarations': []}])[0].get('functionDeclarations', []) if payload.get('tools') else [])}"
    )

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # State tracking for content blocks
    block_index = 0
    text_block_started = False
    has_tool_calls = False
    message_start_emitted = False

    _MAX_429_RETRIES = 5

    try:
        client = _get_gemini_client()

        # Retry loop for 429 rate limits
        for attempt in range(_MAX_429_RETRIES):
            response_cm = client.stream("POST", url, json=final_payload, headers=headers)
            response = await response_cm.__aenter__()

            if response.status_code == 429:
                body = await response.aread()
                await response_cm.__aexit__(None, None, None)

                # Extract wait time from error message (e.g. "after 11s")
                wait_secs = 15
                try:
                    body_text = body.decode("utf-8", errors="replace")
                    match = re.search(r"after\s+(\d+)s", body_text)
                    if match:
                        wait_secs = int(match.group(1)) + 1
                except Exception:
                    pass

                if attempt < _MAX_429_RETRIES - 1:
                    logger.warning(
                        f"Gemini 429 rate limit (attempt {attempt + 1}/{_MAX_429_RETRIES}), "
                        f"waiting {wait_secs}s before retry..."
                    )
                    await asyncio.sleep(wait_secs)
                    continue

                # Exhausted retries
                raise HTTPException(
                    status_code=429,
                    detail={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": (
                                "Gemini API rate limit reached after "
                                f"{_MAX_429_RETRIES} retries. Please wait before retrying."
                            ),
                        },
                    },
                )

            # Not a 429 — break out of retry loop
            break

        try:
            if response.status_code == 401:
                await response.aread()
                raise HTTPException(
                    status_code=401,
                    detail={
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": (
                                "Gemini API key is invalid or OAuth2 token expired. "
                                "Check GEMINI_API_KEY or run `agy` to re-authenticate."
                            ),
                        },
                    },
                )

            if response.status_code >= 400:
                try:
                    body = await response.aread()
                    detail_text = body.decode("utf-8", errors="replace")
                except Exception:
                    detail_text = f"HTTP {response.status_code}"

                logger.error(
                    f"Gemini API returned HTTP {response.status_code}: "
                    f"{detail_text[:200]}"
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail={
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": (
                                f"Gemini API error (HTTP {response.status_code}): "
                                f"{detail_text[:200]}"
                            ),
                        },
                    },
                )

            # Success — now emit the stream opening event
            yield _make_message_start_event(model, message_id)
            message_start_emitted = True

            # Parse Gemini SSE stream and translate to Anthropic format
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: "):]
                if raw.strip() in ("", "[DONE]"):
                    continue

                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug(f"Gemini SSE: skipping non-JSON line: {raw[:80]}")
                    continue

                # Code Assist wraps the response: {"response": {...}, "traceId": "..."}
                if "response" in chunk and "candidates" not in chunk:
                    chunk = chunk["response"]

                candidates = chunk.get("candidates", [])
                if not candidates:
                    continue

                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                for part in parts:
                    if "text" in part:
                        text = part["text"]
                        if text:
                            if not text_block_started:
                                yield _make_text_block_start_event(block_index)
                                text_block_started = True
                            yield _make_text_delta_event(block_index, text)

                    elif "functionCall" in part:
                        func_call = part["functionCall"]
                        tool_name = func_call.get("name", "")
                        tool_args = func_call.get("args", {})

                        has_tool_calls = True

                        # Close any open text block
                        if text_block_started:
                            yield _make_block_stop_event(block_index)
                            block_index += 1
                            text_block_started = False

                        tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        logger.debug(f"Gemini tool call: {tool_name} (id={tool_id})")

                        yield _make_tool_use_block_start_event(block_index, tool_id, tool_name)
                        yield _make_tool_input_delta_event(
                            block_index, json.dumps(tool_args)
                        )
                        yield _make_block_stop_event(block_index)
                        block_index += 1

                # Check finish reason
                finish_reason = candidate.get("finishReason", "")
                if finish_reason and finish_reason not in ("", "STOP"):
                    logger.debug(f"Gemini finish reason: {finish_reason}")

        finally:
            await response_cm.__aexit__(None, None, None)

    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Gemini request timed out: {e}")
        raise HTTPException(
            status_code=504,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Gemini request timed out. Please try again.",
                },
            },
        ) from e
    except httpx.RequestError as e:
        logger.error(f"Gemini network error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Could not reach Gemini API: {e}",
                },
            },
        ) from e

    if not message_start_emitted:
        return

    # Close any open text block
    if text_block_started:
        yield _make_block_stop_event(block_index)

    # If no blocks were emitted at all, emit an empty text block
    if block_index == 0 and not text_block_started and not has_tool_calls:
        yield _make_text_block_start_event(0)
        yield _make_block_stop_event(0)

    # Emit Anthropic stream-closing events
    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    yield _make_message_delta_event(stop_reason)
    yield _make_message_stop_event()
