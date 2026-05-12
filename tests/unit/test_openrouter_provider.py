"""Unit tests for the OpenRouter provider."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from kiro.openrouter_provider import (
    is_openrouter_model,
    get_openrouter_models,
    _convert_anthropic_tools_to_openai,
    _convert_anthropic_messages_to_openai,
    _build_openai_payload,
    stream_openrouter_anthropic,
    stream_openrouter_openai,
)


# ==================================================================================================
# Model detection tests
# ==================================================================================================


class TestIsOpenrouterModel:
    """Tests for is_openrouter_model() routing predicate."""

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_known_prefixes_return_true(self):
        assert is_openrouter_model("openai/gpt-4o") is True
        assert is_openrouter_model("anthropic/claude-sonnet-4") is True
        assert is_openrouter_model("meta-llama/llama-3.1-405b") is True
        assert is_openrouter_model("google/gemini-2.5-pro") is True
        assert is_openrouter_model("deepseek/deepseek-r1") is True
        assert is_openrouter_model("openrouter/auto") is True
        assert is_openrouter_model("mistralai/mistral-large") is True
        assert is_openrouter_model("x-ai/grok-2") is True

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_non_openrouter_models_return_false(self):
        assert is_openrouter_model("claude-sonnet-4") is False
        assert is_openrouter_model("gpt-5.4") is False
        assert is_openrouter_model("gemini-2.5-pro") is False
        assert is_openrouter_model("codex-mini-latest") is False

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_unknown_prefix_returns_false(self):
        assert is_openrouter_model("unknownprovider/some-model") is False

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", False)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_disabled_returns_false(self):
        assert is_openrouter_model("openai/gpt-4o") is False

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "")
    def test_no_api_key_returns_false(self):
        assert is_openrouter_model("openai/gpt-4o") is False


class TestGetOpenrouterModels:
    """Tests for get_openrouter_models()."""

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_returns_models_when_enabled(self):
        models = get_openrouter_models()
        assert len(models) > 0
        assert all("id" in m and "display_name" in m for m in models)

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", False)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "sk-or-test-key")
    def test_returns_empty_when_disabled(self):
        assert get_openrouter_models() == []

    @patch("kiro.openrouter_provider.OPENROUTER_ENABLED", True)
    @patch("kiro.openrouter_provider.OPENROUTER_API_KEY", "")
    def test_returns_empty_when_no_key(self):
        assert get_openrouter_models() == []


# ==================================================================================================
# Conversion tests
# ==================================================================================================


class TestConvertAnthropicToolsToOpenai:
    """Tests for Anthropic → OpenAI tool format conversion."""

    def test_basic_tool_conversion(self):
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                },
            }
        ]
        result = _convert_anthropic_tools_to_openai(anthropic_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather"
        assert result[0]["function"]["parameters"]["properties"]["location"]["type"] == "string"

    def test_multiple_tools(self):
        tools = [
            {"name": "tool_a", "description": "A", "input_schema": {}},
            {"name": "tool_b", "description": "B", "input_schema": {}},
        ]
        result = _convert_anthropic_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_empty_tools(self):
        assert _convert_anthropic_tools_to_openai([]) == []


class TestConvertAnthropicMessagesToOpenai:
    """Tests for Anthropic → OpenAI message format conversion."""

    def test_simple_text_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _convert_anthropic_messages_to_openai(messages)
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_system_prompt_added(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_anthropic_messages_to_openai(messages, system="You are helpful.")
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_assistant_with_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    },
                ],
            }
        ]
        result = _convert_anthropic_messages_to_openai(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"location": "NYC"}

    def test_user_with_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Sunny, 72°F",
                    },
                ],
            }
        ]
        result = _convert_anthropic_messages_to_openai(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "Sunny, 72°F"

    def test_user_with_tool_result_and_text(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Result here",
                    },
                    {"type": "text", "text": "What does this mean?"},
                ],
            }
        ]
        result = _convert_anthropic_messages_to_openai(messages)
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Result here"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What does this mean?"

    def test_tool_result_with_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_456",
                        "content": [
                            {"type": "text", "text": "Part 1. "},
                            {"type": "text", "text": "Part 2."},
                        ],
                    },
                ],
            }
        ]
        result = _convert_anthropic_messages_to_openai(messages)
        assert result[0]["content"] == "Part 1. Part 2."


class TestBuildOpenaiPayload:
    """Tests for _build_openai_payload()."""

    def test_basic_payload(self):
        request_data = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "system": "Be helpful.",
        }
        result = _build_openai_payload(request_data, "openai/gpt-4o")
        assert result["model"] == "openai/gpt-4o"
        assert result["stream"] is True
        assert result["max_tokens"] == 1024
        assert result["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hello"}

    def test_payload_with_tools(self):
        request_data = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}},
            ],
        }
        result = _build_openai_payload(request_data, "openai/gpt-4o")
        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_payload_with_system_as_list(self):
        request_data = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [{"type": "text", "text": "System prompt here."}],
        }
        result = _build_openai_payload(request_data, "openai/gpt-4o")
        assert result["messages"][0]["content"] == "System prompt here."

    def test_optional_params_omitted_when_none(self):
        request_data = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _build_openai_payload(request_data, "openai/gpt-4o")
        assert "max_tokens" not in result
        assert "temperature" not in result
        assert "top_p" not in result
        assert "stop" not in result
        assert "tools" not in result


# ==================================================================================================
# Streaming tests
# ==================================================================================================


class TestStreamOpenrouterAnthropic:
    """Tests for stream_openrouter_anthropic()."""

    @pytest.mark.asyncio
    @patch("kiro.openrouter_provider._get_openrouter_client")
    async def test_text_streaming(self, mock_get_client):
        """Test basic text streaming converts to Anthropic format."""
        sse_lines = [
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}',
            'data: {"choices":[{"delta":{"content":" world"},"index":0}]}',
            'data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_get_client.return_value = mock_client

        request_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }

        events = []
        async for chunk in stream_openrouter_anthropic(request_data, "openai/gpt-4o"):
            events.append(chunk)

        event_types = [_parse_event_type(e) for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        text_deltas = [e for e in events if "text_delta" in e]
        assert any("Hello" in d for d in text_deltas)
        assert any(" world" in d for d in text_deltas)

    @pytest.mark.asyncio
    @patch("kiro.openrouter_provider._get_openrouter_client")
    async def test_tool_call_streaming(self, mock_get_client):
        """Test tool call streaming converts to Anthropic format."""
        sse_lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":""}}]},"index":0}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\":\\"NYC\\"}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_get_client.return_value = mock_client

        request_data = {
            "messages": [{"role": "user", "content": "Weather in NYC?"}],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {}}],
            "max_tokens": 100,
        }

        events = []
        async for chunk in stream_openrouter_anthropic(request_data, "openai/gpt-4o"):
            events.append(chunk)

        event_types = [_parse_event_type(e) for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types

        # Check tool_use block
        tool_start = [e for e in events if "tool_use" in e and "content_block_start" in e]
        assert len(tool_start) == 1
        assert "get_weather" in tool_start[0]
        assert "call_abc" in tool_start[0]

        # Check stop_reason is tool_use
        delta_events = [e for e in events if "message_delta" in e]
        assert any("tool_use" in d for d in delta_events)

    @pytest.mark.asyncio
    @patch("kiro.openrouter_provider._get_openrouter_client")
    async def test_error_401(self, mock_get_client):
        """Test 401 error raises HTTPException."""
        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"Invalid key"}}')

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_get_client.return_value = mock_client

        from fastapi import HTTPException

        request_data = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100}

        with pytest.raises(HTTPException) as exc_info:
            async for _ in stream_openrouter_anthropic(request_data, "openai/gpt-4o"):
                pass
        assert exc_info.value.status_code == 401


class TestStreamOpenrouterOpenai:
    """Tests for stream_openrouter_openai()."""

    @pytest.mark.asyncio
    @patch("kiro.openrouter_provider._get_openrouter_client")
    async def test_passthrough_streaming(self, mock_get_client):
        """Test OpenAI format is passed through."""
        sse_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"},"index":0}]}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: _async_iter(sse_lines)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_cm)
        mock_get_client.return_value = mock_client

        request_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "openai/gpt-4o",
            "stream": True,
        }

        chunks = []
        async for chunk in stream_openrouter_openai(request_data, "openai/gpt-4o"):
            chunks.append(chunk)

        assert any("Hi" in c for c in chunks)
        assert "data: [DONE]\n\n" in chunks


# ==================================================================================================
# Helpers
# ==================================================================================================


async def _async_iter(items):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


def _parse_event_type(event_str: str) -> str:
    """Extract the event type from an SSE event string."""
    for line in event_str.splitlines():
        if line.startswith("event: "):
            return line[7:]
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                return data.get("type", "")
            except json.JSONDecodeError:
                pass
    return ""
