# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python main.py                    # Start server (default: 0.0.0.0:8000)
python main.py --port 9000        # Override port via CLI
pytest                            # Run all tests
pytest tests/unit/test_auth.py    # Run a single test file
pytest -k "test_name"             # Run a specific test by name
python manual_api_test.py         # Manual integration test (requires running gateway)
```

## Architecture

FastAPI proxy gateway that translates OpenAI, Anthropic, and Gemini API formats into Kiro (Amazon Q Developer) API calls. Also routes to Codex (OpenAI backend), Gemini (Google), and OpenRouter (multi-model aggregator). Clients (Claude Code, Cursor, Cline) point their `baseURL` here and use a Kiro subscription — or OpenRouter credits — instead of paying per-token.

### Request Flow

1. Client sends request in OpenAI/Anthropic/Gemini format
2. Route handler (`kiro/routes_*.py`) validates auth via `PROXY_API_KEY`
3. `AccountManager` selects a healthy account (circuit breaker + failover)
4. Converter (`kiro/converters_*.py`) transforms payload to Kiro format
5. `KiroHttpClient` sends request to `q.{region}.amazonaws.com`
6. Streaming layer (`kiro/streaming_*.py`) parses Kiro SSE events back to client format
7. `ThinkingParser` extracts `<thinking>` blocks into native reasoning_content fields

### Key Modules

- `kiro/config.py` — All settings loaded from env vars. Single source of truth for configuration.
- `kiro/account_manager.py` — Multi-account failover with circuit breaker (exponential backoff, probabilistic retry).
- `kiro/auth.py` — Token lifecycle: loads credentials from SQLite (kiro-cli), JSON (Kiro IDE), or raw refresh token. Handles both Kiro Desktop Auth and AWS SSO OIDC.
- `kiro/model_resolver.py` — 4-layer resolution: normalize name → check dynamic cache → check hidden models → pass-through to Kiro.
- `kiro/converters_core.py` — Shared conversion logic (tool descriptions, message normalization).
- `kiro/streaming_core.py` — Unified Kiro SSE parsing with first-token timeout and retry.
- `kiro/thinking_parser.py` — Extracts `<thinking>` tags from streamed responses into structured reasoning blocks.
- `kiro/parsers.py` — AWS event stream parser and bracket tool call extraction.
- `kiro/codex_provider.py` — Routes `gpt-*`/`codex-*` models to OpenAI's Codex CLI backend.
- `kiro/gemini_provider.py` — Routes Gemini models to Google's API.
- `kiro/openrouter_provider.py` — Routes `provider/model` format models to OpenRouter API. Handles Anthropic↔OpenAI conversion, streaming in both formats, tool calling.
- `kiro/truncation_recovery.py` — Injects synthetic messages when Kiro truncates tool calls.
- `kiro/payload_guards.py` — Enforces Kiro's ~615KB payload limit, optional auto-trim.

### Streaming Architecture

The streaming layer is split into three tiers:
- `streaming_core.py` — Kiro SSE parsing, first-token timeout, `KiroEvent` dataclass
- `streaming_openai.py` — Formats KiroEvents as OpenAI SSE chunks
- `streaming_anthropic.py` — Formats KiroEvents as Anthropic SSE events

### Provider Routing

Requests are routed by model name (checked in this order):
- `provider/model` format (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4`) → OpenRouter provider
- `gpt-*` / `codex-*` → Codex provider (OpenAI backend)
- `gemini-*` → Gemini provider (Google API)
- Everything else → Kiro API (Amazon Q)

### OpenRouter Provider

Routes hundreds of models via a single API key. Auth is just `Authorization: Bearer {OPENROUTER_API_KEY}`.

- Model detection: any model with `/` where the prefix is a known provider (openai, anthropic, meta-llama, google, deepseek, etc.)
- OpenAI clients (`/v1/chat/completions`): near pass-through — request forwarded as-is, SSE relayed back
- Anthropic clients (`/v1/messages`): converts Anthropic request → OpenAI format, sends to OpenRouter, converts OpenAI SSE response → Anthropic SSE events
- Full tool calling support in both directions
- Config: `OPENROUTER_API_KEY`, `OPENROUTER_ENABLED`, `OPENROUTER_BASE_URL`

### Authentication Hierarchy

Credential sources (priority order): SQLite DB > JSON file > environment variable. The `AccountManager` wraps multiple `KiroAuthManager` instances for failover.

## Testing

Unit tests in `tests/unit/` are fully isolated — no network calls. They mock HTTP responses. The `manual_api_test.py` script hits the live gateway and is excluded from pytest.

## Conventions

- Python 3.10+, type hints throughout
- Loguru for logging (not stdlib `logging`)
- httpx for async HTTP (not aiohttp/requests)
- Environment variables loaded in `kiro/config.py` only — never call `os.getenv` elsewhere
- Model names use dot notation internally (e.g., `claude-sonnet-4.5`), dashes are normalized on input


### Full SDD documentation

Detailed specs, flowcharts, state machines, C4 diagrams, and traceability matrices are in `_reversa_sdd/`:

| Path                                                                   | Contents                                                                                                                                                  |
| ------------------------------------------------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `_reversa_sdd/architecture.md`                                         | Layered architecture overview and all external integrations                                                                                               |
| `_reversa_sdd/sdd/`                                                    | Component SDDs for auth, account-manager, converters-core, streaming-core, parsers, http-client, routes, model-resolver, thinking-parser, mcp-codex       |
| `_reversa_sdd/adrs/`                                                   | 7 Architecture Decision Records                                                                                                                           |
| `_reversa_sdd/flowcharts/`                                             | Auth flow, full request flow, converter pipeline                                                                                                          |
| `_reversa_sdd/state-machines.md`                                       | FSMs for token lifecycle, Circuit Breaker, ThinkingParser, HTTP retry, streaming, account lazy-init                                                       |
| `_reversa_sdd/c4-context.md` / `c4-containers.md` / `c4-components.md` | C4 diagrams (levels 1–3)                                                                                                                                  |
| `_reversa_sdd/data-dictionary.md`                                      | All data structures: credentials.json, state.json, KiroPayload, UnifiedMessage, KiroEvent, MCP request/response                                           |
| `_reversa_sdd/domain.md`                                               | Domain glossary and 20 business rules                                                                                                                     |
| `_reversa_sdd/permissions.md`                                          | Auth model: client→gateway (PROXY_API_KEY) and gateway→Kiro (access token)                                                                                |
| `_reversa_sdd/traceability/`                                           | Code-spec matrix and spec impact matrix (high-risk components: config.py, converters_core.py, streaming_core.py, parsers.py, auth.py, account_manager.py) |
| `_reversa_sdd/user-stories/`                                           | 10 user stories across 4 epics                                                                                                                            |
| `_reversa_sdd/questions.md`                                            | 6 open questions for validation (FAKE_REASONING default, WEB_SEARCH default, ACCOUNT_SYSTEM default, MCP endpoint, Codex Provider status)                 |
