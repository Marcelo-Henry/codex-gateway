# -*- coding: utf-8 -*-

"""
Tests for Gemini authentication manager (kiro/gemini_auth.py).

Covers:
- API key authentication path
- OAuth2 credentials file path
- Token caching and refresh logic
- Error handling for missing/malformed credentials
- is_gemini_available() availability checks
- clear_token_cache() state reset
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ==================================================================================================
# Helpers
# ==================================================================================================


def _make_oauth_creds(
    client_id: str = "test-client-id",
    client_secret: str = "test-client-secret",
    refresh_token: str = "test-refresh-token",
    token_uri: str = "https://oauth2.googleapis.com/token",
) -> dict:
    """Return a minimal valid OAuth2 credentials dict."""
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "token_uri": token_uri,
    }


def _make_token_response(
    access_token: str = "ya29.test-access-token",
    expires_in: int = 3600,
) -> dict:
    """Return a minimal valid Google token response dict."""
    return {
        "access_token": access_token,
        "expires_in": expires_in,
        "token_type": "Bearer",
    }


# ==================================================================================================
# TestGeminiAuthApiKey
# ==================================================================================================


class TestGeminiAuthApiKey:
    """Tests for the API key authentication path."""

    @pytest.mark.asyncio
    async def test_api_key_returns_correct_header(self, tmp_path, monkeypatch):
        """GEMINI_API_KEY set → returns x-goog-api-key header with that value."""
        import kiro.gemini_auth as auth_mod

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "my-api-key-123")
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        auth_mod.clear_token_cache()

        headers = await auth_mod.get_gemini_auth_headers()

        assert headers == {"x-goog-api-key": "my-api-key-123"}

    @pytest.mark.asyncio
    async def test_api_key_takes_priority_over_oauth(self, tmp_path, monkeypatch):
        """When both API key and OAuth file are configured, API key wins."""
        import kiro.gemini_auth as auth_mod

        # Write a valid OAuth file
        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "priority-key")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        auth_mod.clear_token_cache()

        headers = await auth_mod.get_gemini_auth_headers()

        # Must use API key, not OAuth
        assert "x-goog-api-key" in headers
        assert headers["x-goog-api-key"] == "priority-key"
        assert "Authorization" not in headers

    def test_is_gemini_available_with_api_key(self, monkeypatch):
        """is_gemini_available() returns True when GEMINI_API_KEY is set."""
        import kiro.gemini_auth as auth_mod

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "some-key")
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)

        assert auth_mod.is_gemini_available() is True

    def test_is_gemini_available_false_when_disabled_with_api_key(self, monkeypatch):
        """is_gemini_available() returns False when GEMINI_ENABLED is False, even with key."""
        import kiro.gemini_auth as auth_mod

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "some-key")
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", False)

        assert auth_mod.is_gemini_available() is False


# ==================================================================================================
# TestGeminiAuthOAuth
# ==================================================================================================


class TestGeminiAuthOAuth:
    """Tests for the OAuth2 credentials file path."""

    @pytest.mark.asyncio
    async def test_oauth_reads_credentials_file(self, tmp_path, monkeypatch):
        """OAuth2 path reads credentials file and returns Authorization: Bearer header."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)
        auth_mod.clear_token_cache()

        token_resp = _make_token_response(access_token="ya29.fresh-token")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=token_resp)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("kiro.gemini_auth.httpx.AsyncClient", return_value=mock_client_cm):
            headers = await auth_mod.get_gemini_auth_headers()

        assert headers == {"Authorization": "Bearer ya29.fresh-token"}

    @pytest.mark.asyncio
    async def test_oauth_refreshes_expired_token(self, tmp_path, monkeypatch):
        """OAuth2 path calls the refresh endpoint when cached token is expired."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)

        # Seed cache with an already-expired token
        auth_mod._cached_token = "old-token"
        auth_mod._cached_expires_at = time.time() - 10  # expired 10 seconds ago

        token_resp = _make_token_response(access_token="ya29.refreshed-token")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=token_resp)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("kiro.gemini_auth.httpx.AsyncClient", return_value=mock_client_cm):
            headers = await auth_mod.get_gemini_auth_headers()

        assert headers == {"Authorization": "Bearer ya29.refreshed-token"}
        # Verify the refresh endpoint was called
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "oauth2.googleapis.com/token" in call_kwargs[0][0]

    @pytest.mark.asyncio
    async def test_oauth_caches_valid_token(self, tmp_path, monkeypatch):
        """OAuth2 path does not re-read the file when cached token is still valid."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)

        # Seed cache with a valid (non-expired) token
        auth_mod._cached_token = "ya29.cached-valid-token"
        auth_mod._cached_expires_at = time.time() + 3600  # expires in 1 hour

        mock_client = MagicMock()
        mock_client.post = AsyncMock()
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("kiro.gemini_auth.httpx.AsyncClient", return_value=mock_client_cm):
            headers = await auth_mod.get_gemini_auth_headers()

        # Should return cached token without calling the refresh endpoint
        assert headers == {"Authorization": "Bearer ya29.cached-valid-token"}
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_oauth_raises_on_missing_file(self, tmp_path, monkeypatch):
        """RuntimeError is raised when no API key is set and the OAuth2 file is absent."""
        import kiro.gemini_auth as auth_mod

        missing_path = tmp_path / "nonexistent_creds.json"

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(missing_path))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        auth_mod.clear_token_cache()

        # get_gemini_auth_headers checks file existence before attempting OAuth;
        # when the file is absent it raises RuntimeError (not configured).
        with pytest.raises(RuntimeError, match="not configured"):
            await auth_mod.get_gemini_auth_headers()

    @pytest.mark.asyncio
    async def test_oauth_raises_on_invalid_json(self, tmp_path, monkeypatch):
        """ValueError is raised when the credentials file contains malformed JSON."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "bad_creds.json"
        creds_file.write_text("{ this is not valid json }", encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)
        auth_mod.clear_token_cache()

        with pytest.raises(ValueError, match="invalid JSON"):
            await auth_mod.get_gemini_auth_headers()

    @pytest.mark.asyncio
    async def test_oauth_raises_on_missing_required_fields(self, tmp_path, monkeypatch):
        """ValueError is raised when credentials file is missing required fields."""
        import kiro.gemini_auth as auth_mod

        # File exists but is missing refresh_token
        incomplete_creds = {"client_id": "id", "client_secret": "secret"}
        creds_file = tmp_path / "incomplete_creds.json"
        creds_file.write_text(json.dumps(incomplete_creds), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)
        auth_mod.clear_token_cache()

        with pytest.raises(ValueError, match="missing 'refresh_token'"):
            await auth_mod.get_gemini_auth_headers()

    def test_is_gemini_available_with_oauth_file(self, tmp_path, monkeypatch):
        """is_gemini_available() returns True when OAuth2 credentials file exists."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)

        assert auth_mod.is_gemini_available() is True


# ==================================================================================================
# TestGeminiAuthEdgeCases
# ==================================================================================================


class TestGeminiAuthEdgeCases:
    """Edge case and state management tests."""

    def test_is_gemini_available_false_when_nothing_configured(self, tmp_path, monkeypatch):
        """is_gemini_available() returns False when no API key and no OAuth file."""
        import kiro.gemini_auth as auth_mod

        missing_path = tmp_path / "nonexistent.json"

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(missing_path))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)

        assert auth_mod.is_gemini_available() is False

    def test_clear_cache_resets_state(self, monkeypatch):
        """clear_token_cache() resets both cached token and expiry to None."""
        import kiro.gemini_auth as auth_mod

        # Seed some state
        auth_mod._cached_token = "some-token"
        auth_mod._cached_expires_at = time.time() + 3600

        auth_mod.clear_token_cache()

        assert auth_mod._cached_token is None
        assert auth_mod._cached_expires_at is None

    @pytest.mark.asyncio
    async def test_get_headers_raises_when_not_configured(self, tmp_path, monkeypatch):
        """get_gemini_auth_headers() raises RuntimeError when nothing is configured."""
        import kiro.gemini_auth as auth_mod

        missing_path = tmp_path / "nonexistent.json"

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(missing_path))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)
        auth_mod.clear_token_cache()

        with pytest.raises(RuntimeError, match="not configured"):
            await auth_mod.get_gemini_auth_headers()

    @pytest.mark.asyncio
    async def test_oauth_http_error_raises_runtime_error(self, tmp_path, monkeypatch):
        """HTTP error during token refresh is wrapped in RuntimeError."""
        import kiro.gemini_auth as auth_mod

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_oauth_creds()), encoding="utf-8")

        monkeypatch.setattr(auth_mod, "GEMINI_API_KEY", "")
        monkeypatch.setattr(auth_mod, "GEMINI_AUTH_FILE", str(creds_file))
        monkeypatch.setattr(auth_mod, "GEMINI_ENABLED", True)
        monkeypatch.setattr(auth_mod, "_is_antigravity_keyring_available", lambda: False)
        monkeypatch.setattr(auth_mod, "_read_antigravity_keyring", lambda: None)
        auth_mod.clear_token_cache()

        # Simulate a 401 from Google
        mock_http_response = MagicMock()
        mock_http_response.status_code = 401
        http_error = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_http_response,
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=http_error)
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("kiro.gemini_auth.httpx.AsyncClient", return_value=mock_client_cm):
            with pytest.raises(RuntimeError, match="token refresh failed"):
                await auth_mod.get_gemini_auth_headers()
