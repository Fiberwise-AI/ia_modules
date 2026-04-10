"""Tests for ia_modules.agents.auth — adapters, provider, defaults."""

import os
import pytest
from unittest.mock import patch

from ia_modules.agents.auth import (
    IDPAdapter,
    ClientCredentials,
    MiniOIDCAdapter,
    KeycloakAdapter,
    get_adapter,
    get_default_permissions,
    DEFAULT_A2A_PERMISSIONS,
)
from ia_modules.agents.auth.adapter import IDPAdapter as AdapterABC
from ia_modules.agents.auth.defaults import DEFAULT_A2A_PERMISSIONS as DEFAULTS


# ---------------------------------------------------------------------------
# defaults.py
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_permissions_keys(self):
        assert set(DEFAULT_A2A_PERMISSIONS.keys()) == {"llm", "processor", "custom"}

    def test_get_default_permissions_known_type(self):
        perms = get_default_permissions("processor")
        assert "allowed_modes" in perms
        assert "research" in perms["allowed_modes"]

    def test_get_default_permissions_unknown_falls_back_to_llm(self):
        perms = get_default_permissions("unknown_type")
        llm = get_default_permissions("llm")
        assert perms == llm

    def test_get_default_permissions_returns_deep_copy(self):
        p1 = get_default_permissions("llm")
        p2 = get_default_permissions("llm")
        p1["allowed_modes"].append("execute")
        assert "execute" not in p2["allowed_modes"]

    def test_custom_has_execute_mode(self):
        perms = get_default_permissions("custom")
        assert "execute" in perms["allowed_modes"]

    def test_custom_has_bash_tool(self):
        perms = get_default_permissions("custom")
        assert "Bash" in perms["allowed_tools"]

    def test_llm_has_limits(self):
        perms = get_default_permissions("llm")
        assert perms["limits"]["max_turns"] == 30
        assert perms["limits"]["max_duration_seconds"] == 300


# ---------------------------------------------------------------------------
# adapter.py — ABC contract
# ---------------------------------------------------------------------------


class TestAdapterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            IDPAdapter()

    def test_client_credentials_dataclass(self):
        creds = ClientCredentials(client_id="cid", client_secret="csec")
        assert creds.client_id == "cid"
        assert creds.client_secret == "csec"


# ---------------------------------------------------------------------------
# get_adapter() factory
# ---------------------------------------------------------------------------


class TestGetAdapter:
    def test_default_returns_mini_oidc(self):
        with patch.dict(os.environ, {"AGENT_AUTH_MODE": "local"}, clear=False):
            import importlib
            import ia_modules.agents.auth as auth_pkg
            importlib.reload(auth_pkg)
            adapter = auth_pkg.get_adapter()
            assert isinstance(adapter, MiniOIDCAdapter)

    def test_oidc_mode_returns_keycloak(self):
        env = {
            "AGENT_AUTH_MODE": "oidc",
            "IDP_ADMIN_URL": "http://keycloak:8080/admin",
            "IDP_TOKEN_URL": "http://keycloak:8080/token",
        }
        with patch.dict(os.environ, env, clear=False):
            import importlib
            import ia_modules.agents.auth.keycloak_adapter as kc_mod
            importlib.reload(kc_mod)
            import ia_modules.agents.auth as auth_pkg
            importlib.reload(auth_pkg)
            adapter = auth_pkg.get_adapter()
            assert type(adapter).__name__ == "KeycloakAdapter"


# ---------------------------------------------------------------------------
# MiniOIDCProvider — token signing + JWKS
# ---------------------------------------------------------------------------


class TestMiniOIDCProvider:
    @pytest.fixture
    def provider(self, tmp_path):
        with patch.dict(os.environ, {"MINI_OIDC_KEY_DIR": str(tmp_path)}, clear=False):
            from ia_modules.agents.auth.provider import MiniOIDCProvider
            return MiniOIDCProvider(issuer="http://localhost:5555/oidc")

    def test_discovery(self, provider):
        disc = provider.get_discovery()
        assert disc["issuer"] == "http://localhost:5555/oidc"
        assert "jwks_uri" in disc
        assert disc["jwks_uri"].endswith("/jwks")

    def test_jwks_has_keys(self, provider):
        jwks = provider.get_jwks()
        assert "keys" in jwks
        assert len(jwks["keys"]) == 1
        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["alg"] == "RS256"
        assert "n" in key
        assert "e" in key

    def test_issue_token_returns_jwt(self, provider):
        result = provider.issue_token(
            subject="agent_123",
            audience="a2a-server",
            scopes=["a2a:execute"],
            claims={"org_id": 1},
        )
        assert "access_token" in result
        assert result["token_type"] == "Bearer"
        assert result["expires_in"] > 0

    def test_token_verifies_with_public_key(self, provider):
        from jose import jwt as jose_jwt

        result = provider.issue_token(
            subject="agent_123",
            audience="a2a-server",
            scopes=["a2a:execute"],
            claims={"org_id": 1, "agent_id": "123"},
        )

        from cryptography.hazmat.primitives import serialization
        pub_pem = provider._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        decoded = jose_jwt.decode(
            result["access_token"],
            pub_pem,
            algorithms=["RS256"],
            audience="a2a-server",
        )
        assert decoded["sub"] == "agent_123"
        assert decoded["org_id"] == 1
        assert decoded["agent_id"] == "123"

    def test_keys_persist_across_instances(self, tmp_path):
        with patch.dict(os.environ, {"MINI_OIDC_KEY_DIR": str(tmp_path)}, clear=False):
            from ia_modules.agents.auth.provider import MiniOIDCProvider
            p1 = MiniOIDCProvider(issuer="http://localhost/oidc")
            p2 = MiniOIDCProvider(issuer="http://localhost/oidc")
            assert p1._kid == p2._kid


# ---------------------------------------------------------------------------
# MiniOIDCAdapter — no-DB path (unit-testable parts)
# ---------------------------------------------------------------------------


class TestMiniOIDCAdapterNoDB:
    def test_instantiates_without_db(self):
        adapter = MiniOIDCAdapter(db=None)
        assert adapter.db is None

    def test_get_discovery_url(self):
        with patch.dict(os.environ, {
            "MINI_OIDC_ISSUER": "http://test:5555/oidc",
            "MINI_OIDC_KEY_DIR": "/tmp/test-keys",
        }, clear=False):
            adapter = MiniOIDCAdapter(db=None)
            url = adapter.get_discovery_url()
            assert url == "http://test:5555/oidc"

    @pytest.mark.asyncio
    async def test_register_client_without_db(self):
        adapter = MiniOIDCAdapter(db=None)
        creds = await adapter.register_client("agent-1", org_id=1, permissions={"allowed_modes": ["research"]})
        assert creds.client_id.startswith("agent-agent-1-")
        assert len(creds.client_secret) > 20

    @pytest.mark.asyncio
    async def test_get_token_without_db_raises(self):
        adapter = MiniOIDCAdapter(db=None)
        with pytest.raises(ValueError, match="Invalid client credentials"):
            await adapter.get_token("fake-id", "fake-secret")


# ---------------------------------------------------------------------------
# KeycloakAdapter — env validation
# ---------------------------------------------------------------------------


class TestKeycloakAdapterInit:
    def test_missing_env_raises(self):
        with patch.dict(os.environ, {"IDP_ADMIN_URL": "", "IDP_TOKEN_URL": ""}, clear=False):
            import importlib
            import ia_modules.agents.auth.keycloak_adapter as kc_mod
            importlib.reload(kc_mod)
            with pytest.raises(ValueError, match="IDP_ADMIN_URL"):
                kc_mod.KeycloakAdapter()

    def test_with_env_succeeds(self):
        env = {
            "IDP_ADMIN_URL": "http://keycloak:8080/admin",
            "IDP_TOKEN_URL": "http://keycloak:8080/token",
        }
        with patch.dict(os.environ, env, clear=False):
            import importlib
            import ia_modules.agents.auth.keycloak_adapter as kc_mod
            importlib.reload(kc_mod)
            adapter = kc_mod.KeycloakAdapter()
            assert adapter.db is None


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_all_exports_importable(self):
        from ia_modules.agents.auth import __all__
        for name in __all__:
            assert hasattr(__import__("ia_modules.agents.auth", fromlist=[name]), name)

    def test_permissions_importable_from_agents(self):
        from ia_modules.agents import enforce_agent_claims, ClaimsViolation
        assert callable(enforce_agent_claims)
        assert issubclass(ClaimsViolation, ValueError)
