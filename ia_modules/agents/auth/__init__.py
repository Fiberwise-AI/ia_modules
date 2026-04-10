"""Agent Auth — IDP adapter for agent key authentication.

Supports two modes:
- local: Built-in mini OIDC provider (no external IDP needed)
- oidc: External OIDC-compliant IDP (Keycloak, Auth0, Cognito, etc.)

Set AGENT_AUTH_MODE env var to select. Default: "local".
"""

import os

from .adapter import IDPAdapter, ClientCredentials
from .mini_oidc_adapter import MiniOIDCAdapter
from .keycloak_adapter import KeycloakAdapter
from .defaults import get_default_permissions, DEFAULT_A2A_PERMISSIONS


def get_adapter(db=None) -> IDPAdapter:
    """Get the configured IDP adapter.

    Args:
        db: Database provider (used by both mini OIDC and Keycloak adapters).
    """
    mode = os.getenv("AGENT_AUTH_MODE", "local")
    if mode == "oidc":
        return KeycloakAdapter(db=db)
    return MiniOIDCAdapter(db=db)


__all__ = [
    "IDPAdapter",
    "ClientCredentials",
    "MiniOIDCAdapter",
    "KeycloakAdapter",
    "get_adapter",
    "get_default_permissions",
    "DEFAULT_A2A_PERMISSIONS",
]
