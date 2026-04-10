"""KeycloakAdapter — IDP adapter for external Keycloak (or compatible) IDPs.

Used when AGENT_AUTH_MODE=oidc. Calls the Keycloak Admin REST API to
register/update clients, and the token endpoint for client_credentials.

This adapter can be used with any OIDC provider that has a compatible
admin API. For providers with different admin APIs (Auth0, Cognito),
create a new adapter implementing IDPAdapter.
"""

import json
import logging
import os

import httpx

from .adapter import IDPAdapter, ClientCredentials

logger = logging.getLogger(__name__)

_IDP_ADMIN_URL = os.getenv("IDP_ADMIN_URL", "")
_IDP_TOKEN_URL = os.getenv("IDP_TOKEN_URL", "")
_IDP_ADMIN_CLIENT_ID = os.getenv("IDP_ADMIN_CLIENT_ID", "")
_IDP_ADMIN_CLIENT_SECRET = os.getenv("IDP_ADMIN_CLIENT_SECRET", "")
_IDP_DISCOVERY_URL = os.getenv("OIDC_DISCOVERY_URL", "")
_A2A_AUDIENCE = os.getenv("A2A_AUDIENCE", "a2a-server")


class KeycloakAdapter(IDPAdapter):
    """External Keycloak IDP adapter via Admin REST API."""

    def __init__(self, db=None):
        if not _IDP_ADMIN_URL or not _IDP_TOKEN_URL:
            raise ValueError(
                "AGENT_AUTH_MODE=oidc requires IDP_ADMIN_URL and IDP_TOKEN_URL env vars"
            )
        self.db = db

    async def _get_admin_token(self) -> str:
        """Get an admin access token for Keycloak Admin REST API."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(
                _IDP_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": _IDP_ADMIN_CLIENT_ID,
                    "client_secret": _IDP_ADMIN_CLIENT_SECRET,
                },
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def _find_client(self, admin_token: str, client_id: str) -> dict:
        """Find a Keycloak client by clientId, return the full client representation."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{_IDP_ADMIN_URL}/clients",
                params={"clientId": client_id},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()
            clients = resp.json()
            if not clients:
                raise ValueError(f"Keycloak client not found: {client_id}")
            return clients[0]

    async def _get_client_secret(self, admin_token: str, kc_id: str) -> str:
        """Retrieve the client secret from Keycloak Admin API."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{_IDP_ADMIN_URL}/clients/{kc_id}/client-secret",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()
            return resp.json()["value"]

    def _build_a2a_protocol_mappers(self, agent_id: str, permissions: dict, org_id: int) -> list:
        """Build Keycloak protocol mappers that inject a2a claims into JWTs."""
        return [
            {
                "name": "a2a-permissions",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-hardcoded-claim-mapper",
                "config": {
                    "claim.name": "a2a",
                    "claim.value": json.dumps(permissions),
                    "jsonType.label": "JSON",
                    "id.token.claim": "false",
                    "access.token.claim": "true",
                    "userinfo.token.claim": "false",
                },
            },
            {
                "name": "a2a-org-id",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-hardcoded-claim-mapper",
                "config": {
                    "claim.name": "org_id",
                    "claim.value": str(org_id),
                    "jsonType.label": "int",
                    "id.token.claim": "false",
                    "access.token.claim": "true",
                    "userinfo.token.claim": "false",
                },
            },
            {
                "name": "a2a-agent-id",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-hardcoded-claim-mapper",
                "config": {
                    "claim.name": "agent_id",
                    "claim.value": str(agent_id),
                    "jsonType.label": "String",
                    "id.token.claim": "false",
                    "access.token.claim": "true",
                    "userinfo.token.claim": "false",
                },
            },
        ]

    async def register_client(
        self,
        agent_id: str,
        org_id: int,
        permissions: dict,
    ) -> ClientCredentials:
        """Register a new Keycloak client for this agent."""
        admin_token = await self._get_admin_token()
        client_id = f"agent-{agent_id}"

        payload = {
            "clientId": client_id,
            "enabled": True,
            "publicClient": False,
            "serviceAccountsEnabled": True,
            "clientAuthenticatorType": "client-secret",
            "protocol": "openid-connect",
            "attributes": {
                "a2a.permissions": json.dumps(permissions),
                "a2a.org_id": str(org_id),
                "a2a.agent_id": str(agent_id),
            },
            "protocolMappers": self._build_a2a_protocol_mappers(agent_id, permissions, org_id),
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(
                f"{_IDP_ADMIN_URL}/clients",
                json=payload,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()

            location = resp.headers.get("Location", "")
            if location:
                secret_url = f"{location}/client-secret"
            else:
                kc_client = await self._find_client(admin_token, client_id)
                secret_url = f"{_IDP_ADMIN_URL}/clients/{kc_client['id']}/client-secret"

            secret_resp = await client.post(
                secret_url,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            secret_resp.raise_for_status()
            client_secret = secret_resp.json()["value"]

        logger.info("Registered Keycloak client: %s for agent %s", client_id, agent_id)
        return ClientCredentials(client_id=client_id, client_secret=client_secret)

    async def update_client_claims(self, client_id: str, permissions: dict):
        """Update the a2a attributes and protocol mappers on an existing Keycloak client."""
        admin_token = await self._get_admin_token()
        kc_client = await self._find_client(admin_token, client_id)
        kc_id = kc_client["id"]

        kc_client.setdefault("attributes", {})
        kc_client["attributes"]["a2a.permissions"] = json.dumps(permissions)

        for mapper in kc_client.get("protocolMappers", []):
            if mapper.get("name") == "a2a-permissions":
                mapper["config"]["claim.value"] = json.dumps(permissions)

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.put(
                f"{_IDP_ADMIN_URL}/clients/{kc_id}",
                json=kc_client,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()

        logger.info("Updated Keycloak client claims: %s", client_id)

    async def delete_client(self, client_id: str):
        """Delete a Keycloak client."""
        admin_token = await self._get_admin_token()

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            list_resp = await client.get(
                f"{_IDP_ADMIN_URL}/clients",
                params={"clientId": client_id},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            list_resp.raise_for_status()
            clients = list_resp.json()
            if not clients:
                logger.warning("Keycloak client not found for deletion: %s", client_id)
                return

            kc_id = clients[0]["id"]
            resp = await client.delete(
                f"{_IDP_ADMIN_URL}/clients/{kc_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            resp.raise_for_status()

        logger.info("Deleted Keycloak client: %s", client_id)

    async def get_token(
        self,
        client_id: str,
        client_secret: str,
        audience: str = "",
    ) -> str:
        """Request a client_credentials JWT from Keycloak."""
        audience = audience or _A2A_AUDIENCE

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(
                _IDP_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "audience": audience,
                },
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    async def issue_token_for_agent(self, agent_id: str, audience: str = "") -> str:
        """Issue a JWT for an agent — looks up client_id from DB, gets secret from Keycloak."""
        audience = audience or _A2A_AUDIENCE

        if not self.db:
            raise RuntimeError("KeycloakAdapter requires db for issue_token_for_agent")

        row = await self.db.fetch_one(
            """SELECT idp_client_id
               FROM agent_api_keys
               WHERE agent_id = :agent_id
                 AND idp_client_id IS NOT NULL
                 AND is_active = true
               LIMIT 1""",
            {"agent_id": str(agent_id)},
        )
        if not row:
            raise ValueError(f"No IDP-registered key found for agent {agent_id}")

        idp_client_id = dict(row)["idp_client_id"]

        admin_token = await self._get_admin_token()
        kc_client = await self._find_client(admin_token, idp_client_id)
        client_secret = await self._get_client_secret(admin_token, kc_client["id"])

        return await self.get_token(idp_client_id, client_secret, audience)

    async def validate_token(self, token: str, audience: str = "") -> dict:
        """Validate a JWT by fetching JWKS from the IDP over HTTP."""
        from jose import jwt as jose_jwt, JWTError

        audience = audience or _A2A_AUDIENCE

        discovery_url = self.get_discovery_url()
        if not discovery_url:
            raise ValueError("OIDC_DISCOVERY_URL not configured for token validation")

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            disc_resp = await client.get(f"{discovery_url}/.well-known/openid-configuration")
            disc_resp.raise_for_status()
            jwks_uri = disc_resp.json()["jwks_uri"]

            jwks_resp = await client.get(jwks_uri)
            jwks_resp.raise_for_status()
            jwks_data = jwks_resp.json()

        try:
            claims = jose_jwt.decode(
                token,
                jwks_data,
                algorithms=["RS256"],
                audience=audience,
            )
        except JWTError as e:
            raise ValueError(f"Token validation failed: {e}")

        return claims

    def get_discovery_url(self) -> str:
        return _IDP_DISCOVERY_URL
