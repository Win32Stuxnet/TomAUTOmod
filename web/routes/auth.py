from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from web.auth import DiscordClient

log = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    settings = request.app.state.settings
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state

    params = urlencode({
        "client_id": settings.discord_client_id,
        "redirect_uri": f"{settings.web_base_url}/callback",
        "response_type": "code",
        "scope": "identify guilds",
        "state": state,
    })
    return RedirectResponse(
        f"https://discord.com/api/oauth2/authorize?{params}",
        status_code=302,
    )


@router.get("/callback")
async def callback(request: Request, code: str, state: str) -> RedirectResponse:
    expected = request.session.pop("oauth_state", None)
    log.info("OAuth callback: expected=%s, got=%s", expected, state)
    if not expected or state != expected:
        log.warning("OAuth state mismatch — session may not have persisted.")
        return RedirectResponse("/login", status_code=302)

    settings = request.app.state.settings
    async with DiscordClient() as auth_client:
        tokens = await auth_client.exchange_code(
            settings.discord_client_id,
            settings.discord_client_secret,
            code,
            f"{settings.web_base_url}/callback",
        )
    async with DiscordClient(tokens["access_token"]) as client:
        user = await client.fetch_user()
        guilds = await client.fetch_guilds()

    request.session["user"] = {
        "id": user["id"],
        "username": user["username"],
        "avatar": user.get("avatar"),
    }
    # Only store minimal guild data to stay within cookie size limits (~4KB)
    request.session["guilds"] = [
        {"id": g["id"], "name": g["name"], "icon": g.get("icon"), "permissions": g.get("permissions", 0)}
        for g in guilds
    ]
    log.info("OAuth complete for %s, %d guilds stored.", user["username"], len(guilds))
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/", status_code=302)
