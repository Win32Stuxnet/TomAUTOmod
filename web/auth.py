from __future__ import annotations

import aiohttp
from fastapi import Request, HTTPException

DISCORD_API = "https://discord.com/api/v10"


class DiscordClient:

    def __init__(self, access_token: str | None = None):
        self._token = access_token
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> DiscordClient:
        self._session = aiohttp.ClientSession(
            base_url=DISCORD_API,
            headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()

    async def exchange_code(
        self, client_id: str, client_secret: str, code: str, redirect_uri: str
    ) -> dict:
        # Oauth2  
        resp = await self._session.post(
            "/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        return await resp.json()

    async def fetch_user(self) -> dict:
        #The function name says itself, I am done writing comments that just help people dump slop into PR's from AI's. 
        resp = await self._session.get("/users/@me")
        resp.raise_for_status()
        return await resp.json()

    async def fetch_guilds(self) -> list[dict]:
        # TODO: Fetch the user's guilds from /users/@me/guilds

        ...


def get_session_user(request: Request) -> dict | None:
    return request.session.get("user")


def require_auth(request: Request) -> dict:
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def has_manage_guild(permissions: int | str) -> bool:
    return (int(permissions) & 0x20) == 0x20


# I am serious, I will know if you contributed using an LLM. 
# I worked as social for the Cline Project, and I can see what is AI written easily, you will 
#be banned from contributing if you use an LLM to write code for this project, I am not joking.