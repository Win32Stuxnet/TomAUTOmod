from __future__ import annotations

import aiohttp
from fastapi import Request, HTTPException

DISCORD_API = "https://discord.com/api/v10"


class DiscordClient:

    def __init__(self, access_token: str | None = None):
        """
        Create a DiscordClient configured with an optional access token.
        
        Parameters:
            access_token (str | None): Optional OAuth2 access token to include as a Bearer token on requests; if omitted, requests will be made unauthenticated.
        """
        self._token = access_token
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> DiscordClient:
        """
        Create and attach an aiohttp ClientSession configured for the Discord API and return the client.
        
        If an access token was provided at initialization, the session will include an Authorization: Bearer <token> header. The session is stored on self._session.
        
        Returns:
            DiscordClient: The DiscordClient instance with an open aiohttp.ClientSession bound to DISCORD_API.
        """
        self._session = aiohttp.ClientSession(
            base_url=DISCORD_API,
            headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
        )
        return self

    async def __aexit__(self, *exc) -> None:
        """
        Close the client's aiohttp session when exiting the async context.
        
        If a session was created in __aenter__, it is closed; if no session exists, this is a no-op.
        """
        if self._session:
            await self._session.close()

    async def exchange_code(
        self, client_id: str, client_secret: str, code: str, redirect_uri: str
    ) -> dict:
        # Oauth2  
        """
        Exchange an OAuth2 authorization code for an access/refresh token pair from Discord.
        
        Parameters:
            client_id (str): OAuth2 client identifier.
            client_secret (str): OAuth2 client secret.
            code (str): Authorization code received from the OAuth2 authorization flow.
            redirect_uri (str): Redirect URI used in the authorization request.
        
        Returns:
            dict: JSON-decoded token response payload from Discord's `/oauth2/token` endpoint (e.g., contains `access_token`, `refresh_token`, `expires_in`, `scope`, `token_type`).
        
        Raises:
            aiohttp.ClientResponseError: If the HTTP response status indicates a failure.
        """
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
        """
        Fetch the authenticated Discord user's account object.
        
        Returns:
            dict: The user's account data as returned by the Discord API.
        
        Raises:
            aiohttp.ClientResponseError: If the HTTP response status is not successful.
        """
        resp = await self._session.get("/users/@me")
        resp.raise_for_status()
        return await resp.json()

    async def fetch_guilds(self) -> list[dict]:
        # TODO: Fetch the user's guilds from /users/@me/guilds

        """
        Fetches the authenticated user's guilds from Discord.
        
        Returns:
            list[dict]: A list of guild objects decoded from the Discord API response.
        
        Raises:
            aiohttp.ClientResponseError: If the Discord API responds with a non-success status.
        """
        ...


def get_session_user(request: Request) -> dict | None:
    """
    Retrieve the authenticated user from the request session.
    
    Returns:
        dict: The user dictionary stored under the 'user' session key.
        None: If the 'user' key is not present.
    """
    return request.session.get("user")


def require_auth(request: Request) -> dict:
    """
    Ensure the incoming request has an authenticated session user.
    
    Checks the request session for a "user" entry and returns it. If no user is found, raises an HTTP 401 error with detail "Not authenticated".
    
    Parameters:
        request (Request): HTTP request whose session will be inspected.
    
    Returns:
        dict: Authenticated user data retrieved from the session.
    """
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def has_manage_guild(permissions: int | str) -> bool:
    """
    Check whether the Manage Guild permission bit is set in a Discord permission bitfield.
    
    Parameters:
        permissions (int | str): Discord permissions bitfield as an integer or numeric string.
    
    Returns:
        `true` if the Manage Guild bit (0x20) is set, `false` otherwise.
    """
    return (int(permissions) & 0x20) == 0x20


# I am serious, I will know if you contributed using an LLM. 
# I worked as social for the Cline Project, and I can see what is AI written easily, you will 
#be banned from contributing if you use an LLM to write code for this project, I am not joking.