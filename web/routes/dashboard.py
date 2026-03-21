from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from web.auth import get_session_user, has_manage_guild

router = APIRouter(tags=["dashboard"])

TEMPLATES = Path(__file__).parent.parent / "templates"


def _render(template_name: str, **ctx) -> HTMLResponse:
    html = (TEMPLATES / template_name).read_text(encoding="utf-8")
    for key, value in ctx.items():
        html = html.replace(f"{{{{ {key} }}}}", str(value))
    return HTMLResponse(html)


@router.get("/")
async def index(request: Request) -> HTMLResponse:
    user = get_session_user(request)
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    return _render("index.html")


@router.get("/dashboard")
async def dashboard(request: Request) -> HTMLResponse:
    user = get_session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    guilds = request.session.get("guilds", [])
    manageable = [
        g for g in guilds
        if has_manage_guild(g.get("permissions", 0))
    ]

    guild_cards = ""
    for g in manageable:
        gid = escape(str(g["id"]))
        name = escape(g["name"])
        icon_hash = escape(str(g.get("icon", "")))
        icon = (
            f"https://cdn.discordapp.com/icons/{gid}/{icon_hash}.png"
            if g.get("icon")
            else "https://cdn.discordapp.com/embed/avatars/0.png"
        )
        guild_cards += f"""
        <a href="/dashboard/{gid}" class="guild-card">
            <img src="{icon}" alt="{name}">
            <span>{name}</span>
        </a>"""

    return _render(
        "dashboard.html",
        username=escape(user["username"]),
        guild_cards=guild_cards,
    )


@router.get("/dashboard/{guild_id}")
async def guild_dashboard(request: Request, guild_id: int) -> HTMLResponse:
    user = get_session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return _render("guild.html", guild_id=str(guild_id))
