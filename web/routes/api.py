from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, field_validator, Field

from web.auth import require_guild_permission
from bot.models.guild_config import GuildConfig
from bot.models.filter_rules import FilterRule
from bot.models.custom_command import CustomCommand
from bot.services.config_service import ConfigService
from bot.services.filter_service import FilterService
from bot.services.custom_command_service import CustomCommandService

router = APIRouter(tags=["api"])


# --- Pydantic schemas ---

class ConfigUpdate(BaseModel):
    mod_log_channel_id: int | None = None
    audit_log_channel_id: int | None = None
    welcome_channel_id: int | None = None
    welcome_message: str | None = None
    antispam_enabled: bool | None = None
    antispam_max_messages: int | None = None
    antispam_interval_seconds: int | None = None
    antispam_action: str | None = None
    antispam_duration_seconds: int | None = None
    raid_protection_enabled: bool | None = None
    raid_join_threshold: int | None = None
    raid_join_interval_seconds: int | None = None
    ml_consent: bool | None = None
    log_retention_days: int | None = None
    review_channel_id: int | None = None
    aggression_channel_id: int | None = None
    aggression_strike_count: int | None = None
    aggression_window_hours: int | None = None


class FilterRuleCreate(BaseModel):
    rule_type: str
    pattern: str
    action: str = "delete"


class CustomCommandCreate(BaseModel):
    name: str
    response: str
    description: str = ""


class CustomCommandUpdate(BaseModel):
    response: str | None = None
    description: str | None = None


# --- Helpers ---

def _get_services(request: Request):
    db = request.app.state.db
    return (
        ConfigService(db),
        FilterService(db),
        CustomCommandService(db),
    )


# --- Config ---

@router.get("/guilds/{guild_id}/config")
async def get_config(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    config_svc, _, _ = _get_services(request)
    cfg = await config_svc.get(guild_id)
    return cfg.to_doc()


@router.patch("/guilds/{guild_id}/config")
async def update_config(request: Request, guild_id: int, body: ConfigUpdate):
    require_guild_permission(request, guild_id)
    config_svc, _, _ = _get_services(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    cfg = await config_svc.update(guild_id, **fields)
    return cfg.to_doc()


# --- Filter Rules ---

@router.get("/guilds/{guild_id}/filters")
async def get_filters(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    rules = await filter_svc.get_rules(guild_id)
    return [r.to_doc() for r in rules]


@router.post("/guilds/{guild_id}/filters")
async def add_filter(request: Request, guild_id: int, body: FilterRuleCreate):
    user = require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    rule = FilterRule(
        guild_id=guild_id,
        rule_type=body.rule_type,
        pattern=body.pattern,
        action=body.action,
        created_by=int(user["id"]),
    )
    await filter_svc.add_rule(rule)
    return {"ok": True}


@router.delete("/guilds/{guild_id}/filters/{pattern:path}")
async def delete_filter(request: Request, guild_id: int, pattern: str):
    require_guild_permission(request, guild_id)
    _, filter_svc, _ = _get_services(request)
    removed = await filter_svc.remove_rule(guild_id, pattern)
    if not removed:
        raise HTTPException(404, "Filter not found")
    return {"ok": True}


# --- Custom Commands ---

@router.get("/guilds/{guild_id}/commands")
async def get_commands(request: Request, guild_id: int):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    cmds = await cmd_svc.get_all(guild_id)
    return [c.to_doc() for c in cmds]


@router.post("/guilds/{guild_id}/commands")
async def create_command(request: Request, guild_id: int, body: CustomCommandCreate):
    user = require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    cmd = CustomCommand(
        guild_id=guild_id,
        name=body.name,
        response=body.response,
        description=body.description,
        created_by=int(user["id"]),
    )
    ok = await cmd_svc.add(cmd)
    if not ok:
        raise HTTPException(409, f"Command '{body.name}' already exists")
    return {"ok": True}


@router.patch("/guilds/{guild_id}/commands/{name}")
async def update_command(
    request: Request, guild_id: int, name: str, body: CustomCommandUpdate
):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "No fields to update")
    ok = await cmd_svc.update(guild_id, name, **fields)
    if not ok:
        raise HTTPException(404, "Command not found")
    return {"ok": True}


@router.delete("/guilds/{guild_id}/commands/{name}")
async def delete_command(request: Request, guild_id: int, name: str):
    require_guild_permission(request, guild_id)
    _, _, cmd_svc = _get_services(request)
    ok = await cmd_svc.remove(guild_id, name)
    if not ok:
        raise HTTPException(404, "Command not found")
    return {"ok": True}
