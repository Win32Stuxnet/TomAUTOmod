
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, MutableMapping

import itsdangerous
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)

COOKIE_NAME = "session_id"
SESSION_TTL_SECONDS = 86400  # 24 hours


class SessionDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modified = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.modified = True

    def __delitem__(self, key):
        super().__delitem__(key)
        self.modified = True

    def pop(self, key, *args):
        self.modified = True
        return super().pop(key, *args)

    def clear(self):
        super().clear()
        self.modified = True


class MongoSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str, db, secure_cookie: bool = False) -> None:
        super().__init__(app)
        self.signer = itsdangerous.TimestampSigner(secret_key)
        self.db = db
        self.secure_cookie = secure_cookie

    @property
    def collection(self):
        return self.db._db["web_sessions"]

    async def dispatch(self, request: Request, call_next) -> Response:
        session_id = self._get_session_id(request)
        session_data: dict[str, Any] = {}

        if session_id:
            doc = await self.collection.find_one({"_id": session_id})
            if doc:
                session_data = doc.get("data", {})

        # Patch request.scope so request.session works like Starlette's
        session = SessionDict(session_data)
        request.scope["session"] = session

        response = await call_next(request)

        # Persist to MongoDB if changed
        if session.modified:
            sid = session_id or secrets.token_urlsafe(32)
            if len(session) == 0:
                # Session was cleared — delete from DB
                await self.collection.delete_one({"_id": sid})
                response.delete_cookie(COOKIE_NAME, path="/")
            else:
                await self.collection.update_one(
                    {"_id": sid},
                    {
                        "$set": {
                            "data": dict(session),
                            "updated_at": datetime.now(timezone.utc),
                        },
                        "$setOnInsert": {
                            "created_at": datetime.now(timezone.utc),
                        },
                    },
                    upsert=True,
                )
                signed = self.signer.sign(sid).decode()
                response.set_cookie(
                    COOKIE_NAME,
                    signed,
                    max_age=SESSION_TTL_SECONDS,
                    httponly=True,
                    secure=self.secure_cookie,
                    samesite="lax",
                    path="/",
                )

        return response

    def _get_session_id(self, request: Request) -> str | None:
        cookie = request.cookies.get(COOKIE_NAME)
        if not cookie:
            return None
        try:
            return self.signer.unsign(cookie, max_age=SESSION_TTL_SECONDS).decode()
        except (itsdangerous.BadSignature, itsdangerous.SignatureExpired):
            return None
