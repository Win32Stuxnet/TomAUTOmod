from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.session import MongoSessionMiddleware

from web.routes.auth import router as auth_router
from web.routes.dashboard import router as dashboard_router
from web.routes.api import router as api_router

if TYPE_CHECKING:
    from bot.database import Database
    from config import Settings

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(settings: Settings, db: Database) -> FastAPI:
    """
    Create and configure the FastAPI application for the ModBot web dashboard.
    
    Parameters:
        settings (Settings): Application settings used to configure the app (e.g., web secret key).
        db (Database): Database instance used for session middleware and stored on app state.
    
    Returns:
        FastAPI: Configured FastAPI application with session middleware, mounted static files, and registered routers.
    """
    app = FastAPI(title="ModBot Dashboard", docs_url=None, redoc_url=None)

    app.state.settings = settings
    app.state.db = db

    app.add_middleware(MongoSessionMiddleware, secret_key=settings.web_secret_key, db=db)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(auth_router)
    app.include_router(dashboard_router)
    app.include_router(api_router, prefix="/api")

    log.info("Web dashboard app created.")
    return app
