import asyncio
import logging

import uvicorn

from config import settings
from bot.bot import ModBot
from bot.database import Database
from web.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


async def main() -> None:
    db = Database(settings.mongodb_uri, settings.mongodb_db_name)
    await db.create_indexes()

    # Start the web dashboard
    web_app = create_app(settings, db)
    web_config = uvicorn.Config(web_app, host="0.0.0.0", port=settings.web_port, log_level="info")
    web_server = uvicorn.Server(web_config)

    async with ModBot(settings=settings) as bot:
        # Run both the bot and web server concurrently
        await asyncio.gather(
            bot.start(settings.discord_token),
            web_server.serve(),
        )


if __name__ == "__main__":
    asyncio.run(main())
