import asyncio
import logging

from config import settings
from bot.bot import ModBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


async def main() -> None:
    async with ModBot(settings=settings) as bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(main())
