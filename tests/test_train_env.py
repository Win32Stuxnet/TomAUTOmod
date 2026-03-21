from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from bot.ml.train import main


@pytest.mark.asyncio
async def test_main_reads_mongodb_uri_from_env():
    """train.py should read MONGODB_URI from environment instead of defaulting to localhost."""
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb+srv://atlas.example.com/mydb",
        "MONGODB_DB_NAME": "test_db",
    }):
        with patch("bot.ml.train.load_dotenv"):
            with patch("bot.ml.train.export_data", new_callable=AsyncMock) as mock_export:
                mock_export.return_value = (MagicMock(), MagicMock())
                with patch("bot.ml.train.train"):
                    with patch("sys.argv", ["train.py"]):
                        await main()

                mock_export.assert_called_once()
                called_uri = mock_export.call_args[0][0]
                assert called_uri == "mongodb+srv://atlas.example.com/mydb"


@pytest.mark.asyncio
async def test_main_reads_db_name_from_env():
    """train.py should read MONGODB_DB_NAME from environment."""
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb://localhost:27017",
        "MONGODB_DB_NAME": "custom_db_name",
    }):
        with patch("bot.ml.train.load_dotenv"):
            with patch("bot.ml.train.export_data", new_callable=AsyncMock) as mock_export:
                mock_export.return_value = (MagicMock(), MagicMock())
                with patch("bot.ml.train.train"):
                    with patch("sys.argv", ["train.py"]):
                        await main()

                called_db_name = mock_export.call_args[0][1]
                assert called_db_name == "custom_db_name"


@pytest.mark.asyncio
async def test_main_falls_back_to_localhost_without_env():
    """Without MONGODB_URI set, should fall back to localhost."""
    env = os.environ.copy()
    env.pop("MONGODB_URI", None)
    env.pop("MONGODB_DB_NAME", None)

    with patch.dict(os.environ, env, clear=True):
        with patch("bot.ml.train.load_dotenv"):
            with patch("bot.ml.train.export_data", new_callable=AsyncMock) as mock_export:
                mock_export.return_value = (MagicMock(), MagicMock())
                with patch("bot.ml.train.train"):
                    with patch("sys.argv", ["train.py"]):
                        await main()

                called_uri = mock_export.call_args[0][0]
                assert called_uri == "mongodb://localhost:27017"


@pytest.mark.asyncio
async def test_main_cli_args_override_env():
    """CLI --db argument should override MONGODB_URI from env."""
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb+srv://atlas.example.com/mydb",
    }):
        with patch("bot.ml.train.load_dotenv"):
            with patch("bot.ml.train.export_data", new_callable=AsyncMock) as mock_export:
                mock_export.return_value = (MagicMock(), MagicMock())
                with patch("bot.ml.train.train"):
                    with patch("sys.argv", ["train.py", "--db", "mongodb://override:27017"]):
                        await main()

                called_uri = mock_export.call_args[0][0]
                assert called_uri == "mongodb://override:27017"


@pytest.mark.asyncio
async def test_main_calls_load_dotenv():
    """train.py should call load_dotenv() before reading env vars."""
    with patch("bot.ml.train.load_dotenv") as mock_dotenv:
        with patch("bot.ml.train.export_data", new_callable=AsyncMock) as mock_export:
            mock_export.return_value = (MagicMock(), MagicMock())
            with patch("bot.ml.train.train"):
                with patch("sys.argv", ["train.py"]):
                    await main()

        mock_dotenv.assert_called_once()
