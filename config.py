"""Configuration for the application."""

from os import environ
from pathlib import Path
from dotenv import dotenv_values

# Load .env from project root (directory containing this file)
_env_path = Path(__file__).resolve().parent / ".env"
_config = dotenv_values(_env_path)


def get_env(key: str, default: str | None = None) -> str | None:
    """Return env var: os.environ overrides .env; then default."""
    return environ.get(key) or _config.get(key) or default
