"""
client.py

Centralized OpenAI client + configuration loader.
- Loads environment variables from .env (local only)
- Validates required config (API key)
- Provides a single place to change model defaults later
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    project_name: str


def load_config() -> OpenAIConfig:
    """
    Loads configuration from environment variables (including .env).
    Raises a clear error if required values are missing.
    """
    # Load .env once per process; safe to call multiple times.
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file (not committed)."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"
    project_name = os.getenv("PROJECT_NAME", "ain710-week5-nlp-in-the-wild").strip()

    return OpenAIConfig(api_key=api_key, model=model, project_name=project_name)


def get_client() -> OpenAI:
    """
    Returns a configured OpenAI client.
    """
    cfg = load_config()
    # The OpenAI SDK reads the API key from env by default,
    # but we pass it explicitly for clarity and testability.
    return OpenAI(api_key=cfg.api_key)
