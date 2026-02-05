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
    """
    Configuration for the OpenAI client.

    Attributes:
        api_key: The OpenAI API key.
        model: The model name to use for completions.
        embedding_model: The model name to use for embeddings.
        project_name: The name of the project for logging purposes.
    """
    api_key: str
    model: str
    embedding_model: str
    project_name: str


def load_config() -> OpenAIConfig:
    """
    Loads configuration from environment variables (including .env).

    Returns:
        An OpenAIConfig object containing the project settings.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing from environment.
    """
    # Load .env once per process; safe to call multiple times.
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file (not committed)."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
    project_name = os.getenv("PROJECT_NAME", "ain710-week5-nlp-in-the-wild").strip()

    return OpenAIConfig(
        api_key=api_key, 
        model=model, 
        embedding_model=embedding_model, 
        project_name=project_name
    )


def get_client() -> OpenAI:
    """
    Returns a configured OpenAI client.

    Returns:
        An instance of the OpenAI client initialized with the loaded API key.
    """
    cfg = load_config()
    # The OpenAI SDK reads the API key from env by default,
    # but we pass it explicitly for clarity and testability.
    return OpenAI(api_key=cfg.api_key)


if __name__ == "__main__":
    # Example usage:
    try:
        config = load_config()
        print(f"Loaded config for project: {config.project_name}")
        print(f"Using model: {config.model}")
        
        # client = get_client()
        # print("Client successfully initialized.")
    except RuntimeError as e:
        print(f"Configuration error: {e}")
