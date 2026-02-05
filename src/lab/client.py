"""
client.py

Centralized OpenAI client + configuration loader.
- Loads environment variables from .env (local only)
- Validates required config (API key)
- Provides a single place to change model defaults later
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List
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
        sentiment_prompt: The system prompt for sentiment analysis.
        sentiment_temperature: Default temperature for sentiment calls.
        sentiment_max_output_tokens: Default max tokens for sentiment outputs.
        sentiment_include_logprobs: Default flag to include logprobs for sentiment.
        entity_extraction_prompt: The system prompt for entity extraction.
        narrative_prompt: The prompt template for embedding summaries.
    """
    api_key: str
    model: str
    embedding_model: str
    project_name: str
    sentiment_prompt: str
    sentiment_temperature: float
    sentiment_max_output_tokens: int
    sentiment_include_logprobs: bool
    entity_extraction_prompt: str
    narrative_prompt: str
    test_text: str
    batch_texts: List[str]


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

    sentiment_prompt = os.getenv(
        "SENTIMENT_PROMPT",
        "You are an NLP evaluator. Your job is to classify the sentiment of the user's text.\n"
        "Return ONLY a JSON object that matches the provided schema.\n"
        "Rules:\n"
        "- label must be one of: positive, neutral, negative\n"
        "- confidence must be a number from 0.0 to 1.0\n"
        "- rationale must be short and grounded in the text (no speculation)\n"
    ).strip()

    # Sentiment defaults (overridable per-request)
    try:
        sentiment_temperature = float(os.getenv("SENTIMENT_TEMPERATURE", "0.2"))
    except ValueError:
        sentiment_temperature = 0.2
    try:
        sentiment_max_output_tokens = int(os.getenv("SENTIMENT_MAX_OUTPUT_TOKENS", "200"))
    except ValueError:
        sentiment_max_output_tokens = 200
    sentiment_include_logprobs_env = os.getenv("SENTIMENT_INCLUDE_LOGPROBS", "true").strip().lower()
    sentiment_include_logprobs = sentiment_include_logprobs_env in {"1", "true", "yes", "on"}

    entity_extraction_prompt = os.getenv(
        "ENTITY_EXTRACTION_PROMPT",
        "You are a helpful assistant that extracts entities (people, places, things) from text. "
        "Return only a comma-separated list of entities, or 'None' if none found."
    ).strip()

    narrative_prompt = os.getenv(
        "NARRATIVE_PROMPT",
        "You are an AI educator teaching a class about Natural Language Processing and Embeddings.\n"
        "Provide a concise (2-3 paragraph) narrative overview of the following embedding run results:\n\n"
        "- Total items processed: {total_processed}\n"
        "- Mean Magnitude (L2 Norm): {mean_magnitude}\n"
        "- Unique Entities Found: {unique_entities_count}\n"
        "- Entities list: {entities_list}\n"
        "- Model used: {model}\n\n"
        "Explain what the Mean Magnitude tells us about normalization in this model.\n"
        "Discuss the significance of the entities extracted in relation to the semantic vectors.\n"
        "Keep the tone professional yet encouraging for students."
    ).strip()

    test_text = os.getenv("TEST_TEXT", "I love this new feature!").strip()
    batch_texts_raw = os.getenv(
        "BATCH_TEXTS",
        '\\"I love the design, but the setup was frustrating.\\";'
        '\\"This was a complete waste of time.\\";'
        '\\"Absolutely fantastic experience — would recommend.\\";'
        '\\"Yeah, great… just what I needed (eye roll).\\"'
    ).strip()
    
    # Improved parsing: handles "text1";"text2" OR "\"text1\";\"text2\"" to allow semicolons inside quotes
    # EDUCATIONAL NOTE: We use Regular Expressions (regex) here to handle character escaping.
    # The pattern r'"((?:\\.|[^"\\])*)"' is a classic regex for finding quoted strings.
    # - (?:\\.|[^"\\])*  means: match either an escaped character (\\.) OR any character that isn't a quote or backslash ([^"\\]).
    # - This ensures that if someone puts \" inside their text, it doesn't prematurely end the match.
    if batch_texts_raw.startswith('"') and batch_texts_raw.endswith('"'):
        # Extract matches using regex
        pattern = r'"((?:\\.|[^"\\])*)"'
        matches = re.findall(pattern, batch_texts_raw)
        
        if matches:
            # Unescape characters (e.g., \" -> ") and clean up
            batch_texts = [m.replace('\\"', '"').replace('\\\\', '\\').strip() for m in matches if m.strip()]
        else:
            # Fallback if regex fails but we have quotes
            batch_texts = [batch_texts_raw.strip('"').replace('\\"', '"')]
    else:
        # Fallback to simple semicolon split if not fully quoted
        batch_texts = [t.strip() for t in batch_texts_raw.split(";") if t.strip()]

    return OpenAIConfig(
        api_key=api_key,
        model=model,
        embedding_model=embedding_model,
        project_name=project_name,
        sentiment_prompt=sentiment_prompt,
        sentiment_temperature=sentiment_temperature,
        sentiment_max_output_tokens=sentiment_max_output_tokens,
        sentiment_include_logprobs=sentiment_include_logprobs,
        entity_extraction_prompt=entity_extraction_prompt,
        narrative_prompt=narrative_prompt,
        test_text=test_text,
        batch_texts=batch_texts
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
