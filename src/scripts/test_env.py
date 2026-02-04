"""
test_env.py

Verifies that the environment variables and OpenAI client are correctly configured.
"""

from src.lab.client import load_config, get_client


def main() -> None:
    """
    Loads configuration and initializes the OpenAI client to verify the environment.
    """
    print("Testing environment configuration...")
    try:
        cfg = load_config()
        print("\nConfiguration loaded successfully:")
        print(f" - Project: {cfg.project_name}")
        print(f" - Model: {cfg.model}")
        print(f" - API Key: sk-...{cfg.api_key[-4:] if len(cfg.api_key) > 4 else '****'}")

        client = get_client()
        print(f"\nClient initialized successfully: {type(client)}")
    except Exception as e:
        print(f"\nEnvironment test failed: {e}")


if __name__ == "__main__":
    main()
