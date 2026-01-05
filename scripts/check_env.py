"""
Small helper script to validate local environment setup.

This avoids complex shell quoting and makes it easy to verify that:
- required dependencies are installed in the active venv
- OPENAI_API_KEY is being loaded from .env
"""

from __future__ import annotations


def main() -> None:
    # Import dependencies explicitly so missing packages are obvious.
    import dotenv  # noqa: F401
    import pydantic_settings  # noqa: F401

    # Ensure imports work when executing from project root.
    # (When launched from other working directories, Python may not include the project root.)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from app.core.config import settings  # noqa: E402

    # Do not print the actual key. Only print whether it's set and its length.
    import os  # noqa: E402

    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_exists = env_path.exists()
    env_val = os.getenv("OPENAI_API_KEY") or ""
    print(f"cwd={Path.cwd()}")
    print(f"env_file_exists={env_exists}")
    print(f"env_OPENAI_API_KEY_set={bool(env_val)} len={len(env_val)}")
    print(f"settings_openai_api_key_set={bool(settings.openai_api_key)} len={len(settings.openai_api_key)}")


if __name__ == "__main__":
    main()


