"""
Minimal diagnostic for python-dotenv parsing/loading.

This prints only booleans and lengths (never the secret value) so it's safe to share logs.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"

    vals = dotenv_values(env_path)
    keys_preview = [repr(k) for k in list(vals.keys())[:10]]
    key_from_file = (vals.get("OPENAI_API_KEY") or "").strip()

    loaded = load_dotenv(dotenv_path=str(env_path), override=False)
    key_from_env = os.getenv("OPENAI_API_KEY") or ""

    print(f"env_path={env_path}")
    print(f"env_exists={env_path.exists()}")
    print(f"dotenv_keys_preview={keys_preview}")
    print(f"dotenv_values_has_key={bool(key_from_file)} len={len(key_from_file)}")
    print(f"load_dotenv_returned={bool(loaded)}")
    print(f"os_environ_has_key={bool(key_from_env)} len={len(key_from_env)}")


if __name__ == "__main__":
    main()


