# -*- coding: utf-8 -*-
"""
keys.py - Common API key / configuration loader for all scripts in this folder.

Every other script in this folder does:

    import keys

and then reads keys.OPENAI_API_KEY / keys.PINECONE_API_KEY / keys.GOOGLE_API_KEY,
or calls keys.require(...) when the key is mandatory for that script to run.

Setup:
1. Create a file named `.env` in this same folder (ex04_Hybrid_Retrieval/).
2. Add your keys to it, one per line, e.g.:

    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=...
    PINECONE_API_KEY=...

3. Never commit the `.env` file - it is already ignored via the local .gitignore.

If a key is missing, the corresponding variable below will simply be None and any
script that needs it will raise a clear error when it tries to use it.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file placed next to this script (if it exists).
# override=True so this folder's .env always wins over a stray OPENAI_API_KEY
# (e.g. an OpenRouter key) that may already be set in your shell/system environment.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Many SDKs (OpenAI, google-genai) pick up credentials automatically from the
# environment, so we mirror whatever we found back into os.environ.
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def require(key_name: str, value: str) -> str:
    """Raise a helpful error if a required key was not found in the environment."""
    if not value:
        raise RuntimeError(
            f"{key_name} is not set. Add it to the .env file in "
            f"{os.path.dirname(os.path.abspath(__file__))} or export it as an "
            "environment variable before running this script."
        )
    return value
