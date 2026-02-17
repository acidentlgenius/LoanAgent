"""App-wide configuration and environment settings."""

import os
try:
    import streamlit as st
except ImportError:
    st = None
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

def get_secret(key, default=None):
    """Try st.secrets first, then os.getenv."""
    if st is not None:
        try:
            # Accessing st.secrets might raise FileNotFoundError if no secrets.toml on local
            if key in st.secrets:
                return st.secrets[key]
        except (FileNotFoundError, AttributeError, KeyError):
            pass
    return os.getenv(key, default)

# LLM (Google Gemini)
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY", "")
LLM_MODEL = get_secret("LLM_MODEL", "gemini-2.5-flash-lite")

# Journey limits
MAX_STEPS_GUARD = 25  # Hard terminate if exceeded
TOTAL_JOURNEY_STEPS = 15
MAX_RETRIES_PER_STEP = 2  # Re-prompt for missing fields before moving on

# LangSmith
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "loan-agent")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
