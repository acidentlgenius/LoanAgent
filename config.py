"""App-wide configuration and environment settings."""

import os

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Journey limits
MAX_STEPS_GUARD = 25  # Hard terminate if exceeded
TOTAL_JOURNEY_STEPS = 15

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
