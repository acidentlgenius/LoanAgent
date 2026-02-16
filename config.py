"""App-wide configuration and environment settings."""

import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

# LLM (Google Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Journey limits
MAX_STEPS_GUARD = 25  # Hard terminate if exceeded
TOTAL_JOURNEY_STEPS = 15

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
