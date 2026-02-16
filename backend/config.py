import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./slm_chat.db")

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", "slm-chat-secret-change-in-production")

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Model defaults
DEFAULT_MAX_NEW_TOKENS = 60
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_MIN_NEW_TOKENS = 3
