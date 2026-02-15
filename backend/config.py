from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frontend static build directory
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "out"

# Model defaults
DEFAULT_MAX_NEW_TOKENS = 60
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_MIN_NEW_TOKENS = 3
