"""
Configuration module for loading and validating environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try to load from project root (parent of backend directory)
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Fallback to default .env location

# Required environment variables
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Optional environment variables with defaults
# Resolve chroma_db path relative to project root, not current working directory
_default_chroma_path = project_root / "chroma_db"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
if CHROMA_DB_PATH:
    # If provided, resolve relative to project root if it's a relative path
    chroma_path = Path(CHROMA_DB_PATH)
    if not chroma_path.is_absolute():
        CHROMA_DB_PATH = str(project_root / CHROMA_DB_PATH)
    else:
        CHROMA_DB_PATH = str(chroma_path)
else:
    CHROMA_DB_PATH = str(_default_chroma_path)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "diabetes_guidelines_v1")

# Retrieval configuration
# Justification: making these tunable avoids code changes when adjusting retrieval behavior.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_MIN_SIMILARITY = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.4"))

# LLM retry configuration
# Justification: transient API failures are common; retries reduce user-visible errors.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Request limits
# Justification: centralized request sizing makes validation consistent across services.
MAX_REQUEST_SIZE_KB = int(os.getenv("MAX_REQUEST_SIZE_KB", "10"))

# Telemetry
# Justification: toggleable timing logs help diagnose latency in production.
LOG_TIMINGS = os.getenv("LOG_TIMINGS", "true").lower() in ("1", "true", "yes")

# LangSmith observability
# Justification: explicit config enables tracing without hard-coding secrets.
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() in ("1", "true", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "diabetes-knowledge-hub")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")  # Optional (self-hosted / region)

# CORS configuration
# Comma-separated list of allowed origins, or "*" for all origins (not recommended for production)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
# Parse comma-separated origins into list
if CORS_ORIGINS == "*":
    CORS_ORIGINS_LIST = ["*"]
else:
    CORS_ORIGINS_LIST = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]

# Claude model configuration
CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_TEMPERATURE = 0.1

# Jina configuration
JINA_MODEL = "jina-embeddings-v4"
JINA_TASK = "text-matching"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_BATCH_SIZE = 10
JINA_MAX_RETRIES = 3


def validate_config():
    """
    Validate that all required environment variables are set.
    Raises ValueError if any required variable is missing.
    """
    missing = []
    
    if not CLAUDE_API_KEY:
        missing.append("CLAUDE_API_KEY")
    
    if not JINA_API_KEY:
        missing.append("JINA_API_KEY")
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file or environment."
        )
    
    # Validate paths
    chroma_path = Path(CHROMA_DB_PATH)
    if not chroma_path.exists():
        raise ValueError(
            f"ChromaDB path does not exist: {CHROMA_DB_PATH}\n"
            f"Resolved from: {Path(__file__).parent.parent}\n"
            "Please ensure the vector store has been created."
        )

    # Validate retrieval configuration
    if RETRIEVAL_TOP_K < 1:
        raise ValueError("RETRIEVAL_TOP_K must be >= 1")
    if not (0.0 <= RETRIEVAL_MIN_SIMILARITY <= 1.0):
        raise ValueError("RETRIEVAL_MIN_SIMILARITY must be between 0.0 and 1.0")

    # LangSmith sanity checks (non-fatal)
    # Justification: tracing is optional; warn if enabled without key.
    if LANGSMITH_TRACING and not LANGSMITH_API_KEY:
        print("⚠ LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.")


if __name__ == "__main__":
    # Test configuration loading
    try:
        validate_config()
        print("✓ Configuration validated successfully")
        print(f"  • ChromaDB path: {CHROMA_DB_PATH}")
        print(f"  • Collection: {COLLECTION_NAME}")
        print(f"  • Claude model: {CLAUDE_MODEL}")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")

