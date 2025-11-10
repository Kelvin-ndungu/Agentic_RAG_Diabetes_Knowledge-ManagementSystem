"""
Configuration module for loading and validating environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Required environment variables
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Optional environment variables with defaults
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "diabetes_guidelines_v1")

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
            "Please ensure the vector store has been created."
        )


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

