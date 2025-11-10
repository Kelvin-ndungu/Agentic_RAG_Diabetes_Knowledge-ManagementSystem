"""
Standalone server runner script.
Can be run from the backend directory or project root.
"""
import sys
from pathlib import Path

# Add project root to path if running from backend directory
backend_dir = Path(__file__).parent
project_root = backend_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

