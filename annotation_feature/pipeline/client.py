from pathlib import Path
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def load_environment() -> None:
    """
    Load local environment variables from the project .env file when available.
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=ENV_FILE, override=True)


def create_gemini_client():
    """
    Build a Gemini client after confirming the SDK and API key are available.
    """
    load_environment()

    if genai is None:
        raise ImportError(
            "The Google GenAI SDK is not installed. Install dependencies from requirements.txt first."
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing GEMINI_API_KEY. Set it in your environment or add it to {ENV_FILE}."
        )

    return genai.Client()