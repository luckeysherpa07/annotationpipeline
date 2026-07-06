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


GEMINI_KEY_LIST_FILE = PROJECT_ROOT / "api_key_list" / "gemini_api_key_list"
_gemini_key_pool = []
_key_pool_initialized = False

def _load_gemini_key_list() -> None:
    global _key_pool_initialized
    if _key_pool_initialized:
        return
    if not GEMINI_KEY_LIST_FILE.exists():
        raise FileNotFoundError(f"API key list file not found: {GEMINI_KEY_LIST_FILE}")
    
    with open(GEMINI_KEY_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract the last word which is the actual API key
            key = line.split()[-1].strip()
            if key:
                _gemini_key_pool.append(key)
    
    _key_pool_initialized = True


def create_gemini_client(api_key: str | None = None, api_key_source: str = "list"):
    """
    Build a Gemini client after confirming the SDK and API key are available.
    api_key_source can be "env" to use .env / os.environ, or "list" to use the key pool file.
    """
    load_environment()

    if genai is None:
        raise ImportError(
            "The Google GenAI SDK is not installed. Install dependencies from requirements.txt first."
        )

    if api_key is not None:
        resolved_api_key = api_key.strip()
    elif api_key_source == "list":
        _load_gemini_key_list()
        if not _gemini_key_pool:
            raise RuntimeError("All API keys in the key list have been exhausted!")
        resolved_api_key = _gemini_key_pool.pop(0)
    else:
        resolved_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not resolved_api_key:
        raise RuntimeError(
            f"Missing GEMINI_API_KEY. Set it in your environment, add it to {ENV_FILE}, or provide a valid key list."
        )

    # genai.Client() automatically looks for os.environ["GEMINI_API_KEY"] if not passed explicitly,
    # but since we are rotating, we must pass it explicitly to avoid polluting os.environ globally 
    # or failing when the env var is missing/stale.
    client = genai.Client(
        api_key=resolved_api_key,
        http_options=types.HttpOptions(timeout=300000)
    )
    client._resolved_api_key = resolved_api_key
    return client

def rotate_gemini_client(api_key_source: str = "list"):
    """
    Rotate to the next API key in the pool and return a new Gemini client.
    """
    if api_key_source != "list":
        raise RuntimeError("Cannot rotate API key unless api_key_source is 'list'")
        
    _load_gemini_key_list()
    if not _gemini_key_pool:
        raise RuntimeError("All API keys in the key list have been exhausted!")
        
    resolved_api_key = _gemini_key_pool.pop(0)
    print(f"\n[API KEY ROTATION] Switching to new key starting with {resolved_api_key[:8]}...")
    client = genai.Client(
        api_key=resolved_api_key,
        http_options=types.HttpOptions(timeout=300000)
    )
    client._resolved_api_key = resolved_api_key
    return client
