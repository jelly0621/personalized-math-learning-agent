"""Environment-based configuration for the LLM client."""

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_VISION_MODEL = "deepseek-v4-flash-vision-exp"

# Load the project-local environment once so non-LLM runtime settings, such as
# Basic Auth and the SQLite path, work without constructing an LLM client.
load_dotenv(dotenv_path=ENV_FILE)


class Settings(BaseModel):
    """Validated settings loaded from the project-root .env file."""

    llm_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    vision_model: str = DEFAULT_VISION_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate the required LLM configuration."""

    environment_names = {
        "llm_provider": "LLM_PROVIDER",
        "llm_api_key": "LLM_API_KEY",
        "llm_base_url": "LLM_BASE_URL",
        "llm_model": "LLM_MODEL",
    }
    values = {
        field_name: os.getenv(environment_name, "").strip()
        for field_name, environment_name in environment_names.items()
    }
    values["vision_model"] = (
        os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL).strip()
        or DEFAULT_VISION_MODEL
    )
    missing = [
        environment_names[field_name]
        for field_name, value in values.items()
        if not value
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required LLM configuration: {missing_names}. "
            f"Create {ENV_FILE} from .env.example and fill in these values."
        )

    return Settings(**values)
