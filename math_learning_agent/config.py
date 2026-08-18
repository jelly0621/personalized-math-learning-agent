"""Environment-based configuration for the LLM client."""

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseModel):
    """Validated settings loaded from the project-root .env file."""

    llm_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate the required LLM configuration."""

    load_dotenv(dotenv_path=ENV_FILE)

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
