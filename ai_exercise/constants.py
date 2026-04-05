"""Set up some constants for the project."""

import logging

import chromadb
from openai import AsyncOpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class Settings(BaseSettings):
    """Settings for the demo app.

    Reads from environment variables.
    You can create the .env file from the .env_example file.

    !!! SecretStr is a pydantic type that hides the value in logs.
    If you want to use the real value, you should do:
    SETTINGS.<variable>.get_secret_value()
    """

    class Config:
        """Config for the settings."""

        env_file = ".env"

    openai_api_key: SecretStr
    openai_model: str
    embeddings_model: str

    collection_name: str 
    chunk_size: int 
    k_neighbors: int
    debug_mode: bool

    knowledge_base_urls: list[str] = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json",
    ]


SETTINGS = Settings()  # type: ignore

if SETTINGS.debug_mode:
    logging.getLogger().setLevel(logging.DEBUG)

# clients
openai_client = AsyncOpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())
chroma_client = chromadb.PersistentClient(path="./.chroma_db")
