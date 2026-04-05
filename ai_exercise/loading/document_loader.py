"""Document loader for the RAG example."""

import asyncio
import logging
from typing import Any

import chromadb
import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.openapi_chunker import (chunk_endpoints,
                                                 chunk_supplementary)

logger = logging.getLogger(__name__)


async def get_json_data(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """Send a GET request to the given URL and return the JSON data."""
    logger.info("Fetching JSON data from %s", url)
    response = await client.get(url)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


async def get_all_json_data() -> list[dict[str, Any]]:
    """Fetch JSON data from all knowledge base URLs concurrently."""
    logger.info("Fetching data from %d knowledge base URLs", len(SETTINGS.knowledge_base_urls))
    async with httpx.AsyncClient() as client:
        return list(await asyncio.gather(
            *(get_json_data(client, url) for url in SETTINGS.knowledge_base_urls)
        ))


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk OpenAPI spec into semantically meaningful documents."""
    endpoint_docs = chunk_endpoints(data)
    supplementary_docs = chunk_supplementary(data)
    docs = endpoint_docs + supplementary_docs
    logger.info("Built %d documents (%d endpoints, %d supplementary)",
                len(docs), len(endpoint_docs), len(supplementary_docs))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    result = splitter.split_documents(docs_array)
    logger.info("Split into %d documents (from %d)", len(result), len(docs_array))
    return result


async def add_documents(collection: chromadb.Collection, docs: list[Document], batch_size: int = 100) -> None:
    """Add documents to the collection in batches to avoid OpenAI token limits."""
    logger.info("Adding %d documents to collection in batches of %d", len(docs), batch_size)
    for start_index in range(0, len(docs), batch_size):
        batch = docs[start_index : start_index + batch_size]
        await asyncio.to_thread(
            collection.add,
            documents=[doc.page_content for doc in batch],
            metadatas=[doc.metadata or {} for doc in batch],
            ids=[f"doc_{start_index + offset}" for offset in range(len(batch))],
        )
