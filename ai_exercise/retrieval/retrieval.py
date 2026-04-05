"""Retrieve relevant chunks from a vector store."""

import asyncio
import logging

import chromadb
from openai import AsyncOpenAI

from ai_exercise.llm.prompt import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


async def rewrite_query(client: AsyncOpenAI, query: str, model: str) -> tuple[str, dict[str, int]]:
    """Rewrite a user question to match API documentation vocabulary.

    Returns (rewritten_query, {"prompt": N, "completion": N}).
    """
    logger.info("Rewriting query: %.200s", query)
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    rewritten = response.choices[0].message.content or ""
    usage = response.usage
    tokens = {
        "prompt": usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
    }
    logger.info("Rewritten query: %.200s", rewritten)
    return rewritten, tokens


async def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list[str]:
    """Retrieve k most relevant chunks for the query"""
    logger.info("Retrieving chunks for query: %.200s", query)
    results = await asyncio.to_thread(collection.query, query_texts=[query], n_results=k)

    documents = results["documents"] or [[]]
    chunks = documents[0]
    logger.info("Retrieved %d relevant chunks:\n%s", len(chunks), "\n---\n".join(chunks))
    return chunks
