"""FastAPI app creation, main API routes."""

import logging
from typing import cast

from chromadb.api.types import Embeddable, EmbeddingFunction
from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import create_prompt, get_completion
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (add_documents, build_docs,
                                                 get_all_json_data, split_docs)
from ai_exercise.models import (ChatOutput, ChatQuery, HealthRouteOutput,
                                LoadDocumentsOutput, TokenUsage)
from ai_exercise.retrieval.retrieval import get_relevant_chunks, rewrite_query
from ai_exercise.retrieval.vector_store import create_collection

logger = logging.getLogger(__name__)

app = FastAPI()

collection = create_collection(
    chroma_client, cast(EmbeddingFunction[Embeddable], openai_ef), SETTINGS.collection_name
)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store."""
    logger.info("Starting document loading")
    all_json_data = await get_all_json_data()
    documents = []
    for json_data in all_json_data:
        documents.extend(build_docs(json_data))

    # split docs
    documents = split_docs(documents)

    # load documents into vector store
    await add_documents(collection, documents)

    logger.info("Number of documents in collection: %d", collection.count())

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
async def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    logger.info("Chat query received: %.200s", chat_query.query)
    # Rewrite the query to match API documentation vocabulary
    rewritten, rewrite_tokens = await rewrite_query(
        client=openai_client, query=chat_query.query, model=SETTINGS.openai_model
    )

    # Get relevant chunks from the collection
    relevant_chunks = await get_relevant_chunks(
        collection=collection, query=rewritten, k=SETTINGS.k_neighbors
    )

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=relevant_chunks)

    logger.info("Augmented prompt: %.200s", prompt)

    # Get completion from LLM
    result, completion_tokens = await get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    token_usage = None
    if SETTINGS.debug_mode:
        token_usage = {
            "rewrite": TokenUsage(prompt=rewrite_tokens["prompt"], completion=rewrite_tokens["completion"]),
            "completion": TokenUsage(prompt=completion_tokens["prompt"], completion=completion_tokens["completion"]),
        }

    return ChatOutput(message=result, token_usage=token_usage)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
