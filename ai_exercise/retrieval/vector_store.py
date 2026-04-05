"""Create a vector store."""

import logging

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Embeddable, EmbeddingFunction

logger = logging.getLogger(__name__)


def create_collection(
    client: ClientAPI, embedding_fn: EmbeddingFunction[Embeddable], name: str
) -> chromadb.Collection:
    """Create and return a Chroma collection, or get existing one if it exists"""
    logger.info("Creating or retrieving collection: %s", name)
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn)
