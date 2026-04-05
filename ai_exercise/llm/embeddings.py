"""Embeddings using OpenAI"""

import logging

from chromadb.utils.embedding_functions.openai_embedding_function import \
    OpenAIEmbeddingFunction

from ai_exercise.constants import SETTINGS

logger = logging.getLogger(__name__)

logger.info("Initializing OpenAI embedding function with model: %s", SETTINGS.embeddings_model)
openai_ef = OpenAIEmbeddingFunction(
    api_key=SETTINGS.openai_api_key.get_secret_value(),
    model_name=SETTINGS.embeddings_model,
)
