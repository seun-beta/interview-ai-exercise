"""Generate a response using an LLM."""

from logging import getLogger

from openai import AsyncOpenAI

from ai_exercise.llm.prompt import PROMPT_TEMPLATE

logger = getLogger(__name__)


def create_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context"""
    context_str = "\n\n".join(context)
    logger.info("Context string: %.200s", context_str)
    return PROMPT_TEMPLATE.format(context=context_str, query=query)


async def get_completion(client: AsyncOpenAI, prompt: str, model: str) -> tuple[str, dict[str, int]]:
    """Get completion from OpenAI.

    Returns (result, {"prompt": N, "completion": N}).
    """
    logger.info("Requesting completion from model %s", model)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.choices[0].message.content or ""
    usage = response.usage
    tokens = {
        "prompt": usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
    }
    logger.info("Completion received: %.200s", result)
    return result, tokens
