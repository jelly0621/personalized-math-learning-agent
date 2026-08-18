"""Minimal OpenAI-compatible LLM client."""

from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from math_learning_agent.config import get_settings


class LLMClient:
    """Small wrapper around the official OpenAI Python client."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.llm_model
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def invoke(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        """Send OpenAI-format messages and return assistant text only."""

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("The LLM response did not contain text content.")
        return content
