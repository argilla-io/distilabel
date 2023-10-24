import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Union

import openai
from openai.error import APIError, RateLimitError, ServiceUnavailableError, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from rlxf.llm.base import LLM

if TYPE_CHECKING:
    from rlxf.prompts.base import PromptTemplate


_OPENAI_API_RETRY_ON_EXCEPTIONS = (
    APIError,
    Timeout,
    RateLimitError,
    ServiceUnavailableError,
)
_OPENAI_API_STOP_AFTER_ATTEMPT = 6
_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10


class OpenAILLM(LLM):
    def __init__(
        self,
        model: str,
        prompt_template: "PromptTemplate",
        openai_api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_threads: Union[int, None] = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
        )

        openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert (
            model in self.available_models
        ), f"Provided `model` is not available in your OpenAI account, available models are {self.available_models}"
        self.model = model

        assert (
            openai.api_key is not None
        ), "Either the `openai_api_key` arg or the `OPENAI_API_KEY` environment variable must be set to use the OpenAI API."

    @cached_property
    def available_models(self) -> List[str]:
        return [
            model["id"]
            for model in openai.Model.list().get("data", [])
            if model.get("id") is not None
        ]

    @retry(
        retry=retry_if_exception_type(_OPENAI_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_OPENAI_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
    )
    def _chat_completion_with_backoff(self, **kwargs: Any) -> Any:
        return openai.ChatCompletion.create(**kwargs)

    def _generate(self, input: Dict[str, Any], num_generations: int = 1) -> List[Any]:
        prompt = self.prompt_template.generate_prompt(**input)
        response = self._chat_completion_with_backoff(
            model=self.model,
            messages=prompt,
            n=num_generations,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return [
            self.prompt_template.parse_output(choice["message"]["content"].strip())
            for choice in response["choices"]
        ]
