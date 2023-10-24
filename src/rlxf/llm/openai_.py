import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union

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
        model: Literal[
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
        ],
        prompt_template: "PromptTemplate",
        openai_api_key: Union[str, None] = None,
        num_threads: Union[int, None] = None,
    ) -> None:
        super().__init__(prompt_template)

        self.model = model
        openai.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        assert (
            openai.api_key is not None
        ), "Either the `openai_api_key` arg or the `OPENAI_API_KEY` environment variable must be set to use the OpenAI API."

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

    def __del__(self) -> None:
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

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

    def _generate(self, input: Dict[str, Any], num_generations: int = 1):
        prompt = self.prompt_template.generate_prompt(**input)
        response = self._chat_completion_with_backoff(
            model=self.model, messages=prompt, n=num_generations
        )
        return [
            self.prompt_template.parse_output(choice["message"]["content"].strip())
            for choice in response["choices"]
        ]

    def generate(self, inputs: List[Dict[str, Any]], num_generations: int = 1) -> Any:
        if self.thread_pool_executor is not None:
            return [
                self.thread_pool_executor.submit(self._generate, input, num_generations)
                for input in inputs
            ]

        generations = []
        for input in inputs:
            output = self._generate(input, num_generations)
            generations.append(output)
        return generations

    @property
    def return_futures(self) -> bool:
        return self.thread_pool_executor is not None
