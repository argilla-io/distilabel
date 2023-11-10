# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import openai
from openai.error import APIError, RateLimitError, ServiceUnavailableError, Timeout
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.tasks.utils import Prompt

if TYPE_CHECKING:
    from distilabel.tasks.base import Task


_OPENAI_API_RETRY_ON_EXCEPTIONS = (
    APIError,
    Timeout,
    RateLimitError,
    ServiceUnavailableError,
    ConnectionError,
)
_OPENAI_API_STOP_AFTER_ATTEMPT = 6
_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_OPENAI_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

logger = get_logger()


class OpenAILLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str = "gpt-3.5-turbo",
        openai_api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        frequence_penalty: Union[float, None] = None,
        presence_penalty: Union[float, None] = None,
        temperature: Union[float, None] = None,
        top_p: Union[float, None] = None,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            num_threads=num_threads,
            formatting_fn=formatting_fn,
        )

        self.max_tokens = max_new_tokens
        self.frequence_penalty = frequence_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.__generation_attrs = [
            "max_tokens",
            "frequence_penalty",
            "presence_penalty",
            "temperature",
            "top_p",
        ]

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
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def _chat_completion_with_backoff(self, **kwargs: Any) -> Any:
        return openai.ChatCompletion.create(**kwargs)

    def _generate(
        self,
        input: Dict[str, Any],
        num_generations: int = 1,
    ) -> List[LLMOutput]:
        prompt = self.task.generate_prompt(**input)
        if not isinstance(prompt, Prompt) and self.formatting_fn is not None:
            warnings.warn(
                f"The method `generate_prompt` is not returning a `Prompt` class but a prompt of `type={type(prompt)}`, meaning that a pre-formatting has already been applied in the `task.generate_prompt` method, so the usage of a `formatting_fn` is discouraged.",
                UserWarning,
                stacklevel=2,
            )
            prompt = self.formatting_fn(prompt)
        elif isinstance(prompt, Prompt) and self.formatting_fn is None:
            prompt = prompt.format_as(format="openai")
        if not isinstance(prompt, list):
            raise ValueError(
                f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be a `list`, make sure that `task.generate_prompt` returns a `list` or that the `formatting_fn` formats the prompt as a `list`, where each item follows OpenAI's format of `{'role': ..., 'content': ...}`."
            )
        generation_kwargs = {}
        for generation_attr in self.__generation_attrs:
            value = getattr(self, generation_attr)
            if value is not None:
                generation_kwargs[generation_attr] = value
        raw_responses = self._chat_completion_with_backoff(
            messages=prompt,
            model=self.model,
            n=num_generations,
            **generation_kwargs,
        )
        raw_responses = raw_responses.to_dict_recursive()

        outputs = []
        for raw_response in raw_responses["choices"]:
            try:
                parsed_response = self.task.parse_output(
                    raw_response["message"]["content"].strip()
                )
            except Exception as e:
                logger.error(f"Error parsing OpenAI response: {e}")
                parsed_response = None
            outputs.append(
                LLMOutput(
                    prompt_used=prompt,
                    raw_output=raw_responses,
                    parsed_output=parsed_response,
                )
            )
        return outputs
