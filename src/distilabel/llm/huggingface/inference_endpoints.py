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
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Union

from huggingface_hub import InferenceClient, InferenceTimeoutError
from huggingface_hub.inference._text_generation import TextGenerationError
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
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from distilabel.tasks.base import Task


_INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS = (
    InferenceTimeoutError,
    TextGenerationError,
    ConnectionError,
)
_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT = 6
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

logger = get_logger()


class InferenceEndpointsLLM(LLM):
    def __init__(
        self,
        endpoint_url: str,
        task: "Task",
        token: Union[str, None] = None,
        max_new_tokens: int = 128,
        repetition_penalty: Union[float, None] = None,
        seed: Union[int, None] = None,
        do_sample: bool = False,
        temperature: Union[float, None] = None,
        top_k: Union[int, None] = None,
        top_p: Union[float, None] = None,
        typical_p: Union[float, None] = None,
        num_threads: Union[int, None] = None,
        prompt_format: Union[
            Literal["llama2", "openai", "chatml", "zephyr"], None
        ] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p

        self.__generation_attrs = [
            "do_sample",
            "max_new_tokens",
            "repetition_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "typical_p",
        ]

        self.client = InferenceClient(model=endpoint_url, token=token)

    @retry(
        retry=retry_if_exception_type(_INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def _text_generation_with_backoff(self, **kwargs: Any) -> Any:
        return self.client.text_generation(**kwargs)

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> List[LLMOutput]:
        prompt = self.task.generate_prompt(**input)
        if not isinstance(prompt, Prompt) and self.prompt_formatting_fn is not None:
            warnings.warn(
                f"The method `generate_prompt` is not returning a `Prompt` class but a prompt of `type={type(prompt)}`, meaning that a pre-formatting has already been applied in the `task.generate_prompt` method, so the usage of a `formatting_fn` is discouraged.",
                UserWarning,
                stacklevel=2,
            )
            prompt = self.prompt_formatting_fn(prompt)
        elif isinstance(prompt, Prompt) and self.prompt_formatting_fn is None:
            if self.prompt_format:
                prompt = prompt.format_as(format=self.prompt_format)  # type: ignore
            else:
                prompt = f"{prompt.system_prompt}\n{prompt.formatted_prompt}"
        if not isinstance(prompt, str):
            raise ValueError(
                f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be a `str`, make sure that `task.generate_prompt` returns a `str` or that the `formatting_fn` formats the prompt as a `str`."
            )
        generation_kwargs = {}
        for generation_attr in self.__generation_attrs:
            value = getattr(self, generation_attr)
            if value is not None:
                generation_kwargs[generation_attr] = value
        raw_responses = [
            self._text_generation_with_backoff(
                prompt=prompt,
                **generation_kwargs,
            )
            for _ in range(num_generations)
        ]
        outputs = []
        for raw_response in raw_responses:
            try:
                parsed_response = self.task.parse_output(raw_response)
            except Exception as e:
                logger.error(f"Error parsing Inference Endpoints output: {e}")
                parsed_response = None
            outputs.append(
                LLMOutput(
                    prompt_used=prompt,
                    raw_output=raw_response,
                    parsed_output=parsed_response,
                )
            )
        return outputs
