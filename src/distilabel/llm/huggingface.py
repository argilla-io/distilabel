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
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import torch
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
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from distilabel.llm.base import LLM
from distilabel.logger import get_logger

if TYPE_CHECKING:
    from distilabel.tasks.base import Task


_INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS = (
    InferenceTimeoutError,
    TextGenerationError,
)
_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT = 6
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

logger = get_logger()


class TransformersLLM(LLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
            formatting_fn=formatting_fn,
        )

        self.model = model
        if self.device != "cpu":
            self.model.to(self.device)

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            hasattr(self.tokenizer, "use_default_system_prompt")
            and self.tokenizer.use_default_system_prompt
        ):
            # The `tokenizer` also has a method named `apply_chat_template` that expects a `Conversation` as OpenAI does with the ChatML format
            warnings.warn(
                "The provided `tokenizer` has `use_default_system_prompt=True` which means that the default system prompt will be used, which may collide with the `task` provided as an arg to this class.",
                UserWarning,
                stacklevel=2,
            )

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> Tuple[Any, List[Any]]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        encoding = self.tokenizer(text=prompt, padding=True, return_tensors="pt")
        if self.device != "cpu":
            encoding = encoding.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **encoding,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    num_generations=num_generations,
                ),
            )
        raw_outputs = self.tokenizer.batch_decode(
            generated_ids[:, -(encoding.input_ids.shape[1]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        parsed_outputs = []
        for raw_output in raw_outputs:
            try:
                parsed_output = self.task.parse_output(raw_output)
            except Exception as e:
                logger.error(f"Error parsing Transformers output: {e}")
                parsed_output = {}
            parsed_outputs.append(parsed_output)
        return raw_outputs, parsed_outputs


class InferenceEndpointsLLM(LLM):
    def __init__(
        self,
        endpoint_url: str,
        task: "Task",
        token: Union[str, None] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_threads=num_threads,
            formatting_fn=formatting_fn,
        )

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
        return self.client.text_generation(
            **kwargs, max_new_tokens=self.max_new_tokens, temperature=self.temperature
        )

    def _generate(
        self, input: Dict[str, Any], num_generations: int = 1
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        raw_responses = [
            self._text_generation_with_backoff(prompt=prompt)
            for _ in range(num_generations)
        ]
        parsed_responses = []
        for response in raw_responses:
            try:
                parsed_response = self.task.parse_output(response)
            except Exception as e:
                logger.error(f"Error parsing Inference Endpoints output: {e}")
                parsed_response = {}
            parsed_responses.append(parsed_response)
        return raw_responses, parsed_responses
