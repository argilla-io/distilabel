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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

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


# TODO: add `do_sample=False` if `temperature` is 0, as it will disable temperature sampling
# and will always use the most likely next token in the sequence (greedy decoding).
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
    ) -> List[LLMOutput]:
        prompt = self.task.generate_prompt(**input)
        if self.formatting_fn is not None:
            prompt = self.formatting_fn(prompt)
        raw_responses = [
            self._text_generation_with_backoff(prompt=prompt)
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
