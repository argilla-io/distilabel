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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

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
from distilabel.utils.imports import _HUGGINGFACE_HUB_AVAILABLE

if _HUGGINGFACE_HUB_AVAILABLE:
    from huggingface_hub import InferenceTimeoutError, get_inference_endpoint
    from huggingface_hub.inference._text_generation import TextGenerationError

    _INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS = (
        InferenceTimeoutError,
        TextGenerationError,
    )
else:
    _INFERENCE_ENDPOINTS_API_RETRY_ON_EXCEPTIONS = ()

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats


_INFERENCE_ENDPOINTS_API_STOP_AFTER_ATTEMPT = 6
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_INFERENCE_ENDPOINTS_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10

logger = get_logger()


class InferenceEndpointsLLM(LLM):
    def __init__(
        self,
        endpoint_name: str,
        task: "Task",
        endpoint_namespace: Union[str, None] = None,
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
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the InferenceEndpointsLLM class.

        Args:
            endpoint_name (str): The name of the endpoint.
            task (Task): The task to be performed by the LLM.
            endpoint_namespace (Union[str, None]): The namespace of the endpoint. Defaults to None.
            token (Union[str, None]): The token for the endpoint. Defaults to None.
            max_new_tokens (int): The maximum number of tokens to be generated. Defaults to 128.
            repetition_penalty (Union[float, None]): The repetition penalty to be used for generation. Defaults to None.
            seed (Union[int, None]): The seed for generation. Defaults to None.
            do_sample (bool): Whether to do sampling. Defaults to False.
            temperature (Union[float, None]): The temperature for generation. Defaults to None.
            top_k (Union[int, None]): The top_k for generation. Defaults to None.
            top_p (Union[float, None]): The top_p for generation. Defaults to None.
            typical_p (Union[float, None]): The typical_p for generation. Defaults to None.
            num_threads (Union[int, None]): The number of threads. Defaults to None.
            prompt_format (Union["SupportedFormats", None]): The format of the prompt. Defaults to None.
            prompt_formatting_fn (Union[Callable[..., str], None]): The function for formatting the prompt. Defaults to None.

        Examples:
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import InferenceEndpointsLLM
            >>> task = Task()
            >>> llm = InferenceEndpointsLLM(
            ...     endpoint_name="<INFERENCE_ENDPOINT_NAME>",
            ...     task=task,
            ... )
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _HUGGINGFACE_HUB_AVAILABLE:
            raise ImportError(
                "`InferenceEndpointsLLM` cannot be used as `huggingface-hub` is not "
                "installed, please install it with `pip install huggingface-hub`."
            )

        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p

        self.inference_endpoint = get_inference_endpoint(
            name=endpoint_name, namespace=endpoint_namespace, token=token
        )
        self.inference_endpoint.wait(timeout=30)

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "do_sample": self.do_sample,
                "max_new_tokens": self.max_new_tokens,
                "repetition_penalty": self.repetition_penalty,
                "seed": self.seed,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "typical_p": self.typical_p,
            },
        )

    @property
    def model_name(self) -> str:
        """Returns the model name of the endpoint."""
        return self.inference_endpoint.repository

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
        """Performs text generation with backoff in case of an error."""
        return self.inference_endpoint.client.text_generation(**kwargs)  # type: ignore

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        """Generates `num_generations` for each input in `inputs`.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to 1.

        Returns:
            List[List[LLMOutput]]: the outputs of the LLM.
        """
        prompts = self._generate_prompts(
            inputs, default_format=None, expected_output_type=str
        )
        outputs = []
        for prompt in prompts:
            raw_responses = [
                self._text_generation_with_backoff(
                    prompt=prompt,
                    do_sample=self.do_sample,
                    max_new_tokens=self.max_new_tokens,
                    repetition_penalty=self.repetition_penalty,
                    seed=self.seed,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    typical_p=self.typical_p,
                )
                for _ in range(num_generations)
            ]
            output = []
            for raw_response in raw_responses:
                try:
                    parsed_response = self.task.parse_output(raw_response)
                except Exception as e:
                    logger.error(f"Error parsing Inference Endpoints output: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=raw_response,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
