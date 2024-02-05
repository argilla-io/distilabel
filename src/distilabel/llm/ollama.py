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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union
from urllib import error

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
from distilabel.utils.imports import _OLLAMA_AVAILABLE

if _OLLAMA_AVAILABLE:
    import ollama

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()

_OLLAMA_API_RETRY_ON_EXCEPTIONS = (error.HTTPError,)
_OLLAMA_API_STOP_AFTER_ATTEMPT = 6
_OLLAMA_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER = 1
_OLLAMA_API_WAIT_RANDOM_EXPONENTIAL_MAX = 10


class OllamaLLM(LLM):
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def __init__(
        self,
        model: str,
        task: "Task",
        max_new_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        top_k: Union[int, None] = None,
        top_p: Union[float, None] = None,
        mirostat: Union[int, None] = None,
        mirostat_eta: Union[float, None] = None,
        mirostat_tau: Union[float, None] = None,
        num_ctx: Union[int, None] = None,
        num_gqa: Union[int, None] = None,
        num_gpu: Union[int, None] = None,
        num_threads: Union[int, None] = None,
        repeat_last_n: Union[int, None] = None,
        repeat_penalty: Union[float, None] = None,
        seed: Union[int, None] = None,
        stop: Union[str, None] = None,
        tfs_z: Union[float, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """
        Initializes the OllamaLLM class and aligns with https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

        Args:
            model (str): the model to be used for generation.
            task (Task): the task to be performed by the LLM.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to `None`.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to `None`.
            top_k (int, optional): the top-k value to be used for generation.
                Defaults to `None`.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to `None`.
            mirostat (int, optional): the Mirostat value to enable it or set the version.
                Defaults to `None`.
            mirostat_eta (float, optional): the eta value to be used for Mirostat.
                Defaults to `None`.
            mirostat_tau (float, optional): the tau value to be used for Mirostat.
                Defaults to `None`.
            num_ctx (int, optional): the number of contexts to be used for generation.
                Defaults to `None`.
            num_gqa (int, optional): the number of GQA to be used for generation.
                Defaults to `None`.
            num_gpu (int, optional): the number of GPUs to be used for generation.
                Defaults to `None`.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
            repeat_last_n (Union[int, None], optional): the number of tokens to be used
                for RepeatLastN. Defaults to `None`.
            repeat_penalty (Union[float, None], optional): the penalty to be used for RepeatLastN.
                Defaults to `None`.
            seed (Union[int, None], optional): the seed to be used for generation.
                Defaults to `None`.
            stop (Union[str, None], optional): the stop token to be used for generation. If `None`,
                no stop token will be used. Defaults to `None`.
            tfs_z (Union[float, None], optional): the z value to be used for TFS.
                Defaults to `None`.
            prompt_format (Union[SupportedFormats, None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`..

        Raises:
            ValueError: if the model is not available.
            ValueError: if the Ollama API request failed.

        Examples:
            >>> from distilabel.tasks import TextGenerationTask
            >>> from distilabel.llm import OllamaLLM
            >>> llm = OllamaLLM(model="notus", task=TextGenerationTask())
            >>> llm.generate([{"input": "What's the capital of Spain?"}])
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.mirostat = mirostat
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.num_ctx = num_ctx
        self.num_gqa = num_gqa
        self.num_gpu = num_gpu
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self.stop = stop
        self.tfs_z = tfs_z

        self._api_available()
        self._api_model_available()

    @property
    def model_name(self) -> str:
        """Returns the name of the Ollama model."""
        return self.model

    def _api_available(self):
        """Checks if the Ollama API is available."""
        try:
            ollama.list()
        except ollama.ResponseError as e:
            raise ValueError(
                f"Could not connect to Ollama at {self.OLLAMA_HOST}. Check https://github.com/ollama/ollama-python/tree/main for deployment guide."
            ) from e

    def _api_model_available(self):
        """Checks if the Ollama model is available"""
        try:
            ollama.show(self.model)
        except ollama.ResponseError as e:
            raise ValueError(
                f"Model {self.model} is not available. Run `ollama run {self.model}` to serve the model."
            ) from e

    @retry(
        retry=retry_if_exception_type(_OLLAMA_API_RETRY_ON_EXCEPTIONS),
        stop=stop_after_attempt(_OLLAMA_API_STOP_AFTER_ATTEMPT),
        wait=wait_random_exponential(
            multiplier=_OLLAMA_API_WAIT_RANDOM_EXPONENTIAL_MULTIPLIER,
            max=_OLLAMA_API_WAIT_RANDOM_EXPONENTIAL_MAX,
        ),
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
    )
    def _text_generation_with_backoff(
        self, prompt: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generates text using the Ollama API with backoff."""
        try:
            return ollama.chat(
                model=self.model,
                messages=prompt,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "mirostat": self.mirostat,
                    "mirostat_eta": self.mirostat_eta,
                    "mirostat_tau": self.mirostat_tau,
                    "num_ctx": self.num_ctx,
                    "num_gqa": self.num_gqa,
                    "num_gpu": self.num_gpu,
                    "repeat_last_n": self.repeat_last_n,
                    "repeat_penalty": self.repeat_penalty,
                    "seed": self.seed,
                    "stop": self.stop,
                    "tfs_z": self.tfs_z,
                },
            )
        except ollama.ResponseError as e:
            if e.status_code >= 500:
                raise
            else:
                raise ValueError(
                    f"Ollama API request failed with status_code {e.status_code}."
                ) from e

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "model": self.model,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gqa": self.num_gqa,
                "num_gpu": self.num_gpu,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "seed": self.seed,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
            },
        )

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List[LLMOutput]]:
        prompts = self._generate_prompts(inputs, default_format="openai")
        outputs = []
        for prompt in prompts:
            responses = [
                self._text_generation_with_backoff(prompt=prompt)
                for _ in range(num_generations)
            ]
            output = []
            for response in responses:
                raw_output = response.get("message", {}).get("content")  # type: ignore
                try:
                    parsed_response = self.task.parse_output(raw_output.strip())
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=raw_output,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
