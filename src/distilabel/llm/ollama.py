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

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union
from urllib import error, request

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
        max_new_tokens: int = None,  # num_predict
        temperature: Union[float, None] = None,
        top_k: Union[int, None] = None,
        top_p: Union[float, None] = None,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """
        Initializes the OllamaLLM class and align with https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

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
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
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
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import OllamaLLM
            >>> task = Task()
            >>> llm = OllamaLLM(model="notus", task=task)
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

        self._api_available()
        self._api_model_available()

    @property
    def model_name(self) -> str:
        """Returns the name of the Ollama model."""
        return self.model

    def _api_available(self):
        """calls GET {OLLAMA_HOST}"""
        msg = f"Could not connect to Ollama as {self.OLLAMA_HOST}. Check https://github.com/jmorganca/ollama for deployment guide."
        try:
            response = request.urlopen(self.OLLAMA_HOST)
            if response.getcode() != 200:
                raise Exception
        except Exception as e:
            raise ValueError(msg) from e

    def _api_model_available(self):
        msg = f"Model {self.model} is not available. Run `ollama run {self.model}` to serve the model."
        try:
            self._text_generation_with_backoff(
                prompt=[{"role": "user", "content": "hi"}], max_tokens=1
            )
        except Exception as e:
            raise ValueError(msg) from e

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
    def _text_generation_with_backoff(self, prompt: str, **kwargs) -> str:
        """Calls POST {OLLAMA_HOST}/api/chat"""
        # Request payload
        payload = {
            "model": self.model,
            "messages": prompt,
            "stream": False,
        }
        options = {
            "num_predict": kwargs.get("max_new_tokens") or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        # remove None values
        options = {k: v for k, v in options.items() if v is not None}
        payload["options"] = options

        # Convert payload to JSON
        data = json.dumps(payload).encode("utf-8")

        # Create the request
        url = f"{self.OLLAMA_HOST}/api/chat"
        req = request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with request.urlopen(req) as response:
            # Check if the request was successful (status code 200)
            if response.getcode() == 200:
                # Parse and return the response JSON
                return json.loads(response.read().decode("utf-8"))
            elif response.getcode() >= 500:
                # If the request failed, try again with backoff
                raise error.HTTPError(
                    url=url,
                    code=response.getcode(),
                    msg=f"Server Error {response.getcode()}",
                    hdrs=response.getheaders(),
                    fp=None,
                )
            else:
                raise ValueError(
                    f"Ollama API request failed with status_code {response.getcode()}."
                )

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
                raw_output = response.get("message", {}).get("content")
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
