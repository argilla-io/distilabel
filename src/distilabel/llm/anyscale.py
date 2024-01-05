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

from typing import TYPE_CHECKING, Callable, Union

from distilabel.llm.base import LLM
from distilabel.llm.openai import OpenAILLM
from distilabel.logger import get_logger
from distilabel.utils.imports import _OPENAI_AVAILABLE

if _OPENAI_AVAILABLE:
    from openai import OpenAI

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class AnyscaleLLM(OpenAILLM):
    def __init__(
        self,
        task: "Task",
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        client: Union["OpenAI", None] = None,
        openai_api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the AnyscaleLLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "HuggingFaceH4/zephyr-7b-beta".
            client (Union[OpenAI, None], optional): an OpenAI client to be used for generation.
                If `None`, a new client will be created. Defaults to `None`.
            openai_api_key (Union[str, None], optional): the OpenAI API key to be used for generation.
                If `None`, the `OPENAI_API_KEY` environment variable will be used. Defaults to `None`.
                Visit "https://docs.endpoints.anyscale.com/guides/authenticate/" for more information.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            frequency_penalty (float, optional): the frequency penalty to be used for generation.
                Defaults to 0.0.
            presence_penalty (float, optional): the presence penalty to be used for generation.
                Defaults to 0.0.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 1.0.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 1.0.
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
                Defaults to `None`.

        Raises:
            AssertionError: if the provided `model` is not available in your OpenAI account.

        Examples:
            >>> import os
            >>> from distilabel.tasks import TextGenerationTask
            >>> from distilabel.llm import AnyscaleLLM
            >>> llm = AnyscaleLLM(model="HuggingFaceH4/zephyr-7b-beta", task=TextGenerationTask(), openai_api_key=os.getenv("OPENAI_API_KEY", None))
            >>> llm.generate([{"input": "What's the capital of Spain?"}])
            >>> [[{'model_name': 'HuggingFaceH4/zephyr-7b-beta',
            ...    'prompt_used': [{'role': 'system',
            ...        'content': "You...
        """
        LLM.__init__(
            self,
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "`AnyscaleLLM` cannot be used as `openai` is not installed, please "
                " install it with `pip install openai`."
            )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.client = client or OpenAI(
            api_key=openai_api_key,
            max_retries=6,
            base_url="https://api.endpoints.anyscale.com/v1",
        )

        assert (
            model in self.available_models
        ), f"Provided `model` is not available in your Anyscale account, available models are {self.available_models}"
        self.model = model
