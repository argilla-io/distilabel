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

from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _OPENAI_AVAILABLE

if _OPENAI_AVAILABLE:
    from openai import OpenAI

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class OpenAILLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str = "gpt-3.5-turbo",
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
        """Initializes the OpenAILLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "gpt-3.5-turbo".
            client (Union[OpenAI, None], optional): an OpenAI client to be used for generation.
                If `None`, a new client will be created. Defaults to `None`.
            openai_api_key (Union[str, None], optional): the OpenAI API key to be used for generation.
                If `None`, the `OPENAI_API_KEY` environment variable will be used. Defaults to `None`.
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
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import OpenAILLM
            >>> task = Task()
            >>> llm = OpenAILLM(model="gpt-3.5-turbo", task=task)
        """
        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "`OpenAILLM` cannot be used as `openai` is not installed, please "
                " install it with `pip install openai`."
            )

        self.max_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

        self.client = client or OpenAI(api_key=openai_api_key, max_retries=6)

        assert (
            model in self.available_models
        ), f"Provided `model` is not available in your OpenAI account, available models are {self.available_models}"
        self.model = model

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_tokens": self.max_tokens,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in your OpenAI account."""
        return [model.id for model in self.client.models.list().data]

    @property
    def model_name(self) -> str:
        """Returns the name of the OpenAI model."""
        return self.model

    def _generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[List[LLMOutput]]:
        """Generates `num_generations` for each input in `inputs`.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each
                input. Defaults to 1.

        Returns:
            List[List[LLMOutput]]: the generated outputs.
        """
        prompts = self._generate_prompts(
            inputs, default_format="openai", expected_output_type=list
        )
        outputs = []
        for prompt in prompts:
            chat_completions = self.client.chat.completions.create(
                messages=prompt,
                model=self.model,
                n=num_generations,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=50,
            )

            output = []
            for chat_completion in chat_completions.choices:
                try:
                    parsed_response = self.task.parse_output(
                        chat_completion.message.content.strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    parsed_response = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=chat_completion.message.content,
                        parsed_output=parsed_response,
                    )
                )
            outputs.append(output)
        return outputs
