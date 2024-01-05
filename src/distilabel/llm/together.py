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

import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _TOGETHER_AVAILABLE

if _TOGETHER_AVAILABLE:
    import together
    from together.types import TogetherResponse

if TYPE_CHECKING:
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats


logger = get_logger()


class TogetherInferenceLLM(LLM):
    def __init__(
        self,
        task: "Task",
        model: str,
        api_key: Union[str, None] = None,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        stop: Union[List[str], None] = None,
        logprobs: int = 0,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the OpenAILLM class.

        Args:
            task (Task): the task to be performed by the LLM.
            model (str, optional): the model to be used for generation. Defaults to "gpt-3.5-turbo".
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
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
        if not _TOGETHER_AVAILABLE:
            raise ImportError(
                "`OpenAILLM` cannot be used as `openai` is not installed, please "
                " install it with `pip install openai`."
            )

        together.api_key = api_key or os.getenv("TOGETHER_API_KEY", None)
        if together.api_key is None:
            raise ValueError(
                "No `api_key` provided, please provide one or set the `TOGETHER_API_KEY` "
                "environment variable."
            )

        super().__init__(
            task=task,
            num_threads=num_threads,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        assert (
            model in self.available_models
        ), f"Provided `model` is not available in Together Inference, available models are {self.available_models}"
        self.model = model

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.stop = stop
        self.logprobs = logprobs

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "repetition_penalty": self.repetition_penalty,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stop": self.stop,
                "logprobs": self.logprobs,
            },
        )

    @cached_property
    def available_models(self) -> List[str]:
        """Returns the list of available models in Together Inference."""
        # TODO: exclude the image models
        return [model["name"] for model in together.Models.list()]

    @property
    def model_name(self) -> str:
        """Returns the name of the Together Inference model."""
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
        prompts = self._generate_prompts(inputs, default_format=None)
        outputs = []
        for prompt in prompts:
            batch = []
            for _ in range(num_generations):
                output = together.Complete.create(
                    prompt=prompt,
                    model=self.model,
                    max_tokens=self.max_new_tokens,
                    stop=self.stop,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    logprobs=self.logprobs,
                )
                if isinstance(output, TogetherResponse) and output.choices is not None:
                    for choice in output.choices:
                        try:
                            parsed_response = self.task.parse_output(
                                choice.text.strip()
                            )
                        except Exception as e:
                            logger.error(
                                f"Error parsing Together Inference response: {e}"
                            )
                            parsed_response = None
                        batch.append(
                            LLMOutput(
                                model_name=self.model_name,
                                prompt_used=prompt,
                                raw_output=choice.text,
                                parsed_output=parsed_response,
                            )
                        )
            if len(batch) > 0:
                outputs.append(batch)
        return outputs
