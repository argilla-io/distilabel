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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.utils.imports import _LLAMA_CPP_AVAILABLE

if TYPE_CHECKING:
    from llama_cpp import Llama

    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats

logger = get_logger()


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: "Llama",
        task: "Task",
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        prompt_format: Union[SupportedFormats, None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the LlamaCppLLM class.

        Args:
            model (Llama): the llama-cpp model to be used.
            task (Task): the task to be performed by the LLM.
            max_new_tokens (int, optional): the maximum number of tokens to be generated.
                Defaults to 128.
            temperature (float, optional): the temperature to be used for generation.
                Defaults to 0.8.
            top_p (float, optional): the top-p value to be used for generation.
                Defaults to 0.95.
            top_k (int, optional): the top-k value to be used for generation.
                Defaults to 40.
            repeat_penalty (float, optional): the repeat penalty to be used for generation.
                Defaults to 1.1.
            prompt_format (Union[SupportedFormats, None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`.

        Examples:
            >>> from llama_cpp import Llama
            >>> from distilabel.tasks.text_generation import TextGenerationTask as Task
            >>> from distilabel.llm import LlamaCppLLM
            >>> model = Llama(model_path="path/to/model")
            >>> task = Task()
            >>> llm = LlamaCppLLM(model=model, task=task)
        """
        super().__init__(
            task=task,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        if not _LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "`LlamaCppLLM` cannot be used as `llama_cpp` is not installed, please "
                " install it with `pip install llama-cpp-python`."
            )

        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        self.model = model

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield from super().__rich_repr__()
        yield (
            "parameters",
            {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
            },
        )

    @property
    def model_name(self) -> str:
        """Returns the name of the llama-cpp model, which is the same as the model path."""
        return self.model.model_path

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
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
            inputs, default_format=None, expected_output_type=str
        )
        outputs = []
        for prompt in prompts:
            output = []
            for _ in range(num_generations):
                raw_output = self.model.create_completion(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repeat_penalty=self.repeat_penalty,
                )
                try:
                    parsed_output = self.task.parse_output(
                        raw_output["choices"][0]["text"].strip()
                    )
                except Exception as e:
                    logger.error(f"Error parsing llama-cpp output: {e}")
                    parsed_output = None
                output.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used=prompt,
                        raw_output=raw_output,
                        parsed_output=parsed_output,
                    )
                )
            outputs.append(output)
        return outputs
