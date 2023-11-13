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

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Union

from distilabel.llm.base import LLM
from distilabel.llm.utils import LLMOutput
from distilabel.logger import get_logger
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from llama_cpp import Llama

    from distilabel.tasks.base import Task

logger = get_logger()


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model: "Llama",
        task: "Task",
        max_new_tokens: int = 128,
        temperature: Union[float, None] = None,
        top_p: Union[float, None] = None,
        top_k: Union[int, None] = None,
        repeat_penalty: Union[float, None] = None,
        prompt_format: Union[
            Literal["llama2", "openai", "chatml", "zephyr"], None
        ] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        super().__init__(
            task=task,
            prompt_format=prompt_format,
            prompt_formatting_fn=prompt_formatting_fn,
        )

        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        self.__generation_attrs = [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repeat_penalty",
        ]

        self.model = model

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
            prompt = prompt.format_as(
                format="llama2" if self.prompt_format is None else self.prompt_format  # type: ignore
            )
        if not isinstance(prompt, str):
            raise ValueError(
                f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be a `str`, make sure that `task.generate_prompt` returns a `str` or that the `formatting_fn` formats the prompt as a `str`."
            )
        generation_kwargs = {}
        for generation_attr in self.__generation_attrs:
            value = getattr(self, generation_attr)
            if value is not None:
                generation_kwargs[generation_attr] = value
        outputs = []
        for _ in range(num_generations):
            raw_output = self.model.create_completion(
                prompt,
                **generation_kwargs,
            )
            try:
                parsed_output = self.task.parse_output(
                    raw_output["choices"][0]["text"].strip()
                )
            except Exception as e:
                logger.error(f"Error parsing llama-cpp output: {e}")
                parsed_output = None
            outputs.append(
                LLMOutput(
                    prompt_used=prompt,
                    raw_output=raw_output,
                    parsed_output=parsed_output,
                )
            )
        return outputs
