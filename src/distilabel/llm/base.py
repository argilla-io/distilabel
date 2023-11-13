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
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Type, Union

from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task


class LLM(ABC):
    def __init__(
        self,
        task: Task,
        num_threads: Union[int, None] = None,
        prompt_format: Union[
            Literal["llama2", "openai", "chatml", "zephyr"], None
        ] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        self.task = task

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

        self.prompt_format = prompt_format
        self.prompt_formatting_fn = prompt_formatting_fn

    def __del__(self) -> None:
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

    def _generate_prompts(
        self,
        inputs: List[Dict[str, Any]],
        default_format: Union[
            Literal["llama2", "openai", "chatml", "zephyr"], None
        ] = None,
        expected_output_type: Type = str,
    ) -> List[Any]:
        prompts = []
        for input in inputs:
            prompt = self.task.generate_prompt(**input)
            if not isinstance(prompt, Prompt) and self.prompt_formatting_fn is not None:
                warnings.warn(
                    "The method `generate_prompt` is not returning a `Prompt` class but a prompt"
                    f" of `type={type(prompt)}`, meaning that a pre-formatting has already been"
                    " applied in the `task.generate_prompt` method, so the usage of a `prompt_formatting_fn`"
                    " is discouraged.",
                    UserWarning,
                    stacklevel=2,
                )
                prompt = self.prompt_formatting_fn(prompt)
            elif isinstance(prompt, Prompt) and self.prompt_formatting_fn is None:
                if self.prompt_format is not None or default_format is not None:
                    prompt = prompt.format_as(format=self.prompt_format or default_format)  # type: ignore
                else:
                    warnings.warn(
                        "No `prompt_format` has been specified and no `default_format` is set, so"
                        " the prompt will be concatenated with a line-break and no specific formatting"
                        " by default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    prompt = f"{prompt.system_prompt}\n{prompt.formatted_prompt}"
            if not isinstance(prompt, expected_output_type):
                raise ValueError(
                    f"The provided `prompt={prompt}` is of `type={type(prompt)}`, but it must be of"
                    f" `type={expected_output_type}`, so make sure that `task.generate_prompt` returns"
                    f" a `{expected_output_type}` or that the `formatting_fn` formats the prompt as a "
                    f" `{expected_output_type}`."
                )
            prompts.append(prompt)
        return prompts

    @abstractmethod
    def _generate(self, **kwargs: Any) -> List["LLMOutput"]:
        pass

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Union[List[Future[List["LLMOutput"]]], List[List["LLMOutput"]]]:
        def _progress():
            if progress_callback_func is not None:
                progress_callback_func(advance=num_generations * len(inputs))

        if self.thread_pool_executor is not None:
            futures = []
            for input in inputs:
                future = self.thread_pool_executor.submit(
                    self._generate, [input], num_generations
                )
                future.add_done_callback(lambda future: _progress())
                futures.append(future)
            return futures

        generations = self._generate(inputs, num_generations)
        _progress()
        return generations

    @property
    def return_futures(self) -> bool:
        return self.thread_pool_executor is not None
