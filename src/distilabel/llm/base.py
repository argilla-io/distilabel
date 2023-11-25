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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Type, Union

from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task
    from distilabel.tasks.prompt import SupportedFormats


class LLM(ABC):
    def __init__(
        self,
        task: Task,
        num_threads: Union[int, None] = None,
        prompt_format: Union["SupportedFormats", None] = None,
        prompt_formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        """Initializes the LLM base class.

        Note:
            This class is intended to be used internally, but you anyone can still create
            a subclass, implement the `abstractmethod`s and use it.

        Args:
            task (Task): the task to be performed by the LLM.
            num_threads (Union[int, None], optional): the number of threads to be used
                for parallel generation. If `None`, no parallel generation will be performed.
                Defaults to `None`.
            prompt_format (Union["SupportedFormats", None], optional): the format to be used
                for the prompt. If `None`, the default format of the task will be used, available
                formats are `openai`, `chatml`, `llama2`, `zephyr`, and `default`. Defaults to `None`,
                but `default` (concatenation of `system_prompt` and `formatted_prompt` with a line-break)
                will be used if no `prompt_formatting_fn` is provided.
            prompt_formatting_fn (Union[Callable[..., str], None], optional): a function to be
                applied to the prompt before generation. If `None`, no formatting will be applied.
                Defaults to `None`.
        """
        self.task = task

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

        self.prompt_format = prompt_format
        self.prompt_formatting_fn = prompt_formatting_fn

    def __del__(self) -> None:
        """Shuts down the thread pool executor if it is not `None`."""
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task.__class__.__name__}, num_threads={self.thread_pool_executor._max_workers}, promp_format='{self.prompt_format}', model='{self.model_name}')"

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield "task", self.task
        yield "num_threads", self.thread_pool_executor._max_workers
        yield "prompt_format", self.prompt_format
        if self.prompt_formatting_fn is not None:
            args = f"({', '.join(self.prompt_formatting_fn.__code__.co_varnames)})"
            representation = self.prompt_formatting_fn.__name__ + args
            yield "prompt_formatting_fn", representation
        yield "model", self.model_name

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def _generate_prompts(
        self,
        inputs: List[Dict[str, Any]],
        default_format: Union["SupportedFormats", None] = None,
        expected_output_type: Type = str,
    ) -> List[Any]:
        """Generates the prompts to be used for generation.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            default_format (Union["SupportedFormats", None], optional): the default format to be used
                for the prompt if no `prompt_format` is specified. Defaults to `None`.
            expected_output_type (Type, optional): the expected type of the prompt. Defaults to `str`.

        Returns:
            List[Any]: the generated prompts.

        Raises:
            ValueError: if the generated prompt is not of the expected type.
        """
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
                    prompt = prompt.format_as(
                        format=self.prompt_format or default_format  # type: ignore
                    )
                else:
                    warnings.warn(
                        "No `prompt_format` has been specified and no `default_format` is set, so"
                        " the prompt will be concatenated with a line-break and no specific formatting"
                        " by default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    prompt = prompt.format_as(format="default")
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
    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        pass

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Union[List[Future[List["LLMOutput"]]], List[List["LLMOutput"]]]:
        """Generates the outputs for the given inputs using the LLM.

        Args:
            inputs (List[Dict[str, Any]]): the inputs to be used for generation.
            num_generations (int, optional): the number of generations to be performed for each input.
                Defaults to `1`.
            progress_callback_func (Union[Callable, None], optional): a function to be called at each
                generation step. Defaults to `None`.

        Returns:
            Union[List[Future[List["LLMOutput"]]], List[List["LLMOutput"]]]: the generated outputs.
        """

        def _progress():
            if progress_callback_func is not None:
                advance = (
                    num_generations * len(inputs)
                    if not self.return_futures
                    else num_generations
                )
                progress_callback_func(advance=advance)

        if self.thread_pool_executor is not None:
            futures = []
            for input in inputs:
                future = self.thread_pool_executor.submit(
                    self._generate, [input], num_generations
                )
                future.add_done_callback(lambda _: _progress())
                futures.append(future)
            return futures

        generations = self._generate(inputs, num_generations)
        _progress()
        return generations

    @property
    def return_futures(self) -> bool:
        """Returns whether the LLM returns futures or not."""
        return self.thread_pool_executor is not None
