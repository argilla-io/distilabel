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

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

if TYPE_CHECKING:
    from distilabel.llm.utils import LLMOutput
    from distilabel.tasks.base import Task


class LLM(ABC):
    def __init__(
        self,
        task: Task,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        self.task = task

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

        self.formatting_fn = formatting_fn

    def __del__(self) -> None:
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

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
                progress_callback_func()

        if self.thread_pool_executor is not None:
            futures = []
            for input in inputs:
                future = self.thread_pool_executor.submit(
                    self._generate, input, num_generations
                )
                future.add_done_callback(lambda future: _progress())
                futures.append(future)
            return futures

        generations = []
        for input in inputs:
            generations.append(self._generate(input, num_generations))
            _progress()
        return generations

    @property
    def return_futures(self) -> bool:
        return self.thread_pool_executor is not None
