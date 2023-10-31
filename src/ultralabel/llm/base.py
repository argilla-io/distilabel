from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from ultralabel.tasks.base import Task


class LLM(ABC):
    def __init__(
        self,
        task: Task,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_threads: Union[int, None] = None,
        formatting_fn: Union[Callable[..., str], None] = None,
    ) -> None:
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.thread_pool_executor = (
            ThreadPoolExecutor(max_workers=num_threads)
            if num_threads is not None
            else None
        )

        self.formatting_fn = formatting_fn

    def __del__(self) -> None:
        if self.thread_pool_executor is not None:
            self.thread_pool_executor.shutdown()

    @abstractmethod
    def _generate(self, **kwargs: Any) -> Tuple[Any, List[Any]]:
        pass

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        progress_callback_func: Union[Callable, None] = None,
    ) -> Any:
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
