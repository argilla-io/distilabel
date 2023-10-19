from abc import ABC, abstractmethod
from typing import Any, Generator


class LLM(ABC):
    @abstractmethod
    def batch_generate(
        self, prompts: list[str], responses: list[list[str]] | None = None
    ) -> Generator[list[str], None, None]:
        pass

    @classmethod
    @abstractmethod
    def as_generator(cls, *args: Any, **kwargs: Any) -> "LLM":
        pass

    @classmethod
    @abstractmethod
    def as_ranker(cls, *args: Any, **kwargs: Any) -> "LLM":
        pass

    def _batch_iterator(
        self, prompts: list[str], responses: list[list[str]] | None = None
    ) -> Generator[tuple[str, int | list[str]], None, None]:
        yield from zip(
            prompts, responses if responses is not None else range(len(prompts))
        )
