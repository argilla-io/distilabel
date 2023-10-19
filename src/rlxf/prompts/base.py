from abc import ABC, abstractmethod
from typing import Any


class Prompt(ABC):
    @staticmethod
    @abstractmethod
    def chat_format(instruction: str, *args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def rank_format(prompt: str, responses: list[str]) -> Any:
        pass
