from abc import ABC, abstractmethod
from typing import Any


class Prompt(ABC):
    @staticmethod
    @abstractmethod
    def chat_format(instruction: str, *args: Any, **kwargs: Any) -> str:
        pass

    @staticmethod
    @abstractmethod
    def rank_format(prompt: str, responses: list[str]) -> str:
        pass
