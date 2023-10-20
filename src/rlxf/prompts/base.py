from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PromptTemplate(ABC):
    system_prompt: str

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Any:
        pass
