from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class PromptTemplate(BaseModel, ABC):
    system_prompt: str

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Any:
        pass

    @property
    @abstractmethod
    def input_args_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def output_args_names(self) -> list[str]:
        pass
