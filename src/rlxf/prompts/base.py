from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any


class PromptTemplate(BaseModel, ABC):
    system_prompt: str

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Any:
        pass
