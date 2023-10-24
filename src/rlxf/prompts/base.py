from abc import ABC, abstractmethod
from typing import Any, List

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

    def validate_dataset(self, columns_in_dataset: List[str]) -> None:
        for input_arg_name in self.input_args_names:
            if input_arg_name not in columns_in_dataset:
                raise KeyError(
                    f"LLM expects a column named '{input_arg_name}' in the provided"
                    " dataset, but it was not found."
                )
