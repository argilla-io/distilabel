import importlib.resources as importlib_resources
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, List, Union

from jinja2 import Template
from pydantic import BaseModel


def get_template(template_name: str) -> str:
    return str(importlib_resources.files("rlxf") / "prompts/templates" / template_name)


class PromptTemplate(BaseModel, ABC):
    system_prompt: str

    __jinja2_template__: Union[str, None] = None

    @cached_property
    def template(self) -> "Template":
        if self.__jinja2_template__ is None:
            raise ValueError(
                "You must provide a `__jinja2_template__` attribute to your PromptTemplate subclass."
            )

        return Template(open(self.__jinja2_template__).read())

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def _parse_output(self, output: str) -> Any:
        pass

    def parse_output(self, output: str) -> Any:
        try:
            return self._parse_output(output)
        except Exception:
            return {}

    @property
    @abstractmethod
    def input_args_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def output_args_names(self) -> List[str]:
        pass

    def validate_dataset(self, columns_in_dataset: List[str]) -> None:
        for input_arg_name in self.input_args_names:
            if input_arg_name not in columns_in_dataset:
                raise KeyError(
                    f"LLM expects a column named '{input_arg_name}' in the provided"
                    " dataset, but it was not found."
                )
