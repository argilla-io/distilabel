import importlib.resources as importlib_resources
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union

from jinja2 import Template
from pydantic import BaseModel

from ultralabel.tasks.utils import Prompt

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )


def get_template(template_name: str) -> str:
    return str(
        importlib_resources.files("ultralabel") / "tasks/_templates" / template_name
    )


class Argilla:
    def to_argilla_fields(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedFieldTypes"]:
        raise NotImplementedError(
            "`to_argilla_fields` is not implemented, if you want to export your dataset as an Argilla dataset you will need to implement this method."
        )

    def to_argilla_questions(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedQuestionTypes"]:
        raise NotImplementedError(
            "`to_argilla_questions` is not implemented, if you want to export your dataset as an Argilla dataset you will need to implement this method."
        )

    def to_argilla_record(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> "FeedbackRecord":
        raise NotImplementedError(
            "`to_argilla_record` is not implemented, if you want to export your dataset as an Argilla dataset you will need to implement this method."
        )


# TODO: `pydantic` dependency is not needed, we are safe to remove it
class Task(BaseModel, ABC, Argilla):
    system_prompt: str
    task_description: Union[str, None] = None

    __jinja2_template__: Union[str, None] = None

    @property
    def template(self) -> "Template":
        if self.__jinja2_template__ is None:
            raise ValueError(
                "You must provide a `__jinja2_template__` attribute to your Task subclass."
            )

        return Template(open(self.__jinja2_template__).read())

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> Union[Prompt, Any]:
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Any:
        pass

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
