# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.resources as importlib_resources
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from jinja2 import Template

from distilabel.tasks.prompt import Prompt

try:
    import argilla as rg
except ImportError:
    pass

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )


def get_template(template_name: str) -> str:
    return str(
        importlib_resources.files("distilabel") / "tasks/_templates" / template_name
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

    def _create_argilla_record(
        self,
        fields: Dict[str, Any],
        suggestions: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> rg.FeedbackRecord:
        return rg.FeedbackRecord(
            fields=fields, suggestions=suggestions, metadata=metadata
        )

    def _create_text_field(self, name: str) -> rg.TextField:
        return rg.TextField(name=name)

    def _create_rating_question(
        self, name: str, title: str, values: List[int]
    ) -> rg.RatingQuestion:
        return rg.RatingQuestion(name=name, title=title, values=values)

    def _create_text_question(self, name: str, title: str) -> rg.TextQuestion:
        return rg.TextQuestion(name=name, title=title)

    def _create_metadata_property(
        self, name: str, property_type: str
    ) -> Union[rg.IntegerMetadataProperty, rg.FloatMetadataProperty]:
        if property_type == "integer":
            return rg.IntegerMetadataProperty(name=name)
        elif property_type == "float":
            return rg.FloatMetadataProperty(name=name)
        else:
            raise ValueError(f"Invalid property type: {property_type}")

    def _update_fields_and_metadata(
        self,
        fields: Dict[str, str],
        metadata: Dict[str, Any],
        arg_name: str,
        value: str,
        idx: int = None,
    ):
        suffix = f"-{idx}" if idx is not None else ""
        fields[f"{arg_name}{suffix}"] = value.strip()
        metadata[f"length-{arg_name}{suffix}"] = len(value.strip())

    def _process_dataset_row(
        self, dataset_row: Dict[str, Any], process_function: Callable
    ) -> List:
        processed_items = []
        for arg_name in self.input_args_names:
            self._check_argument_exists(dataset_row, arg_name)
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    processed_items.append(process_function(f"{arg_name}-{idx}"))
            elif isinstance(dataset_row[arg_name], str):
                processed_items.append(process_function(arg_name))
        return processed_items


class Task(ABC, Argilla):
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