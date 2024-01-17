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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union

from jinja2 import Template

from distilabel.tasks.prompt import Prompt
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    from argilla.client.feedback.integrations.sentencetransformers import (
        SentenceTransformersExtractor,
    )

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord
    from argilla.client.feedback.integrations.sentencetransformers import (
        SentenceTransformersExtractor,
    )


def get_template(template_name: str) -> str:
    return str(
        importlib_resources.files("distilabel") / "tasks/_templates" / template_name
    )


class Task(ABC):
    """Abstract class used to define the methods required to create a `Task`, to be used
    within an `LLM`.

    Args:
        system_prompt (str): the system prompt to be used for generation.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.

    Raises:
        ValueError: if the `__jinja2_template__` attribute is not provided.
    """

    system_prompt: str
    task_description: Union[str, None] = None

    __jinja2_template__: Union[str, None] = None
    __type__: Union[Literal["generation", "labelling"], None] = None

    def __rich_repr__(self) -> Generator[Any, None, None]:
        yield "system_prompt", self.system_prompt
        yield "task_description", self.task_description
        yield "input_args_names", self.input_args_names
        yield "output_args_names", self.output_args_names

    @property
    def template(self) -> "Template":
        if self.__jinja2_template__ is None:
            raise ValueError(
                "You must provide a `__jinja2_template__` attribute to your Task subclass."
            )

        return Template(open(self.__jinja2_template__).read())

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> Prompt:
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
        """Validates that the dataset contains the required columns for the task.

        Args:
            columns_in_dataset (List[str]): the columns in the dataset.

        Raises:
            KeyError: if the dataset does not contain the required columns.
        """
        for input_arg_name in self.input_args_names:
            if input_arg_name not in columns_in_dataset:
                raise KeyError(
                    f"LLM expects a column named '{input_arg_name}' in the provided"
                    " dataset, but it was not found."
                )

    def to_argilla_dataset(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> "FeedbackDataset":
        raise NotImplementedError(
            "`to_argilla_dataset` is not implemented, if you want to export your dataset as an Argilla"
            " `FeedbackDataset` you will need to implement this method first."
        )

    def to_argilla_record(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Union["FeedbackRecord", List["FeedbackRecord"]]:
        raise NotImplementedError(
            "`to_argilla_record` is not implemented, if you want to export your dataset as an Argilla"
            " `FeedbackDataset` you will need to implement this method first."
        )

    def add_vectors_to_argilla_dataset(
        self,
        dataset: Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"],
        vector_strategy: Union[bool, "SentenceTransformersExtractor"],
    ) -> Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"]:
        if _ARGILLA_AVAILABLE and vector_strategy:
            try:
                if isinstance(vector_strategy, SentenceTransformersExtractor):
                    ste = vector_strategy

                elif vector_strategy is True:
                    ste = SentenceTransformersExtractor(
                        model="en",
                        show_progress=True,
                    )
                else:
                    raise ValueError(
                        "The `vector_strategy` must be either `True` or a `SentenceTransformersExtractor` instance."
                    )

                dataset = ste.update_dataset(dataset=dataset)
            except Exception as e:
                warnings.warn(
                    f"An error occurred while adding vectors to the dataset: {e}",
                    stacklevel=2,
                )

        elif not _ARGILLA_AVAILABLE and vector_strategy:
            warnings.warn(
                "An error occurred while adding vectors to the dataset: "
                "The `argilla`/`sentence-transformers` packages are not installed or the installed version is not compatible with the"
                " required version. If you want to add vectors to your dataset, please run `pip install 'distilabel[vectors]'`.",
                stacklevel=2,
            )
        else:
            pass
        return dataset

    # Renamed to _to_argilla_record instead of renaming `to_argilla_record` to protected, as that would
    # imply more breaking changes.
    def _to_argilla_record(  # noqa: C901
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Union["FeedbackRecord", List["FeedbackRecord"]]:
        column_names = list(dataset_row.keys())
        if self.__type__ is None or self.__type__ == "generation":
            required_column_names = self.input_args_names + self.output_args_names
        elif self.__type__ == "labelling":
            required_column_names = self.output_args_names
        else:
            raise ValueError("The task type is not supported.")

        dataset_rows = [dataset_row]
        if "generation_model" in dataset_row and isinstance(
            dataset_row["generation_model"], list
        ):
            generation_columns = column_names[
                column_names.index("generation_model") : column_names.index(
                    "labelling_model"
                )
                if "labelling_model" in column_names
                else None
            ]
            if any(
                isinstance(nested, list)
                for column_name in list(
                    set(generation_columns)
                    - {
                        "generation_model",
                        "generation_prompt",
                        "raw_generation_response",
                    }
                )
                for nested in dataset_row[column_name]
            ):
                if any(
                    generation_column in required_column_names
                    for generation_column in generation_columns
                ):
                    unwrapped_dataset_rows = []
                    for row in dataset_rows:
                        for idx in range(len(dataset_row["generation_model"])):
                            unwrapped_dataset_row = {}
                            for key, value in row.items():
                                if key in generation_columns:
                                    unwrapped_dataset_row[key] = value[idx]
                                else:
                                    unwrapped_dataset_row[key] = value
                            unwrapped_dataset_rows.append(unwrapped_dataset_row)
                    dataset_rows = unwrapped_dataset_rows

        if "labelling_model" in dataset_row and isinstance(
            dataset_row["labelling_model"], list
        ):
            labelling_columns = column_names[column_names.index("labelling_model") :]
            if any(
                isinstance(nested, list)
                for column_name in list(
                    set(labelling_columns)
                    - {
                        "labelling_model",
                        "labelling_prompt",
                        "raw_labelling_response",
                    }
                )
                for nested in dataset_row[column_name]
            ):
                if any(
                    labelling_column in required_column_names
                    for labelling_column in labelling_columns
                ):
                    unwrapped_dataset_rows = []
                    for row in dataset_rows:
                        for idx in range(len(dataset_row["labelling_model"])):
                            unwrapped_dataset_row = {}
                            for key, value in row.items():
                                if key in labelling_columns:
                                    unwrapped_dataset_row[key] = value[idx]
                                else:
                                    unwrapped_dataset_row[key] = value
                            unwrapped_dataset_rows.append(unwrapped_dataset_row)
                    dataset_rows = unwrapped_dataset_rows

        if len(dataset_rows) == 1:
            return self.to_argilla_record(dataset_rows[0], *args, **kwargs)

        records = []
        for dataset_row in dataset_rows:
            generated_records = self.to_argilla_record(dataset_row, *args, **kwargs)
            if isinstance(generated_records, list):
                records.extend(generated_records)
            else:
                records.append(generated_records)
        return records
