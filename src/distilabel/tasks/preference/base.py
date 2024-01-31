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

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional

from distilabel.tasks.base import Task
from distilabel.tasks.mixins import RatingToArgillaMixin


@dataclass
class PreferenceTask(RatingToArgillaMixin, Task):
    """A `Task` for preference rating tasks.

    Args:
        system_prompt (str): the system prompt to be used for generation.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.
    """

    __type__: ClassVar[Literal["labelling"]] = "labelling"

    @property
    def input_args_names(self) -> List[str]:
        """Returns the names of the input arguments of the task."""
        return ["input", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        """Returns the names of the output arguments of the task."""
        return ["rating", "rationale"]


@dataclass
class PreferenceTaskNoRationale(PreferenceTask):
    """A `Task` for preference rating tasks, that only returns the rating, without the rationale.

    Args:
        system_prompt (str): the system prompt to be used for generation.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.
    """

    @property
    def output_args_names(self) -> List[str]:
        """Returns the names of the output arguments of the task."""
        return ["rating"]

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        ratings_column: str = "rating",
        ratings_values: Optional[List[int]] = None,
    ):
        """Same definition from the parent class, but removing the rationale column."""
        return super().to_argilla_dataset(
            dataset_row,
            generations_column=generations_column,
            ratings_column=ratings_column,
            rationale_column=None,
            ratings_values=ratings_values,
        )

    def to_argilla_record(  # noqa: C901
        self,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        ratings_column: str = "rating",
        ratings_values: Optional[List[int]] = None,
    ):
        """Same definition from the parent class, but removing the rationale column."""
        return super().to_argilla_record(
            dataset_row,
            generations_column=generations_column,
            ratings_column=ratings_column,
            rationale_column=None,
            ratings_values=ratings_values,
        )
