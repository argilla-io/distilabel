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
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict

from distilabel.tasks.base import Task
from distilabel.tasks.mixins import RatingToArgillaMixin

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord


@dataclass
class CritiqueTask(RatingToArgillaMixin, Task):
    """A `Task` for critique / judge tasks.

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
        return ["critique", "score"]

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        score_column: str = "score",
        critique_column: str = "critique",
        score_values: Optional[List[int]] = None,
    ) -> "FeedbackDataset":
        return super().to_argilla_dataset(
            dataset_row=dataset_row,
            generations_column=generations_column,
            ratings_column=score_column,
            rationale_column=critique_column,
            ratings_values=score_values or [1, 2, 3, 4, 5],
        )

    def to_argilla_record(
        self,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        score_column: str = "score",
        critique_column: str = "critique",
    ) -> Union["FeedbackRecord", List["FeedbackRecord"]]:
        return super().to_argilla_record(
            dataset_row=dataset_row,
            generations_column=generations_column,
            ratings_column=score_column,
            rationale_column=critique_column,
        )


class CritiqueTaskOutput(TypedDict):
    """A `TypedDict` matching the output format of any `CritiqueTask`."""

    score: float
    critique: str
