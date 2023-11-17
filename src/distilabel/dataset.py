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

from typing import TYPE_CHECKING, Any, Union

from datasets import Dataset

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False


if TYPE_CHECKING:
    from argilla import FeedbackDataset

    from distilabel.tasks.base import Task


class CustomDataset(Dataset):
    task: Union["Task", None] = None

    def to_argilla(self, **kwargs: Any) -> "FeedbackDataset":
        if _argilla_installed is False:
            raise ImportError(
                "The argilla library is not installed. Please install it with `pip install argilla`."
            )
        if self.task is None:
            raise ValueError(
                "The task is not set. Please set it with `dataset.task = <task>`."
            )

        rg_dataset = rg.FeedbackDataset(
            fields=self.task.to_argilla_fields(dataset_row=self[0], **kwargs),
            questions=self.task.to_argilla_questions(dataset_row=self[0], **kwargs),
            metadata_properties=self.task.to_argilla_metadata_properties(dataset_row=self[0], **kwargs)
        )
        for dataset_row in self:
            if any(
                dataset_row[input_arg_name] is None  # type: ignore
                for input_arg_name in self.task.input_args_names
            ):
                continue
            rg_dataset.add_records(
                self.task.to_argilla_record(dataset_row=dataset_row, **kwargs)  # type: ignore
            )
        return rg_dataset


class PreferenceDataset(CustomDataset):
    def to_argilla(self, group_ratings_as_ranking: bool = False) -> "FeedbackDataset":
        return super().to_argilla(group_ratings_as_ranking=group_ratings_as_ranking)