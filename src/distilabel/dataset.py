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

import warnings
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

from datasets import Dataset

from distilabel.utils.dataset import load_task_from_disk, save_task_to_disk
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if TYPE_CHECKING:
    from argilla import FeedbackDataset

    from distilabel.tasks.base import Task


class CustomDataset(Dataset):
    """A custom dataset class that extends from `datasets.Dataset` and is used to generate
    an Argilla `FeedbackDataset` instance from the pre-defined configuration within the task
    provided to `Pipeline.generate`.
    """

    task: Union["Task", None] = None

    def to_argilla(self) -> "FeedbackDataset":
        """Converts the dataset to an Argilla `FeedbackDataset` instance, based on the
        task defined in the dataset as part of `Pipeline.generate`.

        Raises:
            ImportError: if the argilla library is not installed.
            ValueError: if the task is not set.

        Returns:
            FeedbackDataset: the Argilla `FeedbackDataset` instance.
        """
        if not _ARGILLA_AVAILABLE:
            raise ImportError(
                "To use `to_argilla` method is required to have `argilla` installed. "
                "Please install it with `pip install argilla`."
            )

        if self.task is None:
            raise ValueError(
                "The task is not set. Please set it with `dataset.task = <task>`."
            )

        try:
            rg_dataset = self.task.to_argilla_dataset(dataset_row=self[0])  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Error while converting the dataset to an Argilla `FeedbackDataset` instance: {e}"
            ) from e

        # try:
        #     rg_dataset = infer_model_metadata_properties(
        #         hf_dataset=self, rg_dataset=rg_dataset
        #     )
        # except Exception as e:
        #     warnings.warn(
        #         f"Error while adding the model metadata properties: {e}",
        #         UserWarning,
        #         stacklevel=2,
        #     )

        for dataset_row in self:
            if any(
                dataset_row[input_arg_name] is None  # type: ignore
                for input_arg_name in self.task.input_args_names
            ):
                continue
            try:
                rg_dataset.add_records(
                    self.task._to_argilla_record(dataset_row=dataset_row)  # type: ignore
                )  # type: ignore
            except Exception as e:
                warnings.warn(
                    f"Error while converting a row into an Argilla `FeedbackRecord` instance: {e}",
                    UserWarning,
                    stacklevel=2,
                )
        return rg_dataset

    def save_to_disk(self, dataset_path: PathLike, **kwargs: Any) -> None:
        """Saves the datataset to disk, also saving the task.

        Args:
            dataset_path: Path to the dataset.
            **kwargs: Additional arguments to be passed to `datasets.Dataset.save_to_disk`.
        """
        super().save_to_disk(dataset_path, **kwargs)
        if self.task is not None:
            save_task_to_disk(dataset_path, self.task)

    @classmethod
    def load_from_disk(cls, dataset_path: PathLike, **kwargs: Any):
        """Load a CustomDataset from disk, also reading the task.

        Args:
            dataset_path: Path to the dataset, as you would do with a standard Dataset.

        Returns:
            The loaded dataset.
        """
        ds = super().load_from_disk(dataset_path, *kwargs)
        # Dynamically remaps the `datasets.Dataset` to be a `CustomDataset` instance
        ds.__class__ = cls
        task = load_task_from_disk(dataset_path)
        ds.task = task
        return ds


@dataclass
class DatasetCheckpoint:
    """A checkpoint class that contains the information of a checkpoint.

    Args:
        path (Path): The path to the checkpoint.
        save_frequency (int): The frequency at which the checkpoint should be saved
            By default is set to -1 (no checkpoint is saved to disk, but the dataset
            is returned upon failure).
        extra_kwargs (dict[str, Any]): Additional kwargs to be passed to the `save_to_disk` method of the Dataset.
    """

    path: Path = Path.cwd() / "dataset_checkpoint"
    save_frequency: int = -1
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Internal fields to keep track of the number of records generated and when to check.
    _total_checks: int = field(repr=False, default=0)

    def do_checkpoint(self, step: int) -> bool:
        """Determines if a checkpoint should be done.

        Args:
            step (int): The number of records generated.

        Returns:
            bool: Whether a checkpoint should be done.
        """
        if self.save_frequency == -1:
            return False

        if (step - self._total_checks * self.save_frequency) // self.save_frequency:
            self._total_checks += 1
            return True
        return False
