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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from datasets import Dataset

from distilabel.utils.dataset import load_task_from_disk, save_task_to_disk
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

    from distilabel.tasks.base import Task


class CustomDataset(Dataset):
    """A custom dataset class that extends from `datasets.Dataset` and is used to generate
    an Argilla `FeedbackDataset` instance from the pre-defined configuration within the task
    provided to `Pipeline.generate`.
    """

    task: Union["Task", None] = None

    def to_argilla(
        self,
        dataset_columns: List[str] = None,
        vector_strategy: Union[bool, "SentenceTransformersExtractor"] = True,
    ) -> "FeedbackDataset":
        """Converts the dataset to an Argilla `FeedbackDataset` instance, based on the
        task defined in the dataset as part of `Pipeline.generate`.

        Args:
            fields (List[str]): the fields to be used for the Argilla `FeedbackDataset` instance.
                By default, the first 5 fields will be used.
            vector_strategy (Union[bool, SentenceTransformersExtractor]): the strategy to be used for
                adding vectors to the dataset. If `True`, the default `SentenceTransformersExtractor`
                will be used with the `TaylorAI/bge-micro-2` model. If `False`, no vectors will be added to the dataset.

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

        # set columns to all input and output columns for the task
        if dataset_columns is None:
            dataset_columns = getattr(self.task, "input_args_names", []) + getattr(
                self.task, "output_args_names", []
            )
        # get the first 5 that align with column selection + f"{column_name}_idx"
        selected_fields = []
        optional_fields = [field.name for field in rg_dataset.fields]
        selected_fields = [
            column
            for column in dataset_columns
            if any(column in optional_field for optional_field in optional_fields)
        ]
        selected_fields = list(dict.fromkeys(selected_fields))
        if len(selected_fields) > 5:
            selected_fields = selected_fields[:5]
            warnings.warn(
                f"More than 5 fields found from {optional_fields}, only the first 5 will be used: {selected_fields} for vectors.",
                stacklevel=2,
            )

        rg_dataset = self.add_vectors_to_argilla_dataset(
            dataset=rg_dataset, vector_strategy=vector_strategy, fields=selected_fields
        )

        return rg_dataset

    def add_vectors_to_argilla_dataset(
        self,
        dataset: Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"],
        vector_strategy: Union[bool, "SentenceTransformersExtractor"],
        fields: List[str] = None,
    ) -> Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"]:
        if _ARGILLA_AVAILABLE and vector_strategy:
            try:
                if isinstance(vector_strategy, SentenceTransformersExtractor):
                    ste: SentenceTransformersExtractor = vector_strategy
                elif vector_strategy:
                    ste = SentenceTransformersExtractor()
                dataset = ste.update_dataset(
                    dataset=dataset, fields=[field.name for field in dataset.fields][:5]
                )
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
        return dataset

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

    Examples:
        >>> from distilabel.dataset import DatasetCheckpoint
        >>> # Save the dataset every 10% of the records generated.
        >>> checkpoint = DatasetCheckpoint(save_frequency=len(dataset) // 10)
        >>> # Afterwards, we can access the checkpoint's checkpoint.path.
    """

    path: Path = Path.cwd() / "ckpt"
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
