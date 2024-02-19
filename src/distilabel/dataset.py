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

import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, get_args

from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import login as hf_hub_login

from distilabel.logger import get_logger
from distilabel.utils.argilla import infer_field_from_dataset_columns
from distilabel.utils.imports import _ARGILLA_AVAILABLE
from distilabel.utils.serialization import (
    TASK_FILE_NAME,
    load_from_dict,
    load_task_from_disk,
    read_json,
)

if _ARGILLA_AVAILABLE:
    from argilla.client.feedback.integrations.sentencetransformers import (
        SentenceTransformersExtractor,
    )
    from argilla.client.feedback.integrations.textdescriptives import (
        TextDescriptivesExtractor,
    )


if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord
    from argilla.client.feedback.integrations.sentencetransformers import (
        SentenceTransformersExtractor,
    )
    from argilla.client.feedback.integrations.textdescriptives import (
        TextDescriptivesExtractor,
    )

    from distilabel.tasks.base import Task

logger = get_logger()


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
        metric_strategy: Union[bool, "TextDescriptivesExtractor"] = True,
    ) -> "FeedbackDataset":
        """Converts the dataset to an Argilla `FeedbackDataset` instance, based on the
        task defined in the dataset as part of `Pipeline.generate`.

        Args:
            dataset_columns (List[str]): the dataset columns or fields to be used for the Argilla `FeedbackDataset` instance.
                By default, the first 5 columns or fields will be used.
            vector_strategy (Union[bool, SentenceTransformersExtractor]): the strategy to be used for
                adding vectors to the dataset. If `True`, the default `SentenceTransformersExtractor`
                will be used with the `TaylorAI/bge-micro-2` model. If `False`, no vectors will be added to the dataset.
            metric_strategy (Union[bool, TextDescriptivesExtractor]): the strategy to be used for
                adding metrics to the dataset. If `True`, the default `TextDescriptivesExtractor`
                will be used. If `False`, no metrics will be added to the dataset.

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

        selected_fields = infer_field_from_dataset_columns(
            dataset_columns=dataset_columns, dataset=rg_dataset, task=self.task
        )
        rg_dataset = self.add_vectors_to_argilla_dataset(
            dataset=rg_dataset, vector_strategy=vector_strategy, fields=selected_fields
        )
        rg_dataset = self.add_metrics_to_argilla_dataset(
            dataset=rg_dataset, metric_strategy=metric_strategy, fields=selected_fields
        )
        return rg_dataset

    def add_vectors_to_argilla_dataset(
        self,
        dataset: Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"],
        vector_strategy: Union[bool, "SentenceTransformersExtractor"],
        fields: List[str] = None,
    ) -> Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"]:
        if _ARGILLA_AVAILABLE and vector_strategy:
            if isinstance(vector_strategy, SentenceTransformersExtractor):
                ste: SentenceTransformersExtractor = vector_strategy
            elif vector_strategy:
                ste = SentenceTransformersExtractor()
            dataset = ste.update_dataset(dataset=dataset, fields=fields)

        elif not _ARGILLA_AVAILABLE and vector_strategy:
            warnings.warn(
                "An error occurred while adding vectors to the dataset: "
                "The `argilla`/`sentence-transformers` packages are not installed or the installed version is not compatible with the"
                " required version. If you want to add vectors to your dataset, please run `pip install 'distilabel[argilla]'`.",
                stacklevel=2,
            )
        return dataset

    def add_metrics_to_argilla_dataset(
        self,
        dataset: Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"],
        metric_strategy: Union[bool, "TextDescriptivesExtractor"],
        fields: List[str] = None,
    ) -> Union["FeedbackRecord", List["FeedbackRecord"], "FeedbackDataset"]:
        if _ARGILLA_AVAILABLE and metric_strategy:
            if isinstance(metric_strategy, TextDescriptivesExtractor):
                tde: TextDescriptivesExtractor = metric_strategy
            elif metric_strategy:
                tde = TextDescriptivesExtractor()

            dataset = tde.update_dataset(dataset=dataset, fields=fields)
        elif not _ARGILLA_AVAILABLE and metric_strategy:
            warnings.warn(
                "An error occurred while adding metrics to the dataset: "
                "The `argilla`/`text-descriptives` packages are not installed or the installed version is not compatible with the"
                " required version. If you want to add metrics to your dataset, please run `pip install 'distilabel[argilla]'`.",
                stacklevel=2,
            )
        return dataset

    def save_to_disk(self, dataset_path: os.PathLike, **kwargs: Any) -> None:
        """Saves the datataset to disk, also saving the task.

        Args:
            dataset_path (os.PathLike): Path to the dataset.
            kwargs (Any): Additional arguments to be passed to `datasets.Dataset.save_to_disk`.

        Examples:
            >>> from distilabel.dataset import CustomDataset
            >>> dataset = CustomDataset(...)
            >>> dataset.save_to_disk("path/to/dataset")
        """
        super().save_to_disk(dataset_path, **kwargs)
        if self.task is not None:
            self.task.save(Path(dataset_path))

    @classmethod
    def load_from_disk(
        cls, dataset_path: os.PathLike, **kwargs: Any
    ) -> "CustomDataset":
        """Load a CustomDataset from disk, also reading the task.

        Args:
            dataset_path (os.PathLike): Path to the dataset.
            kwargs (Any): Keyword arguments passed to Dataset.load_from_disk.

        Returns:
            dataset: The loaded dataset.

        Examples:
            >>> from distilabel.dataset import CustomDataset
            >>> dataset: CustomDataset = CustomDataset.load_from_disk("path/to/dataset")
        """
        ds = super().load_from_disk(dataset_path, **kwargs)
        # Dynamically remaps the `datasets.Dataset` to be a `CustomDataset` instance
        ds.__class__ = cls
        task = load_task_from_disk(dataset_path)
        ds.task = task
        return ds

    def push_to_hub(
        self,
        repo_id: str,
        *args: Optional[Any],
        push_task: bool = True,
        **kwargs: Optional[Any],
    ) -> None:
        """Same method as `datasets.Dataset.push_to_hub`, but also pushes the task to simplify
        creating a CustomDataset from HuggingFace hub.

        Args:
            repo_id (str):
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.
            args (Any): Additional arguments to be passed to `datasets.Dataset.push_to_hub`.
            push_task (bool, optional):
                Whether to push the `Task` contained in the `CustomDataset`. Useful if you want to resuse
                the functionality of the `CustomDataset` out of the box. It will upload a json
                file (distilabel-task.json) containing the task to the hub.
                Defaults to True.
            kwargs (Any): Additional arguments to be passed to `datasets.Dataset.push_to_hub`.

        Examples:
            >>> from distilabel.dataset import CustomDataset
            >>> dataset = CustomDataset(...)
            >>> dataset.push_to_hub("org/dataset-name")
        """
        super().push_to_hub(repo_id, *args, **kwargs)
        if self.task is not None and push_task:
            try:
                logger.info("Pushing task to the hub...")
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                    f.write(json.dumps(self.task.dump(), indent=2))
                    f.flush()

                    HfApi().upload_file(
                        path_or_fileobj=f.name,
                        path_in_repo=TASK_FILE_NAME,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=kwargs.get("token"),
                    )
            except Exception as e:
                logger.info(f"Error while pushing the task to the hub: {e}.")


def load_dataset(path: str, *args: Any, **kwargs: Any) -> Union[Dataset, CustomDataset]:
    """Load a dataset from HuggingFace hub.

    Overloads the `datasets.load_dataset` method to return a `CustomDataset` instance,
    downloading the `Task` from the hub (if any).

    Args:
        path (str): Path to the dataset in the hub.
        args, kwargs: and any other arguments used by `datasets.load_dataset`

    Returns:
        dataset: CustomDataset instance, with the `Task` stored if available.

    Examples:
        >>> from distilabel.dataset import load_dataset
        >>> dataset: CustomDataset = load_dataset("argilla/distilabel-sample-evol-instruct", split="train")
    """
    from datasets import load_dataset as _load_dataset

    ds = _load_dataset(path, *args, **kwargs)
    cds = CustomDataset(ds.data.table)
    # download the task
    try:
        task_path = hf_hub_download(
            repo_id=path, filename=TASK_FILE_NAME, repo_type="dataset"
        )
        task = load_from_dict(read_json(task_path))
        cds.task = task
    except Exception as e:
        logger.info(f"Error while downloading the task from the hub: {e}.")
    return cds


CheckpointStrategies = Literal["disk", "hf-hub"]


@dataclass
class DatasetCheckpoint:
    """A checkpoint class that contains the information of a checkpoint.

    Args:
        path (Path): The path to the checkpoint.
        save_frequency (int): The frequency at which the checkpoint should be saved
            By default is set to -1 (no checkpoint is saved to disk, but the dataset
            is returned upon failure).
        strategy (CheckpointStrategies): The strategy to be used to save the checkpoint.
            Available options are: "disk" (to save the dataset to disk) and "hf-hub"
            (to push the dataset to the HuggingFace hub). By default is set to "disk".
        extra_kwargs (dict[str, Any]): Additional kwargs to be passed to the `save_to_disk` method of the Dataset.

    Examples:
        >>> from distilabel.dataset import DatasetCheckpoint
        >>> # Save the dataset every 10% of the records generated.
        >>> checkpoint = DatasetCheckpoint(save_frequency=len(dataset) // 10)
        >>> # Afterwards, we can access the checkpoint's checkpoint.path.
        >>> # You can also push the dataset to the hub by setting the strategy to `hf-hub`.
        >>> checkpoint = DatasetCheckpoint(
        ...     strategy="hf-hub",
        ...     extra_kwargs={"repo_id": "username/dataset-name"}
        ... )
        >>> # Keep in mind, if the environmnet variable "HF_API_TOKEN" is not set, you are required
        >>> # to pass the token as an argument to the `extra_kwargs`.
        >>> checkpoint = DatasetCheckpoint(
        ...     strategy="hf-hub",
        ...     extra_kwargs={"repo_id": "username/dataset-name", "token": "hf_..."}
        ... )
    """

    path: Path = Path.cwd() / "ckpt"
    save_frequency: int = -1
    strategy: CheckpointStrategies = "disk"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Internal fields to keep track of the number of records generated and when to check.
    _total_checks: int = field(repr=False, default=0)

    def __post_init__(self) -> None:
        """Checks validity of arguments.

        Raises:
            ValueError:
                - If the strategy is not valid.
                - If the `repo_id` is not passed when using the `hf-hub` strategy.
        """
        if self.strategy not in get_args(CheckpointStrategies):
            raise ValueError(
                f"Invalid strategy, valid ones are: {get_args(CheckpointStrategies)}"
            )

        if self.strategy == "hf-hub":
            if not self.extra_kwargs.get("repo_id"):
                raise ValueError(
                    "The `repo_id` is required when using the `hf-hub` strategy, please pass it as on 'extra_kwargs' argument."
                )
            if token := os.getenv("HF_API_TOKEN") or (
                token := self.extra_kwargs.get("token")
            ):
                hf_hub_login(token=token)
            else:
                raise ValueError(
                    "To use the `hf-hub` strategy, a token is required, you can either set it "
                    "as an environment variable `HF_API_TOKEN` or pass it as an argument `token` "
                    "via the `extra_kwargs`."
                )

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

    def save(self, dataset: CustomDataset) -> None:
        """Save the dataset given the strategy selected.

        Args:
            dataset (CustomDataset): Dataset to be saved.

        Raises:
            ValueError: If the strategy is not valid.
        """
        if self.strategy == "disk":
            self._save_to_disk(dataset, **self.extra_kwargs)

        elif self.strategy == "hf-hub":
            self._push_to_hub(dataset, **self.extra_kwargs)

        else:
            # We shouldn't reach this point after __post_init__, but just in case.
            raise ValueError(
                f"Invalid strategy, valid ones are: {get_args(CheckpointStrategies)}"
            )

    def _save_to_disk(
        self, dataset: Union[Dataset, CustomDataset], **kwargs: Any
    ) -> None:
        """Saves the dataset to disk.

        Args:
            dataset (Union[Dataset, CustomDataset]): The dataset to be saved.
        """
        dataset.save_to_disk(self.path, **kwargs)
        logger.info(f"Checkpoint saved to disk: {self.path}.")

    def _push_to_hub(self, dataset: CustomDataset, **kwargs: Any) -> None:
        """Pushes the dataset to the hub.

        Args:
            dataset (Union[Dataset, CustomDataset]): The dataset to be pushed.
        """
        repo_id = kwargs.pop("repo_id", None)
        if not repo_id:
            raise ValueError(
                "The `repo_id` is required when pushing to the hub, please pass it as on 'extra_kwargs' argument."
            )

        try:
            dataset.push_to_hub(repo_id, **kwargs)
            logger.info(f"Checkpoint saved to hub: {repo_id}.")
        except Exception as e:
            # TODO(plaguss): Check possible errors to provide a more informative message.
            # And maybe save to disk as a fallback.
            logger.info(f"Error while pushing the dataset to the hub: {e}.")
