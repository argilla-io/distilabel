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

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import requests
from datasets import Dataset, DatasetInfo, IterableDataset, load_dataset
from pydantic import Field, PrivateAttr
from requests.exceptions import ConnectionError
from upath import UPath

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GeneratorStep

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class LoadFromHub(GeneratorStep):
    """Loads a dataset from the Hugging Face Hub.

    `GeneratorStep` that loads a dataset from the Hugging Face Hub using the `datasets`
    library.

    Attributes:
        repo_id: The Hugging Face Hub repository ID of the dataset to load.
        split: The split of the dataset to load.
        config: The configuration of the dataset to load. This is optional and only needed
            if the dataset has multiple configurations.

    Runtime parameters:
        - `batch_size`: The batch size to use when processing the data.
        - `repo_id`: The Hugging Face Hub repository ID of the dataset to load.
        - `split`: The split of the dataset to load. Defaults to 'train'.
        - `config`: The configuration of the dataset to load. This is optional and only
            needed if the dataset has multiple configurations.
        - `streaming`: Whether to load the dataset in streaming mode or not. Defaults to
            `False`.
        - `num_examples`: The number of examples to load from the dataset.
            By default will load all examples.

    Output columns:
        - dynamic (`all`): The columns that will be generated by this step, based on the
            datasets loaded from the Hugging Face Hub.

    Categories:
        - load
    """

    repo_id: RuntimeParameter[str] = Field(
        default=None,
        description="The Hugging Face Hub repository ID of the dataset to load.",
    )
    split: RuntimeParameter[str] = Field(
        default="train",
        description="The split of the dataset to load. Defaults to 'train'.",
    )
    config: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The configuration of the dataset to load. This is optional and only"
        " needed if the dataset has multiple configurations.",
    )
    streaming: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to load the dataset in streaming mode or not. Defaults to False.",
    )
    num_examples: Optional[RuntimeParameter[int]] = Field(
        default=None,
        description="The number of examples to load from the dataset. By default will load all examples.",
    )
    storage_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The storage options to use when loading the dataset.",
    )

    _dataset: Union[IterableDataset, Dataset, None] = PrivateAttr(...)

    def load(self) -> None:
        """Load the dataset from the Hugging Face Hub"""
        super().load()

        self._dataset = load_dataset(
            self.repo_id,  # type: ignore
            self.config,
            split=self.split,
            streaming=self.streaming,
        )
        num_examples = self._get_dataset_num_examples()
        self.num_examples = (
            min(self.num_examples, num_examples) if self.num_examples else num_examples
        )

        if not self.streaming:
            self._dataset = self._dataset.select(range(self.num_examples))

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Yields batches from the loaded dataset from the Hugging Face Hub.

        Args:
            offset: The offset to start yielding the data from. Will be used during the caching
                process to help skipping already processed data.

        Yields:
            A tuple containing a batch of rows and a boolean indicating if the batch is
            the last one.
        """
        num_returned_rows = 0
        for batch_num, batch in enumerate(
            self._dataset.iter(batch_size=self.batch_size)  # type: ignore
        ):
            if batch_num * self.batch_size < offset:
                continue
            transformed_batch = self._transform_batch(batch)
            batch_size = len(transformed_batch)
            num_returned_rows += batch_size
            yield transformed_batch, num_returned_rows >= self.num_examples

    @property
    def outputs(self) -> List[str]:
        """The columns that will be generated by this step, based on the datasets loaded
        from the Hugging Face Hub.

        Returns:
            The columns that will be generated by this step.
        """
        return self._get_dataset_columns()

    def _transform_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform a batch of data from the Hugging Face Hub into a list of rows.

        Args:
            batch: The batch of data from the Hugging Face Hub.

        Returns:
            A list of rows, where each row is a dictionary of column names and values.
        """
        length = len(next(iter(batch.values())))
        rows = []
        for i in range(length):
            rows.append({col: values[i] for col, values in batch.items()})
        return rows

    def _get_dataset_num_examples(self) -> int:
        """Get the number of examples in the dataset, based on the `split` and `config`
        runtime parameters provided.

        Returns:
            The number of examples in the dataset.
        """
        dataset_info = self._get_dataset_info()
        split = self.split
        if self.config:
            return dataset_info["splits"][split]["num_examples"]
        return dataset_info["default"]["splits"][split]["num_examples"]

    def _get_dataset_columns(self) -> List[str]:
        """Get the columns of the dataset, based on the `config` runtime parameter provided.

        Returns:
            The columns of the dataset.
        """
        dataset_info = self._get_dataset_info()

        if isinstance(dataset_info, DatasetInfo):
            if self.config:
                return list(self._dataset[self.config].info.features.keys())
            return list(self._dataset.info.features.keys())

        if self.config:
            return list(dataset_info["features"].keys())
        return list(dataset_info["default"]["features"].keys())

    def _get_dataset_info(self) -> Dict[str, Any]:
        """Calls the Datasets Server API from Hugging Face to obtain the dataset information.

        Returns:
            The dataset information.
        """
        repo_id = self.repo_id
        config = self.config

        try:
            return _get_hf_dataset_info(repo_id, config)
        except ConnectionError:
            # The previous could fail in case of a internet connection issues.
            # Assuming the dataset is already loaded and we can get the info from the loaded dataset, otherwise it will fail anyway.
            self.load()
            if config:
                return self._dataset[config].info
            return self._dataset.info


@lru_cache
def _get_hf_dataset_info(
    repo_id: str, config: Union[str, None] = None
) -> Dict[str, Any]:
    """Calls the Datasets Server API from Hugging Face to obtain the dataset information.
    The results are cached to avoid making multiple requests to the server.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        config: The configuration of the dataset. This is optional and only needed if the
            dataset has multiple configurations.

    Returns:
        The dataset information.
    """

    params = {"dataset": repo_id}
    if config is not None:
        params["config"] = config

    if "HF_TOKEN" in os.environ:
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    else:
        headers = None

    response = requests.get(
        "https://datasets-server.huggingface.co/info", params=params, headers=headers
    )

    assert (
        response.status_code == 200
    ), f"Failed to get '{repo_id}' dataset info. Make sure you have set the HF_TOKEN environment variable if it is a private dataset."

    return response.json()["dataset_info"]


class LoadFromDisk(LoadFromHub):
    """Loads a dataset from a file in disk.

    Take a look at [Hugging Face Datasets](https://huggingface.co/docs/datasets/loading)
    for more information of the supported file types.
    """

    data_files: Optional[Union[str, Path]] = Field(
        default=None,
        description="The data files, or directory containing the data files, to generate the dataset from.",
    )
    filetype: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The expected filetype. If not provided, it will be inferred from the file extension.",
    )

    def load(self) -> None:
        """Load the dataset from the file/s in disk."""
        super(GeneratorStep, self).load()

        data_path = UPath(self.data_files, storage_options=self.storage_options)

        # def get_filetype(data_path: UPath) -> str:
        #     filetype = data_path.suffix.lstrip(".")
        #     if filetype == "jsonl":
        #         filetype = "json"
        #     return filetype

        # if data_path.is_file():
        #     self.filetype = get_filetype(data_path)
        #     data_files = str(data_path)
        # elif data_path.is_dir():
        #     file_sequence = []
        #     file_map = defaultdict(list)
        #     for file_or_folder in data_path.iterdir():
        #         if file_or_folder.is_file():
        #             file_sequence.append(str(file_or_folder))
        #         elif file_or_folder.is_dir():
        #             for file in file_or_folder.iterdir():
        #                 file_sequence.append(str(file))
        #                 file_map[str(file_or_folder)].append(str(file))

        #     data_files = file_sequence or file_map
        #     # Try to obtain the filetype from any of the files, assuming all files have the same type.
        #     if file_sequence:
        #         self.filetype = get_filetype(UPath(file_sequence[0]))
        #     else:
        #         self.filetype = get_filetype(
        #             UPath(file_map[list(file_map.keys())[0]][0])
        #         )
        (data_files, self.filetype) = self._prepare_data_files(data_path)

        self._dataset = load_dataset(
            self.filetype,
            data_files=data_files,
            split=self.split,
            streaming=self.streaming,
            storage_options=self.storage_options,
        )

        if not self.streaming and self.num_examples:
            self._dataset = self._dataset.select(range(self.num_examples))
        if not self.num_examples:
            if self.streaming:
                # There's no better way to get the number of examples in a streaming dataset,
                # load it again for the moment.
                self.num_examples = len(
                    load_dataset(
                        self.filetype, data_files=self.data_files, split=self.split
                    )
                )
            else:
                self.num_examples = len(self._dataset)

    def _prepare_data_files(self, data_path: UPath) -> Tuple[str, str]:
        """Prepare the loading process by setting the `data_files` attribute."""

        def get_filetype(data_path: UPath) -> str:
            filetype = data_path.suffix.lstrip(".")
            if filetype == "jsonl":
                filetype = "json"
            return filetype

        if data_path.is_file():
            filetype = get_filetype(data_path)
            data_files = str(data_path)
        elif data_path.is_dir():
            file_sequence = []
            file_map = defaultdict(list)
            for file_or_folder in data_path.iterdir():
                if file_or_folder.is_file():
                    file_sequence.append(str(file_or_folder))
                elif file_or_folder.is_dir():
                    for file in file_or_folder.iterdir():
                        file_sequence.append(str(file))
                        file_map[str(file_or_folder)].append(str(file))

            data_files = file_sequence or file_map
            # Try to obtain the filetype from any of the files, assuming all files have the same type.
            if file_sequence:
                filetype = get_filetype(UPath(file_sequence[0]))
            else:
                filetype = get_filetype(UPath(file_map[list(file_map.keys())[0]][0]))
        return data_files, filetype

    @property
    def outputs(self) -> List[str]:
        """
        The columns that will be generated by this step, based on the datasets from a file
        in disk.

        Returns:
            The columns that will be generated by this step.
        """
        # We assume there are Dataset/IterableDataset, not it's ...Dict counterparts
        if self._dataset is Ellipsis:
            raise ValueError(
                "Dataset not loaded yet, you must call `load` method first."
            )

        return self._dataset.column_names
