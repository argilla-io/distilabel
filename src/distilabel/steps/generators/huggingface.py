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
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests
from datasets import IterableDataset, load_dataset
from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GeneratorStep

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


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


class LoadHubDataset(GeneratorStep):
    """A generator step that loads a dataset from the Hugging Face Hub using the `datasets`
    library.

    This step will load the dataset in streaming mode, which means that it will not load the
    entire dataset into memory at once. Instead, it will load the dataset in chunks and yield
    the transformed data as it is loaded from the Hugging Face Hub.

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

    Output columns
        - dynamic, based on the dataset being loaded
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

    _dataset: Union[IterableDataset, None] = PrivateAttr(...)

    def load(self) -> None:
        """Load the dataset from the Hugging Face Hub"""
        super().load()

        self._dataset = load_dataset(
            self.repo_id,  # type: ignore
            self.config,
            split=self.split,
            streaming=True,
        )

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Yields batches from the loaded dataset from the Hugging Face Hub.

        Args:
            offset: The offset to start yielding the data from. Will be used during the caching
                process to help skipping already processed data.

        Yields:
            A tuple containing a batch of rows and a boolean indicating if the batch is
            the last one.
        """
        num_examples = self._get_dataset_num_examples()
        num_returned_rows = 0
        for batch_num, batch in enumerate(
            self._dataset.iter(batch_size=self.batch_size)  # type: ignore
        ):
            if batch_num * self.batch_size < offset:
                continue
            transformed_batch = self._transform_batch(batch)
            batch_size = len(transformed_batch)
            num_returned_rows += batch_size
            yield transformed_batch, num_returned_rows == num_examples

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
        return _get_hf_dataset_info(repo_id, config)
