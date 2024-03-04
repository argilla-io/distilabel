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

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import requests
from datasets import load_dataset

from distilabel.pipeline.step.base import GeneratorStep, RuntimeParameter
from distilabel.pipeline.step.typing import GeneratorStepOutput


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

    response = requests.get(
        "https://datasets-server.huggingface.co/info", params=params
    )

    assert response.status_code == 200, f"Failed to get '{repo_id}' dataset info."

    return response.json()["dataset_info"]


class LoadHubDataset(GeneratorStep):
    """A generator step that loads a dataset from the Hugging Face Hub using the `datasets`
    library.

    This step will load the dataset in streaming mode, which means that it will not load the
    entire dataset into memory at once. Instead, it will load the dataset in chunks and yield
    the transformed data as it is loaded from the Hugging Face Hub.

    This step needs the following runtime parameters:

    - `repo_id`: The Hugging Face Hub repository ID of the dataset to load.
    - `split`: The split of the dataset to load.
    - `config`: The configuration of the dataset to load. This is optional and only needed if the
        dataset has multiple configurations.

    Columns:

    - `input`: None
    - `output`: dynamic, based on the dataset being loaded.
    """

    repo_id: RuntimeParameter[str] = None
    split: RuntimeParameter[str] = None
    config: Optional[RuntimeParameter[str]] = None

    def load(self) -> None:
        """Load the dataset from the Hugging Face Hub"""
        self._values["dataset"] = load_dataset(
            self.repo_id,
            self.config,
            split=self.split,
            streaming=True,
        )

    def process(self) -> GeneratorStepOutput:
        """Yields batches from the loaded dataset from the Hugging Face Hub.

        Yield:
            A tuple containing a batch of rows and a boolean indicating if the batch is
            the last one.
        """
        dataset = self._values["dataset"]
        num_examples = self._get_dataset_num_examples()
        num_returned_rows = 0
        for batch in dataset.iter(batch_size=self.batch_size):
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
