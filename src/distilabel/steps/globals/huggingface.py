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
from typing import TYPE_CHECKING, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.exceptions import DatasetNotFoundError
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class PushToHub(GlobalStep):
    """Push data to a Hugging Face Hub dataset or merge with existing data.

    A `GlobalStep` which creates a `datasets.Dataset` with the input data and either
    pushes it as a new dataset or merges it with an existing dataset on the Hugging Face Hub.

    Attributes:
        repo_id: The Hugging Face Hub repository ID where the dataset will be uploaded.
        split: The split of the dataset that will be pushed or merged. Defaults to `"train"`.
        private: Whether the dataset should be private or not. Defaults to `False`.
        token: The token used to authenticate with the Hub. If not provided, it will be
            obtained from the environment variable `HF_TOKEN` or the local Hugging Face CLI
            configuration. Defaults to `None`.
        merge_with_existing: Whether to merge the new data with an existing dataset. Defaults to `False`.

    Runtime parameters:
        - `repo_id`: The Hugging Face Hub repository ID for the dataset.
        - `split`: The split of the dataset to push or merge.
        - `private`: Whether the dataset should be private.
        - `token`: The authentication token for the Hub.
        - `merge_with_existing`: Whether to merge with an existing dataset.

    Input columns:
        - dynamic (`all`): all columns from the input will be used to create or update the dataset.

    Categories:
        - save
        - dataset
        - huggingface

    Examples:

        Push or merge batches of your dataset to a Hugging Face Hub repository:

        ```python
        from distilabel.steps import PushToHub

        push = PushToHub(repo_id="path_to/repo")
        push.load()

        result = next(
            push.process(
                [
                    {
                        "instruction": "instruction ",
                        "generation": "generation"
                    }
                ],
            )
        )
        # >>> result
        # [{'instruction': 'instruction ', 'generation': 'generation'}]
        ```
    """

    repo_id: RuntimeParameter[str] = Field(
        default=None,
        description="The Hugging Face Hub repository ID where the dataset will be uploaded or merged.",
    )
    split: RuntimeParameter[str] = Field(
        default="train",
        description="The split of the dataset that will be pushed or merged. Defaults to 'train'.",
    )
    private: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether the dataset should be private or not. Defaults to `False`.",
    )
    token: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The token used to authenticate with the Hub. If not provided,"
        " it will be obtained from the environment variable `HF_TOKEN`"
        " or the local Hugging Face CLI configuration. Defaults to `None`.",
    )
    merge_with_existing: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to merge the new data with an existing dataset. Defaults to `False`.",
    )

    def process(self, inputs: StepInput) -> "StepOutput":
        """Process the input data and either push it as a new dataset or merge it with
        an existing dataset on the Hugging Face Hub.

        Args:
            inputs: The input data to be transformed into a `datasets.Dataset`.

        Yields:
            Propagates the received inputs for potential further processing in the pipeline.
        """
        dataset_dict = defaultdict(list)
        for input in inputs:
            for key, value in input.items():
                dataset_dict[key].append(value)
        dataset_dict = dict(dataset_dict)
        new_dataset = Dataset.from_dict(dataset_dict)

        if self.merge_with_existing:
            try:
                existing_dataset = load_dataset(self.repo_id, split=self.split)
                dataset = concatenate_datasets([existing_dataset, new_dataset])
                self._logger.info(
                    f"Successfully merged new data with existing dataset in {self.repo_id}"
                )
            except DatasetNotFoundError:
                self._logger.info(
                    f"Could not find existing dataset at: {self.repo_id}. Creating a new one."
                )
                dataset = new_dataset
            except Exception as e:
                self._logger.error(f"Error during merging process: {e}")
                raise
        else:
            dataset = new_dataset

        dataset.push_to_hub(
            self.repo_id,  # type: ignore
            split=self.split,
            private=self.private,
            token=self.token or os.getenv("HF_TOKEN"),
        )
        self._logger.info(
            f"{'Updated' if self.merge_with_existing else 'Created'} dataset in {self.repo_id}"
        )
        yield inputs
