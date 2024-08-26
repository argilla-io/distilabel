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

from datasets import Dataset
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class PushToHub(GlobalStep):
    """Push data to a Hugging Face Hub dataset.

    A `GlobalStep` which creates a `datasets.Dataset` with the input data and pushes
    it to the Hugging Face Hub.

    Attributes:
        repo_id: The Hugging Face Hub repository ID where the dataset will be uploaded.
        split: The split of the dataset that will be pushed. Defaults to `"train"`.
        private: Whether the dataset to be pushed should be private or not. Defaults to
            `False`.
        token: The token that will be used to authenticate in the Hub. If not provided, the
            token will be tried to be obtained from the environment variable `HF_TOKEN`.
            If not provided using one of the previous methods, then `huggingface_hub` library
            will try to use the token from the local Hugging Face CLI configuration. Defaults
            to `None`.

    Runtime parameters:
        - `repo_id`: The Hugging Face Hub repository ID where the dataset will be uploaded.
        - `split`: The split of the dataset that will be pushed.
        - `private`: Whether the dataset to be pushed should be private or not.
        - `token`: The token that will be used to authenticate in the Hub.

    Input columns:
        - dynamic (`all`): all columns from the input will be used to create the dataset.

    Categories:
        - save
        - dataset
        - huggingface

    Examples:
        Push batches of your dataset to the Hugging Face Hub repository:

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
        description="The Hugging Face Hub repository ID where the dataset will be uploaded.",
    )
    split: RuntimeParameter[str] = Field(
        default="train",
        description="The split of the dataset that will be pushed. Defaults to 'train'.",
    )
    private: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether the dataset to be pushed should be private or not. Defaults"
        " to `False`.",
    )
    token: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The token that will be used to authenticate in the Hub. If not provided,"
        " the token will be tried to be obtained from the environment variable `HF_TOKEN`."
        " If not provided using one of the previous methods, then `huggingface_hub` library"
        " will try to use the token from the local Hugging Face CLI configuration. Defaults"
        " to `None`",
    )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Method that processes the input data, respecting the `datasets.Dataset` formatting,
        and pushes it to the Hugging Face Hub based on the `RuntimeParameter`s attributes.

        Args:
            inputs: that input data within a single object (as it's a GlobalStep) that
                will be transformed into a `datasets.Dataset`.

        Yields:
            Propagates the received inputs so that the `Distiset` can be generated if this is
            the last step of the `Pipeline`, or if this is not a leaf step and has follow up
            steps.
        """
        dataset_dict = defaultdict(list)
        for input in inputs:
            for key, value in input.items():
                dataset_dict[key].append(value)
        dataset_dict = dict(dataset_dict)
        dataset = Dataset.from_dict(dataset_dict)
        dataset.push_to_hub(
            self.repo_id,  # type: ignore
            split=self.split,
            private=self.private,
            token=self.token or os.getenv("HF_TOKEN"),
        )
        yield inputs
