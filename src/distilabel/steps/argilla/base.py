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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from pydantic import Field, PrivateAttr, SecretStr

try:
    import argilla as rg
except ImportError as ie:
    raise ImportError(
        "Argilla is not installed. Please install it using `pip install argilla`."
    ) from ie

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset

    from distilabel.steps.typing import StepOutput


_ARGILLA_API_KEY_ENV_VAR_NAME = "ARGILLA_API_KEY"


class Argilla(Step, ABC):
    """Abstract step that provides a class to subclass from, that contains the boilerplate code
    required to interact with Argilla, as well as some extra validations on top of it. It also defines
    the abstract methods that need to be implemented in order to add a new dataset type as a step.

    Note:
        This class is not intended to be instanced directly, but via subclass.

    Args:
        dataset_name: The name of the dataset in Argilla.
        dataset_workspace: The workspace where the dataset will be created in Argilla. Defaults to
            None, which means it will be created in the default workspace.
        api_url: The URL of the Argilla API. Defaults to `None`, which means it will be read from
            the `ARGILLA_API_URL` environment variable.
        api_key: The API key to authenticate with Argilla. Defaults to `None`, which means it will
            be read from the `ARGILLA_API_KEY` environment variable.

    Runtime parameters:
        - `api_url`: The base URL to use for the Argilla API requests.
        - `api_key`: The API key to authenticate the requests to the Argilla API.

    Input columns:
        - dynamic, based on the `inputs` value provided
    """

    dataset_name: str
    dataset_workspace: Optional[str] = None

    api_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv("ARGILLA_BASE_URL"),
        description="The base URL to use for the Argilla API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_ARGILLA_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Argilla API.",
    )

    _rg_dataset: Optional["RemoteFeedbackDataset"] = PrivateAttr(...)

    def _rg_init(self) -> None:
        """Initializes the Argilla API client with the provided `api_url` and `api_key`."""
        try:
            if "hf.space" in self.api_url and "HF_TOKEN" in os.environ:
                headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
            else:
                headers = None
            rg.init(
                api_url=self.api_url,
                api_key=self.api_key.get_secret_value(),
                extra_headers=headers,
            )  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to initialize the Argilla API: {e}") from e

    def _rg_dataset_exists(self) -> bool:
        """Checks if the dataset already exists in Argilla."""
        return self.dataset_name in [
            dataset.name
            for dataset in rg.FeedbackDataset.list(workspace=self.dataset_workspace)
        ]

    @property
    def outputs(self) -> List[str]:
        """The outputs of the step is an empty list, since the steps subclassing from this one, will
        always be leaf nodes and won't propagate the inputs neither generate any outputs.
        """
        return []

    @abstractmethod
    def load(self) -> None:
        ...

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        ...

    @abstractmethod
    def process(self, inputs: StepInput) -> "StepOutput":
        ...
