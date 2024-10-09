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

import importlib.util
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field, PrivateAttr, SecretStr

try:
    import argilla as rg
except ImportError:
    pass

from distilabel.errors import DistilabelUserError
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import Step, StepInput
from distilabel.steps.typing import StepColumns

if TYPE_CHECKING:
    from argilla import Argilla, Dataset

    from distilabel.steps.typing import StepOutput


_ARGILLA_API_URL_ENV_VAR_NAME = "ARGILLA_API_URL"
_ARGILLA_API_KEY_ENV_VAR_NAME = "ARGILLA_API_KEY"


class ArgillaBase(Step, ABC):
    """Abstract step that provides a class to subclass from, that contains the boilerplate code
    required to interact with Argilla, as well as some extra validations on top of it. It also defines
    the abstract methods that need to be implemented in order to add a new dataset type as a step.

    Note:
        This class is not intended to be instanced directly, but via subclass.

    Attributes:
        dataset_name: The name of the dataset in Argilla where the records will be added.
        dataset_workspace: The workspace where the dataset will be created in Argilla. Defaults to
            `None`, which means it will be created in the default workspace.
        api_url: The URL of the Argilla API. Defaults to `None`, which means it will be read from
            the `ARGILLA_API_URL` environment variable.
        api_key: The API key to authenticate with Argilla. Defaults to `None`, which means it will
            be read from the `ARGILLA_API_KEY` environment variable.

    Runtime parameters:
        - `dataset_name`: The name of the dataset in Argilla where the records will be
            added.
        - `dataset_workspace`: The workspace where the dataset will be created in Argilla.
            Defaults to `None`, which means it will be created in the default workspace.
        - `api_url`: The base URL to use for the Argilla API requests.
        - `api_key`: The API key to authenticate the requests to the Argilla API.

    Input columns:
        - dynamic, based on the `inputs` value provided
    """

    dataset_name: RuntimeParameter[str] = Field(
        default=None, description="The name of the dataset in Argilla."
    )
    dataset_workspace: Optional[RuntimeParameter[str]] = Field(
        default=None,
        description="The workspace where the dataset will be created in Argilla. Defaults "
        "to `None` which means it will be created in the default workspace.",
    )

    api_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(_ARGILLA_API_URL_ENV_VAR_NAME),
        description="The base URL to use for the Argilla API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_ARGILLA_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Argilla API.",
    )

    outputs: StepColumns = Field(
        default_factory=list,
        exclude=True,
        description=(
            "The outputs of the step is an empty list, since the steps subclassing from this one, will "
            "always be leaf nodes and won't propagate the inputs neither generate any outputs."
        ),
    )

    _client: Optional["Argilla"] = PrivateAttr(...)
    _dataset: Optional["Dataset"] = PrivateAttr(...)

    def model_post_init(self, __context: Any) -> None:
        """Checks that the Argilla Python SDK is installed, and then filters the Argilla warnings."""
        super().model_post_init(__context)

        if importlib.util.find_spec("argilla") is None:
            raise ImportError(
                "Argilla is not installed. Please install it using `pip install argilla"
                " --upgrade`."
            )

    def _client_init(self) -> None:
        """Initializes the Argilla API client with the provided `api_url` and `api_key`."""
        try:
            self._client = rg.Argilla(  # type: ignore
                api_url=self.api_url,
                api_key=self.api_key.get_secret_value(),  # type: ignore
                headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
                if isinstance(self.api_url, str)
                and "hf.space" in self.api_url
                and "HF_TOKEN" in os.environ
                else {},
            )
        except Exception as e:
            raise DistilabelUserError(
                f"Failed to initialize the Argilla API: {e}",
                page="sections/how_to_guides/advanced/argilla/",
            ) from e

    @property
    def _dataset_exists_in_workspace(self) -> bool:
        """Checks if the dataset already exists in Argilla in the provided workspace if any.

        Returns:
            `True` if the dataset exists, `False` otherwise.
        """
        return (
            self._client.datasets(  # type: ignore
                name=self.dataset_name,  # type: ignore
                workspace=self.dataset_workspace,
            )
            is not None
        )

    def load(self) -> None:
        """Method to perform any initialization logic before the `process` method is
        called. For example, to load an LLM, stablish a connection to a database, etc.
        """
        super().load()

        if self.api_url is None or self.api_key is None:
            raise DistilabelUserError(
                "`Argilla` step requires the `api_url` and `api_key` to be provided. Please,"
                " provide those at step instantiation, via environment variables `ARGILLA_API_URL`"
                " and `ARGILLA_API_KEY`, or as `Step` runtime parameters via `pipeline.run(parameters={...})`.",
                page="sections/how_to_guides/advanced/argilla/",
            )

        self._client_init()

    @abstractmethod
    def process(self, *inputs: StepInput) -> "StepOutput": ...
