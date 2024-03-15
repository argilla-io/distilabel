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
from typing import TYPE_CHECKING, Any, List, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import Annotated

try:
    import argilla as rg
except ImportError as ie:
    raise ImportError(
        "Argilla is not installed. Please install it using `pip install argilla`."
    ) from ie

from distilabel.steps.base import Step

if TYPE_CHECKING:
    from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset

    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class Argilla(Step, ABC):
    """Abstract step that provides a class to subclass from, that contains the boilerplate code
    required to interact with Argilla, as well as some extra validations on top of it. It also defines
    the abstract methods that need to be implemented in order to add a new dataset type as a step.

    Note:
        This class is not intended to be instanced directly, but via subclass.

    Args:
        api_url: The URL of the Argilla API. Defaults to `None`, which means it will be read from
            the `ARGILLA_API_URL` environment variable.
        api_key: The API key to authenticate with Argilla. Defaults to `None`, which means it will
            be read from the `ARGILLA_API_KEY` environment variable.
        dataset_name: The name of the dataset in Argilla.
        dataset_workspace: The workspace where the dataset will be created in Argilla. Defaults to
            None, which means it will be created in the default workspace.
    """

    api_url: Annotated[Optional[str], Field(validate_default=True)] = None
    api_key: Annotated[Optional[SecretStr], Field(validate_default=True)] = None

    dataset_name: str
    dataset_workspace: Optional[str] = None

    _rg_dataset: Optional["RemoteFeedbackDataset"] = PrivateAttr(...)

    @field_validator("api_url")
    @classmethod
    def api_url_must_not_be_none(cls, v: Optional[str]) -> str:
        """Ensures that either the `api_url` or the environment variable `ARGILLA_API_URL` are set."""
        v = v or os.getenv("ARGILLA_API_URL", None)  # type: ignore
        if v is None:
            raise ValueError(
                "You must provide an API URL either via `api_url` arg or setting `ARGILLA_API_URL` environment variable to use Argilla."
            )
        return v

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_none(cls, v: Union[str, SecretStr, None]) -> SecretStr:
        """Ensures that either the `api_key` or the environment variable `ARGILLA_API_KEY` are set.

        Additionally, the `api_key` when provided is casted to `pydantic.SecretStr` to prevent it
        from being leaked and/or included within the logs or the serialization of the object.
        """
        v = v or os.getenv("ARGILLA_API_KEY", None)  # type: ignore
        if v is None:
            raise ValueError(
                "You must provide an API key either via `api_key` arg or setting `ARGILLA_API_URL` environment variable to use Argilla."
            )
        if not isinstance(v, SecretStr):
            v = SecretStr(v)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

    def _rg_init(self) -> None:
        """Initializes the Argilla API client with the provided `api_url` and `api_key`."""
        try:
            rg.init(api_url=self.api_url, api_key=self.api_key.get_secret_value())  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to initialize the Argilla API: {e}") from e

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
    def process(self, inputs: "StepInput") -> "StepOutput":
        ...
