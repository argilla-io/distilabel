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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from distilabel.steps.base import Step, StepInput
from distilabel.utils.argilla import ArgillaBase

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class ArgillaStepBase(Step, ArgillaBase, ABC):
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

    @property
    def outputs(self) -> "StepColumns":
        """The outputs of the step is an empty list, since the steps subclassing from this one, will
        always be leaf nodes and won't propagate the inputs neither generate any outputs.
        """
        return []

    @property
    @abstractmethod
    def inputs(self) -> "StepColumns": ...

    @abstractmethod
    def process(self, *inputs: StepInput) -> "StepOutput": ...
