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

from typing import TYPE_CHECKING, List

from pydantic import PrivateAttr
from typing_extensions import override

try:
    import argilla as rg
except ImportError as ie:
    raise ImportError(
        "Argilla is not installed. Please install it using `pip install argilla`."
    ) from ie

from distilabel.integrations.argilla.base import Argilla
from distilabel.steps.base import StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class TextGenerationToArgilla(Argilla):
    """Step that creates a dataset in Argilla during the load phase, and then pushses the input
    batches into it as records. This dataset is a text-generation dataset, where there's one field
    per each input, and then a label question to rate the quality of the completion in either bad
    (represented with 👎) or good (represented with 👍).

    Args:
        dataset_name: The name of the dataset in Argilla.
        dataset_workspace: The workspace where the dataset will be created in Argilla. Defaults to
            `None`, which means it will be created in the default workspace.
        api_url: The URL of the Argilla API. Defaults to `None`, which means it will be read from
            the `ARGILLA_API_URL` environment variable.
        api_key: The API key to authenticate with Argilla. Defaults to `None`, which means it will
            be read from the `ARGILLA_API_KEY` environment variable.

    Runtime parameters:
        - `api_url`: The base URL to use for the Argilla API requests.
        - `api_key`: The API key to authenticate the requests to the Argilla API.

    Input columns:
        - instruction (`str`): The instruction that was used to generate the completion.
        - generation (`str`): The completion that was generated based on the input instruction.
    """

    _instruction: str = PrivateAttr(...)
    _generation: str = PrivateAttr(...)

    def load(self) -> None:
        """Sets the `_instruction` and `_generation` attributes based on the `inputs_mapping`, otherwise
        uses the default values; and then uses those values to createa a `FeedbackDataset` suited for
        the prompt-completion scenario. And then it pushes it to Argilla.
        """
        self._rg_init()

        self._instruction = (
            self.input_mappings["instruction"] if self.input_mappings else "instruction"
        )
        self._generation = (
            self.input_mappings["generation"] if self.input_mappings else "generation"
        )

        if self._rg_dataset_exists():
            _rg_dataset = rg.FeedbackDataset.from_argilla(
                name=self.dataset_name,
                workspace=self.dataset_workspace,
            )

            for field in _rg_dataset.fields:
                if (
                    field.name not in [self._instruction, self._generation]
                    and field.required
                ):
                    raise ValueError(
                        f"The dataset {self.dataset_name} in the workspace {self.dataset_workspace} already exists,"
                        f" but contains at least a required field that is neither `{self._instruction}` nor `{self._generation}`."
                    )

            self._rg_dataset = _rg_dataset
        else:
            _rg_dataset = rg.FeedbackDataset(
                fields=[
                    rg.TextField(name=self._instruction),  # type: ignore
                    rg.TextField(name=self._generation),  # type: ignore
                ],
                questions=[
                    rg.LabelQuestion(  # type: ignore
                        name="quality",
                        title=f"What's the quality of the {self._generation} for the given {self._instruction}?",
                        labels=["👍", "👎"],
                    )
                ],
            )
            self._rg_dataset = _rg_dataset.push_to_argilla(
                name=self.dataset_name, workspace=self.dataset_workspace
            )

    @property
    def inputs(self) -> List[str]:
        """The inputs for the step are the `prompt` and the `completion`."""
        return ["instruction", "generation"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        """Creates and pushes the records as FeedbackRecords to the Argilla dataset.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        self._rg_dataset.add_records(  # type: ignore
            [
                rg.FeedbackRecord(
                    fields={
                        self._instruction: input["instruction"],
                        self._generation: input["generation"],
                    },
                )
                for input in inputs
            ]
        )
        # Empty yield as it's intended to be a leaf step
        yield [{}]
