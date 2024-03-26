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

import hashlib
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


class PreferenceRatingToArgilla(Argilla):
    """Step that creates a dataset in Argilla during the load phase, and then pushses the input
    batches into it as records. This dataset is a preference dataset, where there's one field
    for the instruction and one extra field per each generation within the same record, and then
    a rating question per each of the generation fields. The rating question asks the annotator to
    set a rating from 1 to 5 for each of the provided generations.

    Args:
        num_generations: The number of generations to include in the dataset.
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
        - generations (`List[str]`): The completion that was generated based on the input instruction.
    """

    num_generations: int

    _id: str = PrivateAttr(default="id")
    _instruction: str = PrivateAttr(...)
    _generations: str = PrivateAttr(...)

    def load(self) -> None:
        """Sets the `_instruction` and `_generations` attributes based on the `inputs_mapping`, otherwise
        uses the default values; and then uses those values to createa a `FeedbackDataset` suited for
        the text-generation scenario. And then it pushes it to Argilla.
        """
        self._rg_init()

        self._instruction = self.input_mappings.get("instruction", "instruction")
        self._generations = self.input_mappings.get("generations", "generations")

        if self._rg_dataset_exists():
            _rg_dataset = rg.FeedbackDataset.from_argilla(
                name=self.dataset_name,
                workspace=self.dataset_workspace,
            )

            for field in _rg_dataset.fields:
                if (
                    field.name
                    not in [self._id, self._instruction]
                    + [
                        f"{self._generations}-{idx}"
                        for idx in range(self.num_generations)
                    ]
                    and field.required
                ):
                    raise ValueError(
                        f"The dataset {self.dataset_name} in the workspace {self.dataset_workspace} already exists,"
                        f" but contains at least a required field that is neither `{self._id}`, `{self._instruction}`,"
                        f" nor `{self._generations}`."
                    )

            self._rg_dataset = _rg_dataset
        else:
            _rg_dataset = rg.FeedbackDataset(
                fields=[
                    rg.TextField(name=self._id, title=self._id),  # type: ignore
                    rg.TextField(name=self._instruction, title=self._instruction),  # type: ignore
                ]
                + [
                    rg.TextField(  # type: ignore
                        name=f"{self._generations}-{idx}",
                        title=f"{self._generations}-{idx}",
                        required=True if idx == 0 else False,
                    )
                    for idx in range(self.num_generations)
                ],
                questions=[
                    rg.RatingQuestion(  # type: ignore
                        name=f"{self._generations}-{idx}",
                        title=f"Rate the quality of the {self._generations}-{idx}",
                        description=f"Ignore this question if the corresponding `{self._generations}-{idx}` field is not available."
                        if idx != 0
                        else None,
                        values=[1, 2, 3, 4, 5],
                        required=True if idx == 0 else False,
                    )
                    for idx in range(self.num_generations)
                ],
            )
            self._rg_dataset = _rg_dataset.push_to_argilla(
                name=self.dataset_name, workspace=self.dataset_workspace
            )

    @property
    def inputs(self) -> List[str]:
        """The inputs for the step are the `instruction` and the `generations`."""
        return ["instruction", "generations"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        """Creates and pushes the records as FeedbackRecords to the Argilla dataset.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        records = []
        for input in inputs:
            instruction_id = hashlib.sha256(
                input["instruction"].encode("utf-8")
            ).hexdigest()
            generations = {
                f"{self._generations}-{idx}": generation
                for idx, generation in enumerate(input["generations"])
            }
            records.append(
                rg.FeedbackRecord(  # type: ignore
                    fields={
                        "id": instruction_id,
                        "instruction": input["instruction"],
                        **generations,
                    },
                )
            )
        self._rg_dataset.add_records(records)  # type: ignore
        yield inputs
