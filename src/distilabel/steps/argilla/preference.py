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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import PrivateAttr
from typing_extensions import override

try:
    import argilla as rg
except ImportError:
    pass

from distilabel.steps.argilla.base import Argilla
from distilabel.steps.base import StepInput

if TYPE_CHECKING:
    from argilla import (
        RatingQuestion,
        SuggestionSchema,
        TextField,
        TextQuestion,
    )

    from distilabel.steps.typing import StepOutput


class PreferenceToArgilla(Argilla):
    """Step that creates a dataset in Argilla during the load phase, and then pushes the input
    batches into it as records. This dataset is a preference dataset, where there's one field
    for the instruction and one extra field per each generation within the same record, and then
    a rating question per each of the generation fields. The rating question asks the annotator to
    set a rating from 1 to 5 for each of the provided generations.

    Note:
        This step is meant to be used in conjunction with the `UltraFeedback` step, or any other step
        generating both ratings and responses for a given set of instruction and generations for the
        given instruction. But alternatively, it can also be used with any other task or step generating
        only the `instruction` and `generations`, as the `ratings` and `rationales` are optional.

    Attributes:
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
        - ratings (`List[str]`, optional): The ratings for the generations. If not provided, the
            generated ratings won't be pushed to Argilla.
        - rationales (`List[str]`, optional): The rationales for the ratings. If not provided, the
            generated rationales won't be pushed to Argilla.
    """

    num_generations: int

    _id: str = PrivateAttr(default="id")
    _instruction: str = PrivateAttr(...)
    _generations: str = PrivateAttr(...)
    _ratings: str = PrivateAttr(...)
    _rationales: str = PrivateAttr(...)

    def load(self) -> None:
        """Sets the `_instruction` and `_generations` attributes based on the `inputs_mapping`, otherwise
        uses the default values; and then uses those values to create a `FeedbackDataset` suited for
        the text-generation scenario. And then it pushes it to Argilla.
        """
        super().load()

        # Both `instruction` and `generations` will be used as the fields of the dataset
        self._instruction = self.input_mappings.get("instruction", "instruction")
        self._generations = self.input_mappings.get("generations", "generations")
        # Both `ratings` and `rationales` will be used as suggestions to the default questions of the dataset
        self._ratings = self.input_mappings.get("ratings", "ratings")
        self._rationales = self.input_mappings.get("rationales", "rationales")

        if self._rg_dataset_exists():
            _rg_dataset = rg.FeedbackDataset.from_argilla(  # type: ignore
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
            _rg_dataset = rg.FeedbackDataset(  # type: ignore
                fields=[
                    rg.TextField(name=self._id, title=self._id),  # type: ignore
                    rg.TextField(name=self._instruction, title=self._instruction),  # type: ignore
                    *self._generation_fields(),  # type: ignore
                ],
                questions=self._rating_rationale_pairs(),  # type: ignore
            )
            self._rg_dataset = _rg_dataset.push_to_argilla(
                name=self.dataset_name, workspace=self.dataset_workspace
            )

    def _generation_fields(self) -> List["TextField"]:
        """Method to generate the fields for each of the generations."""
        return [
            rg.TextField(  # type: ignore
                name=f"{self._generations}-{idx}",
                title=f"{self._generations}-{idx}",
                required=True if idx == 0 else False,
            )
            for idx in range(self.num_generations)
        ]

    def _rating_rationale_pairs(
        self,
    ) -> List[Union["RatingQuestion", "TextQuestion"]]:
        """Method to generate the rating and rationale questions for each of the generations."""
        questions = []
        for idx in range(self.num_generations):
            questions.extend(
                [
                    rg.RatingQuestion(  # type: ignore
                        name=f"{self._generations}-{idx}-rating",
                        title=f"Rate {self._generations}-{idx} given {self._instruction} based on the annotation guidelines.",
                        description=f"Ignore this question if the corresponding `{self._generations}-{idx}` field is not available."
                        if idx != 0
                        else None,
                        values=[1, 2, 3, 4, 5],
                        required=True if idx == 0 else False,
                    ),
                    rg.TextQuestion(  # type: ignore
                        name=f"{self._generations}-{idx}-rationale",
                        title=f"Specify the rationale for {self._generations}-{idx}'s rating.",
                        description=f"Ignore this question if the corresponding `{self._generations}-{idx}` field is not available."
                        if idx != 0
                        else None,
                        required=False,
                    ),
                ]
            )
        return questions

    @property
    def inputs(self) -> List[str]:
        """The inputs for the step are the `instruction` and the `generations`. Optionally, one could also
        provide the `ratings` and the `rationales` for the generations."""
        return ["instruction", "generations"]

    def _add_suggestions_if_any(
        self, input: Dict[str, Any]
    ) -> List["SuggestionSchema"]:
        """Method to generate the suggestions for the `FeedbackRecord` based on the input."""
        # Since the `suggestions` i.e. answers to the `questions` are optional, will default to {}
        suggestions = []
        # If `ratings` is in `input`, then add those as suggestions
        if self._ratings in input:
            suggestions.extend(
                [
                    {
                        "question_name": f"{self._generations}-{idx}-rating",
                        "value": rating,
                    }
                    for idx, rating in enumerate(input[self._ratings])
                    if rating is not None
                    and isinstance(rating, int)
                    and rating in [1, 2, 3, 4, 5]
                ],
            )
        # If `rationales` is in `input`, then add those as suggestions
        if self._rationales in input:
            suggestions.extend(
                [
                    {
                        "question_name": f"{self._generations}-{idx}-rationale",
                        "value": rationale,
                    }
                    for idx, rationale in enumerate(input[self._rationales])
                    if rationale is not None and isinstance(rationale, str)
                ],
            )
        return suggestions

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Creates and pushes the records as FeedbackRecords to the Argilla dataset.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        records = []
        for input in inputs:
            # Generate the SHA-256 hash of the instruction to use it as the metadata
            instruction_id = hashlib.sha256(
                input["instruction"].encode("utf-8")  # type: ignore
            ).hexdigest()

            generations = {
                f"{self._generations}-{idx}": generation
                for idx, generation in enumerate(input["generations"])  # type: ignore
            }

            records.append(  # type: ignore
                rg.FeedbackRecord(  # type: ignore
                    fields={
                        "id": instruction_id,
                        "instruction": input["instruction"],  # type: ignore
                        **generations,
                    },
                    suggestions=self._add_suggestions_if_any(input),  # type: ignore
                )
            )
        self._rg_dataset.add_records(records)  # type: ignore
        yield inputs
