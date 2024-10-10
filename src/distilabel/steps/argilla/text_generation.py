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
from typing import TYPE_CHECKING

from pydantic import PrivateAttr
from typing_extensions import override

try:
    import argilla as rg
except ImportError:
    pass

from distilabel.errors import DistilabelUserError
from distilabel.steps.argilla.base import ArgillaBase
from distilabel.steps.base import StepInput
from distilabel.steps.typing import StepColumns

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class TextGenerationToArgilla(ArgillaBase):
    """Creates a text generation dataset in Argilla.

    `Step` that creates a dataset in Argilla during the load phase, and then pushes the input
    batches into it as records. This dataset is a text-generation dataset, where there's one field
    per each input, and then a label question to rate the quality of the completion in either bad
    (represented with 👎) or good (represented with 👍).

    Note:
        This step is meant to be used in conjunction with a `TextGeneration` step and no column mapping
        is needed, as it will use the default values for the `instruction` and `generation` columns.

    Attributes:
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
        - generation (`str` or `List[str]`): The completions that were generated based on the input instruction.

    Examples:
        Push a text generation dataset to an Argilla instance:

        ```python
        from distilabel.steps import PreferenceToArgilla

        to_argilla = TextGenerationToArgilla(
            num_generations=2,
            api_url="https://dibt-demo-argilla-space.hf.space/",
            api_key="api.key",
            dataset_name="argilla_dataset",
            dataset_workspace="my_workspace",
        )
        to_argilla.load()

        result = next(
            to_argilla.process(
                [
                    {
                        "instruction": "instruction",
                        "generation": "generation",
                    }
                ],
            )
        )
        # >>> result
        # [{'instruction': 'instruction', 'generation': 'generation'}]
        ```
    """

    inputs: StepColumns = ["instruction", "generation"]
    _id: str = PrivateAttr(default="id")
    _instruction: str = PrivateAttr(...)
    _generation: str = PrivateAttr(...)

    def load(self) -> None:
        """Sets the `_instruction` and `_generation` attributes based on the `inputs_mapping`, otherwise
        uses the default values; and then uses those values to create a `FeedbackDataset` suited for
        the text-generation scenario. And then it pushes it to Argilla.
        """
        super().load()

        self._instruction = self.input_mappings.get("instruction", "instruction")
        self._generation = self.input_mappings.get("generation", "generation")

        if self._dataset_exists_in_workspace:
            _dataset = self._client.datasets(  # type: ignore
                name=self.dataset_name,  # type: ignore
                workspace=self.dataset_workspace,  # type: ignore
            )

            for field in _dataset.fields:
                if not isinstance(field, rg.TextField):  # type: ignore
                    continue
                if (
                    field.name not in [self._id, self._instruction, self._generation]
                    and field.required
                ):
                    raise DistilabelUserError(
                        f"The dataset '{self.dataset_name}' in the workspace '{self.dataset_workspace}'"
                        f" already exists, but contains at least a required field that is"
                        f" neither `{self._id}`, `{self._instruction}`, nor `{self._generation}`,"
                        " so it cannot be reused for this dataset.",
                        page="components-gallery/steps/textgenerationtoargilla/",
                    )

            self._dataset = _dataset
        else:
            _settings = rg.Settings(  # type: ignore
                fields=[
                    rg.TextField(name=self._id, title=self._id),  # type: ignore
                    rg.TextField(name=self._instruction, title=self._instruction),  # type: ignore
                    rg.TextField(name=self._generation, title=self._generation),  # type: ignore
                ],
                questions=[
                    rg.LabelQuestion(  # type: ignore
                        name="quality",
                        title=f"What's the quality of the {self._generation} for the given {self._instruction}?",
                        labels={"bad": "👎", "good": "👍"},  # type: ignore
                    )
                ],
            )
            _dataset = rg.Dataset(  # type: ignore
                name=self.dataset_name,
                workspace=self.dataset_workspace,
                settings=_settings,
                client=self._client,
            )
            self._dataset = _dataset.create()

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
                input["instruction"].encode("utf-8")
            ).hexdigest()

            generations = input["generation"]

            # If the `generation` is not a list, then convert it into a list
            if not isinstance(generations, list):
                generations = [generations]

            # Create a `generations_set` to avoid adding duplicates
            generations_set = set()

            for generation in generations:
                # If the generation is already in the set, then skip it
                if generation in generations_set:
                    continue
                # Otherwise, add it to the set
                generations_set.add(generation)

                records.append(
                    rg.Record(  # type: ignore
                        fields={
                            self._id: instruction_id,
                            self._instruction: input["instruction"],
                            self._generation: generation,
                        },
                    ),
                )
        self._dataset.records.log(records)  # type: ignore
        yield inputs
