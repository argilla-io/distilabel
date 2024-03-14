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

from typing_extensions import override

try:
    import argilla as rg
except ImportError as ie:
    raise ImportError(
        "Argilla is not installed. Please install it using `pip install argilla`."
    ) from ie

from distilabel.integrations.argilla.base import _Argilla

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class PromptCompletionToArgilla(_Argilla):
    """Step that creates a dataset in Argilla during the load phase, and then pushses the input
    batches into it as records. This dataset is a prompt-completion dataset, where there's one field
    per each input, and then a label question to rate the quality of the completion in either bad,
    good or excellent.

    Args:
        api_url: The URL of the Argilla API. Defaults to `None`, which means it will be read from
            the `ARGILLA_API_URL` environment variable.
        api_key: The API key to authenticate with Argilla. Defaults to `None`, which means it will
            be read from the `ARGILLA_API_KEY` environment variable.
        dataset_name: The name of the dataset in Argilla.
        dataset_workspace: The workspace where the dataset will be created in Argilla. Defaults to
            None, which means it will be created in the default workspace.

    Columns:

    - `prompt`
    - `completion`
    """

    def load(self) -> None:
        """Sets the `_prompt` and `_completion` attributes based on the `inputs_mapping`, otherwise
        uses the default values; and then uses those values to createa a `FeedbackDataset` suited for
        the prompt-completion scenario. And then it pushes it to Argilla.
        """
        self._rg_init()

        self._prompt = (
            self.input_mappings["prompt"] if self.input_mappings else "prompt"
        )
        self._completion = (
            self.input_mappings["completion"] if self.input_mappings else "completion"
        )

        _rg_dataset = rg.FeedbackDataset(
            fields=[
                rg.TextField(name=self._prompt),  # type: ignore
                rg.TextField(name=self._completion),  # type: ignore
            ],
            questions=[
                rg.LabelQuestion(  # type: ignore
                    name="quality",
                    title=f"What's the quality of the {self._completion} for the given {self._prompt}?",
                    labels=["bad", "good", "excellent"],
                )
            ],
        )
        self._rg_dataset = _rg_dataset.push_to_argilla(
            name=self.dataset_name, workspace=self.dataset_workspace
        )

    @property
    def inputs(self) -> List[str]:
        return ["prompt", "completion"]

    @override
    def process(self, inputs: "StepInput") -> "StepOutput":
        """Creates and pushes the records as FeedbackRecords to the Argilla dataset."""
        self._rg_dataset.add_records(  # type: ignore
            [
                rg.FeedbackRecord(
                    fields={
                        self._prompt: input["prompt"],
                        self._completion: input["completion"],
                    },
                )
                for input in inputs
            ]
        )
        # Empty yield as it's intended to be a leaf step
        yield [{}]
