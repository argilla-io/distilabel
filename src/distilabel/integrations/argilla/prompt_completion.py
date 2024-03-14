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

from distilabel.integrations.argilla.base import Argilla

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class PromptCompletionToArgilla(Argilla):
    def load(self) -> None:
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
        yield [{}]
