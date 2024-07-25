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

from distilabel.embeddings.base import Embeddings
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class EmbeddingGeneration(Step):
    embeddings: Embeddings

    @property
    def inputs(self) -> List[str]:
        return ["text"]

    @property
    def outputs(self) -> List[str]:
        return ["embedding"]

    def load(self) -> None:
        """Loads the `Embeddings` model."""
        super().load()

        self.embeddings.load()

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        embeddings = self.embeddings.encode(inputs=[input["text"] for input in inputs])
        for input, embedding in zip(inputs, embeddings):
            input["embedding"] = embedding
        yield inputs
