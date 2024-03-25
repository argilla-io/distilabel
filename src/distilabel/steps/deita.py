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

from typing import TYPE_CHECKING, List, Type

from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from scikit_learn.neighbors import NearestNeighbors

    from distilabel.steps.typing import StepOutput


class DeitaFiltering(GlobalStep):
    data_budget: RuntimeParameter[int] = Field(
        default=None, description="The desired size of the dataset after filtering."
    )
    diversity_threshold: RuntimeParameter[float] = Field(
        default=0.9,
        description="If a row has a cosine distance with respect to it's nearest neighbor"
        " greater than this value, it will be included in the filtered dataset.",
    )

    _NearestNeighbors: Type["NearestNeighbors"] = PrivateAttr(...)

    def load(self) -> None:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as ie:
            raise ImportError(
                "`scikit-learn` is not installed. Please install it using `pip install huggingface-hub`."
            ) from ie

        self._NearestNeighbors = NearestNeighbors

    @property
    def inputs(self) -> List[str]:
        return ["evol_instruction_score", "evol_response_score", "embedding"]

    @property
    def outputs(self) -> List[str]:
        return ["deita_score", "nearest_neighbor_distance"]

    def _compute_deita_score(self, inputs: StepInput) -> StepInput:
        for input in inputs:
            input["deita_score"] = (
                input["evol_instruction_score"] * input["evol_response_score"]
            )
        return inputs

    def _compute_nearest_neighbor(self, inputs: StepInput) -> StepInput:
        embeddings = [input["embedding"] for input in inputs]
        nn = self._NearestNeighbors(
            n_neighbors=2, metric="cosine", algorithm="brute"
        ).fit(embeddings)
        distances, _ = nn.kneighbors(embeddings, return_distance=True)
        for distance, input in zip(distances, inputs):
            input["nearest_neighbor_distance"] = distance
        return inputs

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        inputs = self._compute_deita_score(inputs)
        inputs = self._compute_nearest_neighbor(inputs)
        inputs.sort(key=lambda x: x["deita_score"])

        selected_rows = []
        for input in inputs:
            if len(selected_rows) >= self.data_budget:  # type: ignore
                break
            if input["nearest_neighbor_distance"] >= self.diversity_threshold:
                selected_rows.append(input)
        yield selected_rows
