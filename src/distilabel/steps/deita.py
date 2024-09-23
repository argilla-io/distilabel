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

from typing import TYPE_CHECKING, List, Literal

import numpy as np
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class DeitaFiltering(GlobalStep):
    """Filter dataset rows using DEITA filtering strategy.

    Filter the dataset based on the DEITA score and the cosine distance between the embeddings.
    It's an implementation of the filtering step from the paper 'What Makes Good Data
    for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning'.

    Attributes:
        data_budget: The desired size of the dataset after filtering.
        diversity_threshold: If a row has a cosine distance with respect to it's nearest
            neighbor greater than this value, it will be included in the filtered dataset.
            Defaults to `0.9`.
        normalize_embeddings: Whether to normalize the embeddings before computing the cosine
            distance. Defaults to `True`.

    Runtime parameters:
        - `data_budget`: The desired size of the dataset after filtering.
        - `diversity_threshold`: If a row has a cosine distance with respect to it's nearest
            neighbor greater than this value, it will be included in the filtered dataset.

    Input columns:
        - evol_instruction_score (`float`): The score of the instruction generated by
            `ComplexityScorer` step.
        - evol_response_score (`float`): The score of the response generated by
            `QualityScorer` step.
        - embedding (`List[float]`): The embedding generated for the conversation of the
            instruction-response pair using `GenerateEmbeddings` step.

    Output columns:
        - deita_score (`float`): The DEITA score for the instruction-response pair.
        - deita_score_computed_with (`List[str]`): The scores used to compute the DEITA
            score.
        - nearest_neighbor_distance (`float`): The cosine distance between the embeddings
            of the instruction-response pair.

    Categories:
        - filtering

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)

    Examples:
        Filter the dataset based on the DEITA score and the cosine distance between the embeddings:

        ```python
        from distilabel.steps import DeitaFiltering

        deita_filtering = DeitaFiltering(data_budget=1)

        deita_filtering.load()

        result = next(
            deita_filtering.process(
                [
                    {
                        "evol_instruction_score": 0.5,
                        "evol_response_score": 0.5,
                        "embedding": [-8.12729941, -5.24642847, -6.34003029],
                    },
                    {
                        "evol_instruction_score": 0.6,
                        "evol_response_score": 0.6,
                        "embedding": [2.99329242, 0.7800932, 0.7799726],
                    },
                    {
                        "evol_instruction_score": 0.7,
                        "evol_response_score": 0.7,
                        "embedding": [10.29041806, 14.33088073, 13.00557506],
                    },
                ],
            )
        )
        # >>> result
        # [{'evol_instruction_score': 0.5, 'evol_response_score': 0.5, 'embedding': [-8.12729941, -5.24642847, -6.34003029], 'deita_score': 0.25, 'deita_score_computed_with': ['evol_instruction_score', 'evol_response_score'], 'nearest_neighbor_distance': 1.9042812683723933}]
        ```

    Citations:
        ```
        @misc{liu2024makesgooddataalignment,
            title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning},
            author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
            year={2024},
            eprint={2312.15685},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2312.15685},
        }
        ```
    """

    data_budget: RuntimeParameter[int] = Field(
        default=None, description="The desired size of the dataset after filtering."
    )
    diversity_threshold: RuntimeParameter[float] = Field(
        default=0.9,
        description="If a row has a cosine distance with respect to it's nearest neighbor"
        " greater than this value, it will be included in the filtered dataset.",
    )
    normalize_embeddings: RuntimeParameter[bool] = Field(
        default=True,
        description="Whether to normalize the embeddings before computing the cosine distance.",
    )
    distance_metric: RuntimeParameter[Literal["cosine", "manhattan"]] = Field(
        default="cosine",
        description="The distance metric to use. Currently only 'cosine' is supported.",
    )

    @property
    def inputs(self) -> List[str]:
        return ["evol_instruction_score", "evol_response_score", "embedding"]

    @property
    def outputs(self) -> List[str]:
        return ["deita_score", "nearest_neighbor_distance", "deita_score_computed_with"]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Filter the dataset based on the DEITA score and the cosine distance between the
        embeddings.

        Args:
            inputs: The input data.

        Returns:
            The filtered dataset.
        """
        inputs = self._compute_deita_score(inputs)
        inputs = self._compute_nearest_neighbor(inputs)
        inputs.sort(key=lambda x: x["deita_score"], reverse=True)

        selected_rows = []
        for input in inputs:
            if len(selected_rows) >= self.data_budget:  # type: ignore
                break
            if input["nearest_neighbor_distance"] >= self.diversity_threshold:
                selected_rows.append(input)
        yield selected_rows

    def _compute_deita_score(self, inputs: StepInput) -> StepInput:
        """Computes the DEITA score for each instruction-response pair. The DEITA score is
        the product of the instruction score and the response score.

        Args:
            inputs: The input data.

        Returns:
            The input data with the DEITA score computed.
        """
        for input_ in inputs:
            evol_instruction_score = input_.get("evol_instruction_score")
            evol_response_score = input_.get("evol_response_score")

            if evol_instruction_score and evol_response_score:
                deita_score = evol_instruction_score * evol_response_score
                score_computed_with = ["evol_instruction_score", "evol_response_score"]
            elif evol_instruction_score:
                self._logger.warning(
                    "Response score is missing for the instruction-response pair. Using"
                    " instruction score as DEITA score."
                )
                deita_score = evol_instruction_score
                score_computed_with = ["evol_instruction_score"]
            elif evol_response_score:
                self._logger.warning(
                    "Instruction score is missing for the instruction-response pair. Using"
                    " response score as DEITA score."
                )
                deita_score = evol_response_score
                score_computed_with = ["evol_response_score"]
            else:
                self._logger.warning(
                    "Instruction and response scores are missing for the instruction-response"
                    " pair. Setting DEITA score to 0."
                )
                deita_score = 0
                score_computed_with = []

            input_.update(
                {
                    "deita_score": deita_score,
                    "deita_score_computed_with": score_computed_with,
                }
            )
        return inputs

    def _compute_nearest_neighbor(self, inputs: StepInput) -> StepInput:
        """Computes the cosine distance between the embeddings of the instruction-response
        pairs and the nearest neighbor.

        Args:
            inputs: The input data.

        Returns:
            The input data with the cosine distance computed.
        """
        embeddings = np.array([input["embedding"] for input in inputs])
        if self.normalize_embeddings:
            embeddings = self._normalize_embeddings(embeddings)
        self._logger.info("📏 Computing nearest neighbor distance...")

        if self.distance_metric == "cosine":
            self._logger.info("📏 Using cosine distance.")
            distances = self._cosine_distance(embeddings)
        else:
            self._logger.info("📏 Using manhattan distance.")
            distances = self._manhattan_distance(embeddings)

        for distance, input in zip(distances, inputs):
            input["nearest_neighbor_distance"] = distance
        return inputs

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize the embeddings.

        Args:
            embeddings: The embeddings to normalize.

        Returns:
            The normalized embeddings.
        """
        self._logger.info("⚖️ Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _cosine_distance(self, embeddings: np.array) -> np.array:  # type: ignore
        """Computes the cosine distance between the embeddings.

        Args:
            embeddings: The embeddings.

        Returns:
            The cosine distance between the embeddings.
        """
        cosine_similarity = np.dot(embeddings, embeddings.T)
        cosine_distance = 1 - cosine_similarity
        # Ignore self-distance
        np.fill_diagonal(cosine_distance, np.inf)
        return np.min(cosine_distance, axis=1)

    def _manhattan_distance(self, embeddings: np.array) -> np.array:  # type: ignore
        """Computes the manhattan distance between the embeddings.

        Args:
            embeddings: The embeddings.

        Returns:
            The manhattan distance between the embeddings.
        """
        manhattan_distance = np.abs(embeddings[:, None] - embeddings).sum(-1)
        # Ignore self-distance
        np.fill_diagonal(manhattan_distance, np.inf)
        return np.min(manhattan_distance, axis=1)
