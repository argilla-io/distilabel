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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Step

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class PairRM(Step):
    """Rank the candidates based on the input using the `LLM` model.

    Attributes:
        model: The model to use for the ranking. Defaults to `"llm-blender/PairRM"`.
        input_batch_size: The batch size to use when processing the input. Defaults to `8`.
        instructions: The instructions to use for the model. Defaults to `None`.

    Input columns:
        - inputs (`List[Dict[str, Any]]`): The input text or conversation to rank the candidates for.
        - candidates (`List[Dict[str, Any]]`): The candidates to rank.

    Output columns:
        - ranks (`List[int]`): The ranks of the candidates based on the input.
        - ranked_candidates (`List[Dict[str, Any]]`): The candidates ranked based on the input.

    References:
        - [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561).
        - [Pair Ranking Model](https://huggingface.co/llm-blender/PairRM).

    Note:
        This step differs to other tasks as there is a single implementation of this model
        currently, and we will use a specific `LLM`.
    """

    model: str = "llm-blender/PairRM"
    input_batch_size: int = 8
    instructions: Optional[str] = None

    def load(self) -> None:
        try:
            import llm_blender
        except ImportError as e:
            raise ImportError(
                "The `llm_blender` package is required to use the `PairRM` class."
                "Please install it with `pip install git+https://github.com/yuchenlin/LLM-Blender.git`."
            ) from e
        self._blender = llm_blender.Blender()
        self._blender.loadranker(self.model)

    @property
    def inputs(self) -> List[str]:
        """The input columns correspond to the two required arguments from `Blender.rank`:
        `inputs` and `candidates`."""
        return ["input", "candidates"]

    @property
    def outputs(self) -> List[str]:
        """The outputs will include the `ranks` and the `ranked_candidates`."""
        return ["ranks", "ranked_candidates"]

    def format_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """The input is expected to be a dictionary with the keys `input` and `candidates`,
        where the `input` corresponds to the instruction of a model and `candidates` are a
        list of responses to be ranked.
        """
        return input

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Generates the ranks for the candidates based on the input.

        The ranks are the positions of the candidates, where lower is better,
        and the ranked candidates correspond to the candidates sorted according to the
        ranks obtained.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            An iterator with the inputs containing the `ranks` and the `ranked_candidates`.
        """
        input_texts = []
        candidates = []
        for input in inputs:
            formatted_input = self.format_input(input)
            input_texts.append(formatted_input["input"])
            candidates.append(formatted_input["candidates"])

        instructions = (
            [self.instructions] * len(input_texts) if self.instructions else None
        )

        ranks = self._blender.rank(
            input_texts,
            candidates,
            instructions=instructions,
            return_scores=False,
            batch_size=self.input_batch_size,
        )
        # Sort the candidates based on the ranks
        ranked_candidates = np.take_along_axis(
            np.array(candidates), ranks - 1, axis=1
        ).tolist()
        ranks = ranks.tolist()
        for input, rank, ranked_candidate in zip(inputs, ranks, ranked_candidates):
            input["ranks"] = rank
            input["ranked_candidates"] = ranked_candidate

        yield inputs
