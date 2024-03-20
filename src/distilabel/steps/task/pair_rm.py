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

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from distilabel.steps.task.base import Step

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class PairRM(Step):
    """Rank the candidates based on the input using the `LLM` model.

    Input columns:
        inputs (`List[Dict[str, Any]]`): The input text or conversation to rank the candidates for.
        candidates (`List[Dict[str, Any]]`): The candidates to rank.

    Output columns:
        ranks (`List[int]`): The ranks of the candidates based on the input.
        ranked_candidates (`List[Dict[str, Any]]`): The candidates ranked based on the input.

    References:
        - [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561).
        - [Pair Ranking Model](https://huggingface.co/llm-blender/PairRM).
    """

    model: str = "llm-blender/PairRM"
    batch_size: int = 8

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
        return ["inputs", "candidates"]

    @property
    def outputs(self) -> List[str]:
        return ["ranks", "ranked_candidates"]

    def format_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: This must prepare the data with the inputs and the candidate texts
        return {"inputs": input["instruction"], "candidates": input["responses"]}

    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore
        input_texts = []
        candidates = []
        for input in inputs:
            formatted_input = self.format_input(input)
            input_texts.append(formatted_input["inputs"])
            candidates.append(formatted_input["candidates"])

        instructions = None  # NOTE: How can we pass these automatically?
        ranks = self._blender.rank(
            inputs,
            candidates,
            instructions=instructions,
            return_scores=False,
            batch_size=self.batch_size,
        )
        inputs["ranks"] = ranks.tolist()
        inputs["ranked_candidates"] = np.take_along_axis(
            np.array(inputs["candidates"]), ranks - 1, axis=1
        )
        yield inputs
