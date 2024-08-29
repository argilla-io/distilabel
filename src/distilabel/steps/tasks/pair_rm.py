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

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Step

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class PairRM(Step):
    """Rank the candidates based on the input using the `LLM` model.

    Attributes:
        model: The model to use for the ranking. Defaults to `"llm-blender/PairRM"`.
        instructions: The instructions to use for the model. Defaults to `None`.

    Input columns:
        - inputs (`List[Dict[str, Any]]`): The input text or conversation to rank the candidates for.
        - candidates (`List[Dict[str, Any]]`): The candidates to rank.

    Output columns:
        - ranks (`List[int]`): The ranks of the candidates based on the input.
        - ranked_candidates (`List[Dict[str, Any]]`): The candidates ranked based on the input.
        - model_name (`str`): The model name used to rank the candidate responses. Defaults to `"llm-blender/PairRM"`.

    References:
        - [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561).
        - [Pair Ranking Model](https://huggingface.co/llm-blender/PairRM).

    Categories:
        - preference

    Note:
        This step differs to other tasks as there is a single implementation of this model
        currently, and we will use a specific `LLM`.

    Examples:
        Rank LLM candidates:

        ```python
        from distilabel.steps.tasks import PairRM

        # Consider this as a placeholder for your actual LLM.
        pair_rm = PairRM()

        pair_rm.load()

        result = next(
            scorer.process(
                [
                    {"input": "Hello, how are you?", "candidates": ["fine", "good", "bad"]},
                ]
            )
        )
        # result
        # [
        #     {
        #         'input': 'Hello, how are you?',
        #         'candidates': ['fine', 'good', 'bad'],
        #         'ranks': [2, 1, 3],
        #         'ranked_candidates': ['good', 'fine', 'bad'],
        #         'model_name': 'llm-blender/PairRM',
        #     }
        # ]
        ```

    Citations:
        ```
        @misc{jiang2023llmblenderensemblinglargelanguage,
            title={LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion},
            author={Dongfu Jiang and Xiang Ren and Bill Yuchen Lin},
            year={2023},
            eprint={2306.02561},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2306.02561},
        }
        ```
    """

    model: str = "llm-blender/PairRM"
    instructions: Optional[str] = None

    def load(self) -> None:
        """Loads the PairRM model provided via `model` with `llm_blender.Blender`, which is the
        custom library for running the inference for the PairRM models."""
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
    def inputs(self) -> "StepColumns":
        """The input columns correspond to the two required arguments from `Blender.rank`:
        `inputs` and `candidates`."""
        return ["input", "candidates"]

    @property
    def outputs(self) -> "StepColumns":
        """The outputs will include the `ranks` and the `ranked_candidates`."""
        return ["ranks", "ranked_candidates", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """The input is expected to be a dictionary with the keys `input` and `candidates`,
        where the `input` corresponds to the instruction of a model and `candidates` are a
        list of responses to be ranked.
        """
        return {"input": input["input"], "candidates": input["candidates"]}

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Generates the ranks for the candidates based on the input.

        The ranks are the positions of the candidates, where lower is better,
        and the ranked candidates correspond to the candidates sorted according to the
        ranks obtained.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            An iterator with the inputs containing the `ranks`, `ranked_candidates`, and `model_name`.
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
            input["model_name"] = self.model

        yield inputs
