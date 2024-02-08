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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from distilabel.tasks.base import get_template
from distilabel.tasks.critique.base import CritiqueTask, CritiqueTaskOutput
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from argilla import FeedbackDataset

_ULTRACM_TEMPLATE = get_template("ultracm.jinja2")


@dataclass
class UltraCMTask(CritiqueTask):
    """A `CritiqueTask` following the prompt templated used by UltraCM (from UltraFeedback).

    Args:
        system_prompt (str, optional): the system prompt to be used for generation. Defaults to `None`.

    Disclaimer:
        Since the UltraCM model has been trained with OpenAI API generated data, the prompting
        strategy may just be consistent / compliant with either GPT-3.5 or GPT-4 from OpenAI API, or
        with their own model. Any other model may fail on the generation of a structured output, as
        well as providing an incorrect / inaccurate critique.

    References:
        - [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377)
        - [`UltraFeedback - GitHub Repository`](https://github.com/OpenBMB/UltraFeedback)
        - [`openbmb/UltraCM-13b`](https://huggingface.co/openbmb/UltraCM-13b)
    """

    __jinja2_template__: ClassVar[str] = _ULTRACM_TEMPLATE

    system_prompt: str = (
        "User: A one-turn chat between a curious user and an artificial intelligence"
        " assistant. The assistant gives helpful, very detailed, and polite answers to"
        " the user's questions.</s>"
    )

    def generate_prompt(self, input: str, generations: List[str], **_: Any) -> Prompt:
        """Generates a prompt following the UltraCM specification.

        Args:
            input (str): the input to be used for the prompt.
            generations (List[str]): the generations to be used for the prompt, in
                this case, the ones to be critiqued.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.critique import UltraCMTask
            >>> task = UltraCMTask()
            >>> task.generate_prompt(
            ...     input="What are the first 5 Fibonacci numbers?",
            ...     generations=["0 1 1 2 3", "0 1 1 2 3"],
            ... )
            Prompt(
                system_prompt="User: A one-turn chat between a curious user ...",
                formatted_prompt="User: Given my answer to an instruction, your role ...",
            )
        """
        render_kwargs = {
            "instruction": input,
            "completion": generations,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=f"User: {self.template.render(**render_kwargs)}</s>\nAssistant: ### Feedback\nOverall Score: ",
        )

    def parse_output(self, output: str) -> CritiqueTaskOutput:  # type: ignore
        """Parses the output of the model into the desired format."""
        pattern = r"(\d+(?:\.\d+)?)\s*(.*)"
        match = re.match(pattern, output)
        if match:
            return CritiqueTaskOutput(
                score=float(match.group(1)),
                critique=match.group(2).strip(),
            )

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        score_column: str = "score",
        critique_column: str = "critique",
        score_values: Optional[List[int]] = None,
    ) -> "FeedbackDataset":
        return super().to_argilla_dataset(
            dataset_row=dataset_row,
            generations_column=generations_column,
            score_column=score_column,
            critique_column=critique_column,
            score_values=score_values or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
