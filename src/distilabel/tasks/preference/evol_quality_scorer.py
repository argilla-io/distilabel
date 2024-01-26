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
from typing import Any, Dict, List

from distilabel.tasks.base import get_template
from distilabel.tasks.preference.base import PreferenceTaskNoRationale
from distilabel.tasks.prompt import Prompt

_EVOL_QUALITY_SCORER_TEMPLATE = get_template("evol-quality-scorer.jinja2")


@dataclass
class EvolQualityScorerTask(PreferenceTaskNoRationale):
    """A `PreferenceTask` following the `Quality Scorer` specification for rating instructions
    in terms of quality.

    This task is inspired by the Evol Quality Scorer in the Deita framework: *Deita is an open-sourced project
    designed to facilitate Automatic Data Selection for instruction tuning in Large Language Models (LLMs).*

    The task follows the same scheme as the Evol Complexity Scorer, but the instructions are scored in terms of
    quality, obtaining a quality score *q* for each instruction.

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    system_prompt: str = ""
    task_description: str = """Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth,
creativity, and level of detail of the response."""
    __jinja2_template__: str = _EVOL_QUALITY_SCORER_TEMPLATE

    def generate_prompt(self, input: str, generations: List[str], **_: Any) -> Prompt:
        """Generates a prompt following the *Evol Quality* specification in *Deita*.

        Args:
            input (str): the instruction for which the model will score the responses.
            generations (List[str]): the generations to be used for the prompt.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.preference import EvolQualityScorerTask
            >>> task = EvolQualityScorerTask()
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?", ["0 1 1 2 3", "0 1 1 2 3"])
            Prompt(
                system_prompt='',
                formatted_prompt='Rank the following responses provided by different AI assistants to the user’s
            question\naccording to the quality of their response. Score each response from 1 to 2, with 3\nreserved for
            responses that are already very well written and cannot be improved further.\n\nUse the following
            format:\n[Response 1] Score:\n[Response 2] Score:\n...\n#Question#: What are the first 5 Fibonacci
            numbers?\n#Response List#:\n\n[Response 1] 0 1 1 2 3\n[Response 2] 0 1 1 6 9'
            )
        """
        render_kwargs = {
            "instruction": input,
            "responses": generations,
            "task_description": self.task_description,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the task, returning a list with the rating of each instruction.

        Args:
            output (str): The output of the LLM raw.

        Returns:
            Dict[str, List[str]]: A dict with containing the ratings of each instruction.
        """
        output = output.lower().split("\n")
        scores = [
            float(re.sub(r"\[response \d+\] score:", "", o).strip()) for o in output
        ]
        return {self.output_args_names[0]: scores}
