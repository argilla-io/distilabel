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

_EVOL_COMPLEXITY_SCORER_TEMPLATE = get_template("evol-complexity-scorer.jinja2")


@dataclass
class EvolComplexityScorerTask(PreferenceTaskNoRationale):
    """A `PreferenceTask` following the `Complexity Scorer` specification for ranking and scoring instructions
    in terms of complexity.

    This task is inspired by the Evol Complexity Scorer in the Deita framework: *Deita is an open-sourced project
    designed to facilitate Automatic Data Selection for instruction tuning in Large Language Models (LLMs).*

    The task is defined as follows:
    Ask an LLM (in the original paper they used ChatGPT) to rank and score the instructions (the number of instructions
    is dynamic in the sense that you can compare any number, in *Deita* the chose 6) to obtain a complexity
    score *c* for each instruction.

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_COMPLEXITY_SCORER_TEMPLATE

    def generate_prompt(self, generations: List[str], **_: Any) -> Prompt:
        """Generates a prompt following the *Evol Complexity* specification in *Deita*.

        Args:
            generations (List[str]): the generations to be used for the prompt.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.preference import EvolComplexityScorerTask
            >>> task = EvolComplexityScorerTask()
            >>> task.generate_prompt(["instruction 1", "instruction 2"])
            Prompt(
                system_prompt='',
                formatted_prompt='Ranking the following questions according to the difficulty and complexity. Score 1-2.\nYou
            can give a score of 3 if the question is too complex for you to answer it. You should\nrespond with the
            format:\n[1] Score: 1\n[2] Score: 2\n...\n\n[1] instruction 1\n[2] instruction 2'
            )
        """
        render_kwargs = {"instructions": generations}
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def input_args_names(self) -> List[str]:
        """Returns the names of the input arguments of the task."""
        return ["generations"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the task, returning a list with the rank/score of each instruction.

        Args:
            output (str): The output of the LLM raw.

        Returns:
            Dict[str, List[str]]: A dict with containing the ranks/scores of each instruction.
        """
        output = output.lower().split("\n")
        scores = [float(re.sub(r"\[\d+\] score:", "", o).strip()) for o in output]
        return {self.output_args_names[0]: scores}
