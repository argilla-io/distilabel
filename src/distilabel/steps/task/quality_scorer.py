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
from typing import Any, Dict, List, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.task.base import Task
from distilabel.steps.task.typing import ChatType

_QUALITY_SCORER_TEMPLATE = """
Rank the following pair of instructions and responses according to their quality. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Score 1-{{ responses|length }}.
Score each response from 1 to {{ responses|length }}, with {{ (responses|length) + 1}} reserved for responses that are already very well written and cannot be improved further. You should respond with the format:
[1] Score: 1
[2] Score: 2
...
#Question#: {{ instruction }}
#Response List#:
{% for response in responses %}
[{{ loop.index }}] {{ response }}
{%- endfor %}
""".lstrip()

_PARSE_SCORE_LINE_REGEX = re.compile(r"\[\d+\] score: (\d+)", re.IGNORECASE)


class QualityScorer(Task):
    """
    QualityScorerTask is a pre-defined task that defines the `instruction` as the input
    and `score` as the output. This task is used to rate the quality of instructions and responses.
    It is inspired by the Evol Quality Scorer in the Deita framework: : *Deita is an open-sourced project
    designed to facilitate Automatic Data Selection for instruction tuning in Large Language Models (LLMs).*
    The task follows the same scheme as the Evol Complexity Scorer, but the instruction-response pairs are scored in
    terms of quality, obtining a quality score *q* for each instruction.

    Input columns:
        instruction (`List[str]`): The instruction to be scored.
        responses (`List[str]`): The responses to that instruction to be scored. Each response forms a pair with the instruction.

    Output columns:
        quality_score (`List[float]`): The quality score for each instruction.

    Reference:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(_QUALITY_SCORER_TEMPLATE)

    @property
    def inputs(self) -> List[str]:
        """The input for the task are `instruction` and `responses`."""
        return ["instruction", "responses"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [{"role": "user", "content": self._template.render(**input)}]  # type: ignore

    @property
    def outputs(self):
        """The output for the task is the `generation` and the `model_name`."""
        return ["quality_scores"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the rating of each instruction-response tuple.

        Args:
            output (str): the raw output of the LLM.

        Returns:
            Dict[str, List[str]]: A dict with containing the ratings of each instruction-response.
        """

        if output is None:
            return {self.outputs[0]: [None] * len(input["responses"])}

        scores = []
        score_lines = output.split("\n")

        for i, line in enumerate(score_lines):
            match = _PARSE_SCORE_LINE_REGEX.match(line)
            score = float(match.group(1)) if match else None
            scores.append(score)
            if i == len(input["responses"]) - 1:
                break

        return {self.outputs[0]: scores}
