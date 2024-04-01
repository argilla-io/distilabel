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

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

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
    """QualityScorer is a pre-defined task that defines the `instruction` as the input
    and `score` as the output. This task is used to rate the quality of instructions and responses.
    It's an implementation of the quality score task from the paper 'What Makes Good Data
    for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning'.
    The task follows the same scheme as the Complexity Scorer, but the instruction-response pairs
    are scored in terms of quality, obtaining a quality score for each instruction.

    Attributes:
        _template: a Jinja2 template used to format the input for the LLM.

    Input columns:
        - instruction (`str`): The instruction that was used to generate the `responses`.
        - responses (`List[str]`): The responses to be scored. Each response forms a pair with the instruction.

    Output columns:
        - quality_score (`List[float]`): The quality score for each instruction.

    References:
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

    def format_input(self, input: Dict[str, Any]) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [{"role": "user", "content": self._template.render(**input)}]  # type: ignore

    @property
    def outputs(self):
        """The output for the task is a list of `quality_scores` containing the quality score for each
        response in `responses`."""
        return ["scores"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the score of each instruction-response pair.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with containing the scores for each instruction-response pair.
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
