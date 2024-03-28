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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType

_COMPLEXITY_SCORER_TEMPLATE = """
Ranking the following questions according to the difficulty and complexity. Score 1-{{ instructions|length }}.
You can give a score of {{ (instructions|length) + 1 }} if the question is too complex for you to answer it. You should
respond with the format:
[1] Score: 1
[2] Score: 2
...
{% for instruction in instructions %}
[{{ loop.index }}] {{ instruction }}
{%- endfor %}
""".lstrip()

_PARSE_SCORE_LINE_REGEX = re.compile(r"\[\d+\] score: (\d+)", re.IGNORECASE)


class ComplexityScorer(Task):
    """This task is used to rank a list of instructions based on their complexity. It's
    an implementation of the complexity score task from the paper 'What Makes Good Data
    for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning'.

    Attributes:
        _template: The Jinja2 template used to format the input data.

    Input columns:
        - instructions (`List[str]`): The list of instructions to be scored.

    Output columns:
        - complexity_score (`List[float]`): The complexity score for each instruction.

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(_COMPLEXITY_SCORER_TEMPLATE)

    @property
    def inputs(self) -> List[str]:
        return ["instructions"]

    @property
    def outputs(self) -> List[str]:
        return ["scores"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [{"role": "user", "content": self._template.render(**input)}]  # type: ignore

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        if output is None:
            return {"scores": [None] * len(input["instructions"])}

        scores = []
        score_lines = output.split("\n")
        for i, line in enumerate(score_lines):
            match = _PARSE_SCORE_LINE_REGEX.match(line)
            score = float(match.group(1)) if match else None
            scores.append(score)
            if i == len(input["instructions"]) - 1:
                break

        return {"scores": scores}
