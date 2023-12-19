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
from typing import ClassVar, Dict, List

from distilabel.tasks.base import get_template
from distilabel.tasks.critique.base import CritiqueTask, CritiqueTaskOutput
from distilabel.tasks.prompt import Prompt

_PROMETHEUS_TEMPLATE = get_template("prometheus.jinja2")


@dataclass
class PrometheusTask(CritiqueTask):
    scoring_criteria: str
    score_descriptions: Dict[int, str]

    system_prompt: str = "You are a fair evaluator language model."

    __jinja2_template__: ClassVar[str] = _PROMETHEUS_TEMPLATE

    @property
    def input_args_names(self) -> List[str]:
        return super().input_args_names + ["ref_completion"]

    def generate_prompt(
        self,
        instruction: str,
        completion: str,
        ref_completion: str,
    ) -> str:
        render_kwargs = {
            "instruction": instruction,
            "completion": completion,
            "ref_completion": ref_completion,
            "scoring_criteria": self.scoring_criteria,
            "score_descriptions": self.score_descriptions,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        ).format_as(format="llama2")  # type: ignore

    def parse_output(self, output: str) -> CritiqueTaskOutput:  # type: ignore
        """Parses the output of the model into the desired format."""
        # We use a regex instead of splitting by the delimiter because the
        # critique may contain the delimiter, and using the regex is safer.
        pattern = r"(.+?)\. \[RESULT\] (\d+)"
        match = re.match(pattern, output)
        if match:
            return CritiqueTaskOutput(
                score=float(match.group(2)),
                critique=match.group(1).strip(),
            )
