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
from distilabel.tasks.preference.base import PreferenceTask
from distilabel.tasks.prompt import Prompt

_EVOL_COMPLEXITY_SCORER_TEMPLATE = get_template("evol-complexity-scorer.jinja2")


@dataclass
class EvolComplexityScorerTask(PreferenceTask):
    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_COMPLEXITY_SCORER_TEMPLATE

    def generate_prompt(self, generations: List[str], **_: Any) -> Prompt:
        render_kwargs = {
            "instructions": generations,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def input_args_names(self) -> List[str]:
        """Returns the names of the input arguments of the task."""
        return ["generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["ranks"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the task."""
        output = output.lower().split("\n")
        scores = [int(re.sub(r"\[\d+\] score:", "", o).strip()) for o in output]
        return {self.output_args_names[0]: scores}
