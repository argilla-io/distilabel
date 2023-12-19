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
from typing import ClassVar

from distilabel.tasks.base import get_template
from distilabel.tasks.critique.base import CritiqueTask, CritiqueTaskOutput

_ULTRACM_TEMPLATE = get_template("ultracm.jinja2")


@dataclass
class UltraCMTask(CritiqueTask):
    __jinja2_template__: ClassVar[str] = _ULTRACM_TEMPLATE

    system_prompt: str = (
        "User: A one-turn chat between a curious user and an artificial intelligence"
        " assistant. The assistant gives helpful, very detailed, and polite answers to"
        " the user's questions.</s>"
    )

    def generate_prompt(self, instruction: str, completion: str) -> str:
        render_kwargs = {
            "instruction": instruction,
            "completion": completion,
        }
        return f"{self.system_prompt}\nUser: {self.template.render(**render_kwargs)}</s>\nAssistant: ### Feedback\nOverall Score: "

    def parse_output(self, output: str) -> CritiqueTaskOutput:  # type: ignore
        """Parses the output of the model into the desired format."""
        pattern = r"(\d+(?:\.\d+)?)\s*(.*)"
        match = re.match(pattern, output)
        if match:
            return CritiqueTaskOutput(
                score=float(match.group(1)),
                critique=match.group(2).strip(),
            )
