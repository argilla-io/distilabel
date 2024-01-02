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
    __jinja2_template__: ClassVar[str] = _ULTRACM_TEMPLATE

    system_prompt: str = (
        "User: A one-turn chat between a curious user and an artificial intelligence"
        " assistant. The assistant gives helpful, very detailed, and polite answers to"
        " the user's questions.</s>"
    )

    def generate_prompt(self, input: str, generations: str, **_: Any) -> Prompt:
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
