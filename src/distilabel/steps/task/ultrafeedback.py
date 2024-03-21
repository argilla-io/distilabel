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

from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.task.base import Task
from distilabel.steps.task.typing import ChatType

_ULTRAFEEDBACK_TEMPLATE = """
{{ task_description }}
{%- for rating in ratings %}
{{ rating.value }}. {{ rating.description }}
{%- endfor %}

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
{% for index in range(responses|length) %}
<text {{ index + 1}}> [Text {{ index + 1}}]
{%- endfor %}

### Output
{%- for index in range(responses|length) %}

#### Output for Text {{ index + 1}}
Rating: [Rating for text {{ index + 1}}]
Rationale: [Rationale for the rating in short sentences]

{%- endfor %}

---

## Annotation

### Input
Instruction: {{ input }}

Texts:
{% for response in responses %}
<text {{ loop.index }}> {{ response }}
{%- endfor %}

### Output
""".lstrip()


class UltraFeedback(Task):
    """UltraFeedback: Boosting Language Models with High-quality Feedback.

    References:
        - [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377)
        - [`UltraFeedback - GitHub Repository`](https://github.com/OpenBMB/UltraFeedback)

    Columns:

    - `input`: instruction, generations
    - `output`: ratings, rationales, model_name
    """

    task_description: Optional[str] = None
    ratings: Optional[List[Dict[str, Any]]] = None
    system_prompt: str = (
        "Your role is to evaluate text quality based on given criteria."
    )

    _template: Template = PrivateAttr(default=...)

    def load(self) -> None:
        super().load()

        self._template = Template(_ULTRAFEEDBACK_TEMPLATE)

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`, and the `generations` for it."""
        return ["instruction", "generations"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "input": input["instruction"],
            "responses": input["generations"],
        }
        messages.append(
            {"role": "user", "content": self._template.render(**render_kwargs)},
        )
        return messages

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["ratings", "rationales", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `ratings` and `rationales` for
        each of the provided `generations` for the given `instruction`. The `model_name`
        will be automatically included within the `process` method of `Task`.

        Args:
            output: a string representing the output of the LLM via the `process` method.
            input: the input to the task, as required by some tasks to format the output.

        Returns:
            A dictionary containing the `ratings` and `rationales` for each of the provided
            `generations` for the given `instruction`.
        """
        formatted_output = {"rationales": [], "ratings": []}

        if output:
            for section in output.split("#### Output for Text ")[1:]:
                rating, rationale = section.split("\n")[1:3]

                rating = float(rating.split(": ")[1])
                formatted_output["ratings"].append(rating)

                rationale = rationale.split(": ")[1]
                formatted_output["rationales"].append(rationale)

        return formatted_output
