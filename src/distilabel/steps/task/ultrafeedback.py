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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import Any, Dict, List, Literal, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.task.base import Task
from distilabel.steps.task.typing import ChatType


class UltraFeedback(Task):
    """UltraFeedback: Boosting Language Models with High-quality Feedback.

    Attributes:
        task: The task to perform with the `UltraFeedback` model. The available tasks are:
            - `helpfulness`: Evaluate text outputs based on helpfulness.
            - `honesty`: Evaluate text outputs based on honesty.
            - `instruction-following`: Evaluate text outputs based on given instructions.
            - `truthfulness`: Evaluate text outputs based on truthfulness.

    Input columns:
        instruction (`str`): The reference instruction to evaluate the text outputs.
        generations (`List[str]`): The text outputs to evaluate for the given instruction.

    Output columns:
        ratings (`List[float]`): The ratings for each of the provided text outputs.
        rationales (`List[str]`): The rationales for each of the provided text outputs.
        model_name (`str`): The name of the model used to generate the ratings and rationales.

    References:
        - [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377)
        - [`UltraFeedback - GitHub Repository`](https://github.com/OpenBMB/UltraFeedback)
    """

    task: Literal[
        "helpfulness",
        "honesty",
        "instruction-following",
        "truthfulness",
    ]

    _system_prompt: str = PrivateAttr(
        default=(
            "Your role is to evaluate text quality based on given criteria.\n"
            'You\'ll receive an instructional description ("Instruction") and {no_texts} text outputs ("Text").\n'
            "Understand and interpret instructions to evaluate effectively.\n"
            "Provide annotations for each text with a rating and rationale.\n"
            "The {no_texts} texts given are independent, and should be evaluated separately.\n"
        )
    )
    _template: Optional["Template"] = PrivateAttr(default=...)

    def load(self) -> None:
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "task"
            / "templates"
            / "ultrafeedback"
            / f"{self.task}.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`, and the `generations` for it."""
        return ["instruction", "generations"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {
                "role": "system",
                "content": self._system_prompt.format(
                    no_texts=len(input["generations"])
                ),
            },
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    instruction=input["instruction"], generations=input["generations"]
                ),
            },
        ]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["ratings", "rationales", "raw", "model_name"]

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
