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

from dataclasses import dataclass
from typing import Dict, List

from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

_SELF_INSTRUCT_TEMPLATE = get_template("self-instruct.jinja2")

@dataclass
class SelfInstructTask(TextGenerationTask):
    """A `TextGenerationTask` following the Self-Instruct specification for building
    the prompts.

    Reference: https://github.com/yizhongw/self-instruct

    Args:
        system_prompt (str, optional): the system prompt to be used. Defaults to `None`.
        principles (Dict[str, List[str]], optional): the principles to be used for the system prompt.
            Defaults to `None`.
        principles_distribution (Union[Dict[str, float], Literal["balanced"], None], optional): the
            distribution of principles to be used for the system prompt. Defaults to `None`.
        application_description (str, optional): the description of the AI application. Defaults to
            "AI assistant".
        num_instructions (int, optional): the number of instructions to be used for the prompt.
            Defaults to 5.
    """

    system_prompt: str = (
        "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks."
        "You are given a task description and a set of instructions for how to write the prompts for a specific AI application."
    )
    application_description: str = "AI assistant"
    num_instructions: int = 5

    __jinja2_template__: str = _SELF_INSTRUCT_TEMPLATE

    def generate_prompt(self, input: str) -> Prompt:
        """Generates a prompt following the Self-Instruct specification.

        Args:
            input (str): the input to be used for the prompt.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import SelfInstructTask
            >>> task = SelfInstructTask(system_prompt="You are a helpful assistant.", num_instructions=2)
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?")
            Prompt(
                system_prompt="You are a helpful assistant.",
                formatted_prompt="# Task Description\nDevelop 2 user queries that ...",
            )
        """
        render_kwargs = {
            "application_description": self.application_description,
            "num_instructions": self.num_instructions,
            "input": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format."""
        return {"generations": output.split("\n")}
