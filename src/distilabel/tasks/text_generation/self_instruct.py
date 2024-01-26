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
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.mixins import InstructTaskMixin

_SELF_INSTRUCT_TEMPLATE = get_template("self-instruct.jinja2")


@dataclass
class SelfInstructTask(InstructTaskMixin, TextGenerationTask):
    """A `TextGenerationTask` following the Self-Instruct specification for building
    the prompts.

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
        criteria_for_query_generation (str, optional): the criteria for query generation that we want
            our model to have. Default value covers default behaviour for SelfInstructTask. This value is
            passed to the .jinja template, where extra instructions are added to ensure correct output format.

    References:
        - [`Self-Instruct: Aligning Language Models with Self-Generated Instructions`](https://arxiv.org/abs/2212.10560)
        - [`Self-Instruct - GitHub Repository`](https://github.com/yizhongw/self-instruct)
    """

    system_prompt: str = (
        "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks."
        " You are given a task description and a set of instructions for how to write the prompts for an"
        " specific AI application."
    )

    application_description: str = "AI assistant"
    num_instructions: int = 5

    criteria_for_query_generation: str = (
        "Incorporate a diverse range of verbs, avoiding repetition.\n"
        "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.\n"
        "Design queries to be self-contained and standalone.\n"
        'Blend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.'
    )

    __jinja2_template__: str = _SELF_INSTRUCT_TEMPLATE

    def generate_prompt(self, input: str, **_: Any) -> Prompt:
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
                formatted_prompt="# Task Description ...",
            )
        """
        render_kwargs = {
            "application_description": self.application_description,
            "num_instructions": self.num_instructions,
            "criteria_for_query_generation": self.criteria_for_query_generation,
            "input": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def output_args_names(self) -> List[str]:
        return ["instructions"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format."""
        pattern = re.compile(r"\d+\.\s*(.*?)\n")
        return {"instructions": pattern.findall(output)}
