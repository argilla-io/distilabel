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

from distilabel.steps.task.base import Task
from distilabel.steps.task.typing import ChatType

_SELF_INSTRUCT_TEMPLATE = """
# Task Description
Develop {{ num_instructions }} user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model's textual capabilities.

# Criteria for Queries
{{ criteria_for_query_generation }}
Write each query on a separate line and avoid using numbered lists or bullet points.

# AI Application
{{ application_description }}

# Context
{{ input }}

# Output
"""

_PARSE_OUTPUT_REGEX = re.compile(r"(?<=# Output\n)(.*)", re.IGNORECASE)

_DEFAULT_APPLICATION_DESCRIPTION: str = "AI assistant"

_DEFAULT_NUMBER_INSTRUCTIONS: int = 5

_DEFAULT_CRITERIA_FOR_QUERY_GENERATION: str = (
    "Incorporate a diverse range of verbs, avoiding repetition.\n"
    "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.\n"
    "Design queries to be self-contained and standalone.\n"
    'Blend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.'
)


class SelfInstruct(Task):
    """
    SelfInstructTask is a pre-defined task that, given a number of instructions, a certain criteria for query_generation,
    an application description, and an input, generates a number of instruction related to the given input and following
    what is stated in the criteria for query generation and the application description. It is based in the SelfInstruct
    framework from the paper 'Self-Instruct: Aligning Language Models with Self-Generated Instructions'.

    Input columns:
        num_instructions (`int`): The number of instructions to be generated.
        criteria_for_query_generation (`str`): The criteria for the query generation.
        application_description (`str`): The description of the AI application that
            one want to build with these instructions.
        input (`str`): The input to generate the instructions. It's also called seed in the paper

    Output columns:
        instructions (`List[str]`): The generated instructions.

    Reference:
        - [`Self-Instruct: Aligning Language Models with Self-Generated Instructions`](https://arxiv.org/abs/2212.10560)
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(_SELF_INSTRUCT_TEMPLATE)

    @property
    def inputs(self) -> List[str]:
        """The input for the task are `num_instructions`, `criteria_for_query_generation`, `application_description` and `input`."""
        return [
            "num_instructions",
            "criteria_for_query_generation",
            "application_description",
            "input",
        ]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""

        if "num_instructions" not in input.keys():
            input["num_instructions"] = _DEFAULT_NUMBER_INSTRUCTIONS
        if "criteria_for_query_generation" not in input.keys():
            input[
                "criteria_for_query_generation"
            ] = _DEFAULT_CRITERIA_FOR_QUERY_GENERATION
        if "application_description" not in input.keys():
            input["application_description"] = _DEFAULT_APPLICATION_DESCRIPTION

        return [{"role": "user", "content": self._template.render(**input)}]  # type: ignore

    @property
    def outputs(self):
        """The output for the task is a list of `instructions` containing the generated instructions."""
        return ["instructions"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the generated instructions.
        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.
        Returns:
            A dict with containing the generated instructions.
        """

        if output is None:
            return {self.outputs[0]: [None] * len(input["instructions"])}

        instructions = []
        instruction_lines = output.split("\n")

        for i, line in enumerate(instruction_lines):
            match = _PARSE_OUTPUT_REGEX.match(line)
            instruction = float(match.group(1)) if match else None
            instructions.append(instruction)
            if i == len(input["instructions"]) - 1:
                break

        return {self.outputs[0]: instructions}
