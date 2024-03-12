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

from enum import EnumType
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field
from typing_extensions import override

from distilabel.steps.task.base import Task
from distilabel.steps.task.evol_instruct.utils import MutationTemplates
from distilabel.steps.task.typing import ChatType

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class EvolInstruct(Task):
    """WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Reference:
        - https://arxiv.org/abs/2304.12244
        - https://github.com/h2oai/h2o-wizardlm
    """

    num_evolutions: int
    store_evolutions: bool = False
    generate_answers: bool = False
    mutation_templates: EnumType = Field(default=MutationTemplates)

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

        np.random.seed(42)

    @cached_property
    def _english_nouns(self) -> List[str]:
        """A list of English nouns to be used as part of the starting prompts for the task.

        Reference:
            - https://github.com/h2oai/h2o-wizardlm
        """
        _path = str(
            importlib_resources.files("distilabel")
            / "steps/task/evol_instruct/english_nouns.txt"
        )
        with open(_path, mode="r") as f:
            return [line.strip() for line in f.readlines()]

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["instruction"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [{"role": "user", "content": input}]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        # TODO: having to define a `model_name` column every time as the `Task.outputs` is not ideal,
        # this could be handled always and the value could be included within the DAG validation when
        # a `Task` is used, since all the `Task` subclasses will have an `llm` with a `model_name` attr.
        _outputs = [
            "evolved_instruction"
            if not self.store_evolutions
            else "evolved_instructions",
            "model_name",
        ]
        if self.generate_answers:
            _outputs.append("answer")
        return _outputs

    def format_output(
        self, instructions: Union[str, List[str]], answer: Optional[str] = None
    ) -> Dict[str, Any]:  # type: ignore
        """The output is formatted as a dictionary with the `instruction`. The `model_name`
        will be automatically included."""
        if not self.output_mappings:
            self.output_mappings = {k: k for k in self.outputs}

        _output = {}
        if not self.store_evolutions:
            _output[self.output_mappings["evolved_instruction"]] = instructions[-1]
        else:
            _output[self.output_mappings["evolved_instructions"]] = instructions

        if self.generate_answers and answer is not None:
            _output[self.output_mappings["answer"]] = answer

        _output[self.output_mappings["model_name"]] = self.llm.model_name
        return _output

    @override
    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        enum_attributes = [
            member.name  # type: ignore
            for member in self.mutation_templates.__members__.values()  # type: ignore
        ]

        instructions: List[List[str]] = [[input["instruction"]] for input in inputs]
        for iter_no in range(self.num_evolutions):
            formatted_prompts = []
            for instruction in instructions:
                mutation = np.random.choice(enum_attributes)
                formatted_prompts.append(
                    self.mutation_templates[mutation].value.replace(  # type: ignore
                        "<PROMPT>", instruction[-1]
                    )
                )

            formatted_prompts = [
                self.format_input(prompt) for prompt in formatted_prompts
            ]
            generated_prompts = self.llm.generate(
                formatted_prompts,
                **self.generation_kwargs,  # type: ignore
            )

            evolved_instructions = []
            for generated_prompt in generated_prompts:
                evolved_instructions.append(
                    generated_prompt.split("Prompt#:")[-1].strip()
                )

            if self.store_evolutions:
                instructions = [
                    instruction + [evolved_instruction]
                    for instruction, evolved_instruction in zip(
                        instructions, evolved_instructions
                    )
                ]
            else:
                instructions = [
                    [evolved_instruction]
                    for evolved_instruction in evolved_instructions
                ]

            self._logger.info(
                f"🔄 Ran iteration {iter_no} evolving {len(instructions)} instructions!"
            )

        self._logger.info(f"🎉 Finished evolving {len(instructions)} instructions!")

        if self.generate_answers:
            self._logger.info(
                f"🧠 Generating answers for the {len(instructions)} evolved instructions!"
            )

            _formatted_instructions = [
                self.format_input(instruction[-1]) for instruction in instructions
            ]
            answers = self.llm.generate(
                _formatted_instructions,
                **self.generation_kwargs,  # type: ignore
            )

            self._logger.info(
                f"🎉 Finished generating answers for the {len(instructions)} evolved instructions!"
            )

            for input, instruction, answer in zip(inputs, instructions, answers):
                input.update(self.format_output(instruction, answer))
            yield inputs

        for input, instruction in zip(inputs, instructions):
            input.update(self.format_output(instruction))
        yield inputs
