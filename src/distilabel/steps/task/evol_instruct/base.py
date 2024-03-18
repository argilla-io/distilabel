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

if sys.version_info < (3, 11):
    from enum import EnumMeta as EnumType
else:
    from enum import EnumType

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field
from typing_extensions import override

from distilabel.steps.base import RuntimeParameter
from distilabel.steps.task.base import Task
from distilabel.steps.task.evol_instruct.utils import (
    MutationTemplatesEvolComplexity,
    MutationTemplatesEvolInstruct,
)
from distilabel.steps.task.typing import ChatType

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.typing import StepOutput


class EvolInstruct(Task):
    """WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Reference:
        - https://arxiv.org/abs/2304.12244
        - https://github.com/h2oai/h2o-wizardlm
        - https://github.com/nlpxucan/WizardLM/Evol_Instruct

    Runtime parameters:

    - `seed`: The number of evolutions to be run.

    Columns:

    - `input`: instruction
    - `output`: there's multiple scenarios:
        - `store_evolutions=False`, `generate_answers=False` -> (evolved_instruction, model_name)
        - `store_evolutions=True`, `generate_answers=False` -> (evolved_instructions, model_name)
        - `store_evolutions=False`, `generate_answers=True` -> (evolved_instruction, model_name, answer)
        - `store_evolutions=True`, `generate_answers=True` -> (evolved_instructions, model_name, answer)
    """

    num_evolutions: int
    store_evolutions: bool = False
    generate_answers: bool = False
    mutation_templates: EnumType = Field(default=MutationTemplatesEvolInstruct)

    seed: RuntimeParameter[int] = Field(
        default=42,
        description="As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
    )

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

        np.random.seed(self.seed)

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
        """The output for the task are the `evolved_instruction/s`, the `answer` if `generate_answers=True`
        and the `model_name`."""
        # TODO: having to define a `model_name` column every time as the `Task.outputs` is not ideal,
        # this could be handled always and the value could be included within the DAG validation when
        # a `Task` is used, since all the `Task` subclasses will have an `llm` with a `model_name` attr.
        _outputs = [
            (
                "evolved_instruction"
                if not self.store_evolutions
                else "evolved_instructions"
            ),
            "model_name",
        ]
        if self.generate_answers:
            _outputs.append("answer")
        return _outputs

    def format_output(
        self, instructions: Union[str, List[str]], answer: Optional[str] = None
    ) -> Dict[str, Any]:  # type: ignore
        """The output for the task is a dict with: `evolved_instruction` or `evolved_instructions`,
        depending whether the value is either `False` or `True` for `store_evolutions`, respectively;
        `answer` if `generate_answers=True`; and, finally, the `model_name`.

        Args:
            instructions: The instructions to be included within the output.
            answer: The answer to be included within the output if `generate_answers=True`.

        Returns:
            If `store_evolutions=False` and `generate_answers=True` return {"evolved_instruction": ..., "model_name": ..., "answer": ...};
            if `store_evolutions=True` and `generate_answers=True` return {"evolved_instructions": ..., "model_name": ..., "answer": ...};
            if `store_evolutions=False` and `generate_answers=False` return {"evolved_instruction": ..., "model_name": ...};
            if `store_evolutions=True` and `generate_answers=False` return {"evolved_instructions": ..., "model_name": ...}.
        """
        _output = {}
        if not self.store_evolutions:
            _output["evolved_instruction"] = instructions[-1]
        else:
            _output["evolved_instructions"] = instructions

        if self.generate_answers and answer is not None:
            _output["answer"] = answer

        _output["model_name"] = self.llm.model_name
        return _output

    @property
    def mutation_templates_names(self) -> List[str]:
        """Returns the names i.e. keys of the provided `mutation_templates` enum."""
        return [
            member.name  # type: ignore
            for member in self.mutation_templates.__members__.values()  # type: ignore
        ]

    def _apply_random_mutation(self, instruction: str) -> str:
        """Applies a random mutation from the ones provided as part of the `mutation_templates`
        enum, and returns the provided instruction within the mutation prompt.

        Args:
            instruction: The instruction to be included within the mutation prompt.

        Returns:
            A random mutation prompt with the provided instruction.
        """
        mutation = np.random.choice(self.mutation_templates_names)
        return self.mutation_templates[mutation].value.replace("<PROMPT>", instruction)  # type: ignore

    def _evolve_instructions(self, inputs: "StepInput") -> List[List[str]]:
        """Evolves the instructions provided as part of the inputs of the task.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list where each item is a list with either the last evolved instruction if
            `store_evolutions=False` or all the evolved instructions if `store_evolutions=True`.
        """

        instructions: List[List[str]] = [[input["instruction"]] for input in inputs]

        for iter_no in range(self.num_evolutions):
            formatted_prompts = []
            for instruction in instructions:
                formatted_prompts.append(self._apply_random_mutation(instruction[-1]))

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

        return instructions

    def _generate_answers(self, instructions: List[List[str]]) -> List[str]:
        """Generates the answer for the last instruction in `instructions`.

        Args:
            instructions: A list of lists where each item is a list with either the last
                evolved instruction if `store_evolutions=False` or all the evolved instructions
                if `store_evolutions=True`.
        Returns:
            A list of answers for the last instruction in `instructions`.
        """
        _formatted_instructions = [
            self.format_input(instruction[-1]) for instruction in instructions
        ]
        return self.llm.generate(
            _formatted_instructions,
            **self.generation_kwargs,  # type: ignore
        )

    @override
    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """

        instructions = self._evolve_instructions(inputs)

        if self.store_evolutions:
            # Remove the input instruction from the `evolved_instructions` list
            instructions = [instruction[1:] for instruction in instructions]

        if not self.generate_answers:
            for input, instruction in zip(inputs, instructions):
                input.update(self.format_output(instruction))
            yield inputs

        self._logger.info(f"🎉 Finished evolving {len(instructions)} instructions!")

        if self.generate_answers:
            self._logger.info(
                f"🧠 Generating answers for the {len(instructions)} evolved instructions!"
            )

            answers = self._generate_answers(instructions)

            self._logger.info(
                f"🎉 Finished generating answers for the {len(instructions)} evolved instructions!"
            )

            for idx, (input, instruction) in enumerate(zip(inputs, instructions)):
                input.update(self.format_output(instruction, answers[idx]))
            yield inputs


class EvolComplexity(EvolInstruct):
    """
    What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
    and
    WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Reference:

        - https://arxiv.org/abs/2312.15685
        - https://arxiv.org/abs/2304.12244

    Runtime parameters:

    - `seed`: The number of evolutions to be run.

    Columns:

    - `input`: instruction
    - `output`: there's multiple scenarios:
        - `store_evolutions=False`, `generate_answers=False` -> (evolved_instruction, model_name)
        - `store_evolutions=True`, `generate_answers=False` -> (evolved_instructions, model_name)
        - `store_evolutions=False`, `generate_answers=True` -> (evolved_instruction, model_name, answer)
        - `store_evolutions=True`, `generate_answers=True` -> (evolved_instructions, model_name, answer)
    """

    mutation_templates: EnumType = Field(default=MutationTemplatesEvolComplexity)
