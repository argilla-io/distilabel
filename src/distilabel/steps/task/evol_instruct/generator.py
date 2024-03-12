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

from distilabel.steps.base import RuntimeParameter

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from enum import EnumType
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.steps.task.base import GeneratorTask
from distilabel.steps.task.evol_instruct.utils import GenerationMutationTemplates
from distilabel.steps.task.typing import ChatType

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class EvolInstructGenerator(GeneratorTask):
    """WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Reference:
        - https://arxiv.org/abs/2304.12244
        - https://github.com/h2oai/h2o-wizardlm
    """

    num_instructions: int
    generate_answers: bool = False
    mutation_templates: EnumType = Field(default=GenerationMutationTemplates)

    min_length: RuntimeParameter[int] = Field(default=256)
    max_length: RuntimeParameter[int] = Field(default=1024)

    _seed_texts: Optional[List[str]] = PrivateAttr(default_factory=list)
    _prompts: Optional[List[str]] = PrivateAttr(default_factory=list)

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

        np.random.seed(42)

        self._seed_texts = []
        for _ in range(self.num_instructions * 10):
            num_words = np.random.choice([1, 2, 3, 4])
            self._seed_texts.append(
                self.mutation_templates.FRESH_START.value.replace(  # type: ignore
                    "<PROMPT>",
                    ", ".join(
                        [
                            np.random.choice(self._english_nouns).strip()
                            for _ in range(num_words)
                        ]
                    ),
                )
            )

        self._prompts = [
            np.random.choice(self._seed_texts) for _ in range(self.num_instructions)
        ]

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

    def format_input(self, input: Dict[str, Any]) -> ChatType:  # type: ignore
        pass

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `instruction`, the `answer` if `generate_answers=True`
        and the `model_name`."""
        # TODO: having to define a `model_name` column every time as the `Task.outputs` is not ideal,
        # this could be handled always and the value could be included within the DAG validation when
        # a `Task` is used, since all the `Task` subclasses will have an `llm` with a `model_name` attr.
        if self.generate_answers:
            return ["instruction", "answer", "model_name"]
        return ["instruction", "model_name"]

    def format_output(
        self, instruction: str, answer: Optional[str] = None
    ) -> Dict[str, Any]:  # type: ignore
        """The output is formatted as a dictionary with the `instruction`. The `model_name`
        will be automatically included."""
        if not self.output_mappings:
            self.output_mappings = {k: k for k in self.outputs}

        if self.generate_answers:
            return {
                self.output_mappings["instruction"]: instruction,
                self.output_mappings["answer"]: answer,
                self.output_mappings["model_name"]: self.llm.model_name,
            }
        return {
            self.output_mappings["instruction"]: instruction,
            self.output_mappings["model_name"]: self.llm.model_name,
        }

    @override
    def process(self) -> "GeneratorStepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """

        mutated_instructions = []
        mutation_no = 0

        iter_no = 0
        while len(mutated_instructions) < self.num_instructions:
            formatted_prompts = []
            for idx in range(self.num_instructions):
                if (
                    iter_no == 0
                    or "Write one question or request containing" in self._prompts[idx]  # type: ignore
                ):
                    mutation = "FRESH_START"
                else:
                    enum_attributes = [
                        member.name
                        for member in self.mutation_templates.__members__.values()  # type: ignore
                    ]
                    mutation = np.random.choice(enum_attributes)
                    if mutation == "FRESH_START":
                        self._prompts[idx] = np.random.choice(self._seed_texts)  # type: ignore
                formatted_prompts.append(
                    self.mutation_templates[mutation].value.replace(  # type: ignore
                        "<PROMPT>",
                        self._prompts[idx],  # type: ignore
                    )  # type: ignore
                    if iter_no != 0
                    else self._prompts[idx]  # type: ignore
                )

            formatted_prompts = [
                [{"role": "user", "content": prompt}] for prompt in formatted_prompts
            ]
            generated_prompts = self.llm.generate(
                formatted_prompts,
                **self.generation_kwargs,  # type: ignore
            )
            for idx, generated_prompt in enumerate(generated_prompts):
                generated_prompt = generated_prompt.split("Prompt#:")[-1].strip()
                if self.max_length >= len(generated_prompt) >= self.min_length:  # type: ignore
                    mutated_instructions.append(generated_prompt)
                    self._prompts[idx] = np.random.choice(self._seed_texts)  # type: ignore
                else:
                    self._prompts[idx] = generated_prompt  # type: ignore

            self._logger.info(
                f"🔄 Ran iteration {iter_no} with {len(mutated_instructions)} instructions already evolved!"
            )
            iter_no += 1

            if len(mutated_instructions) > self.num_instructions:
                mutated_instructions = mutated_instructions[: self.num_instructions]
            if len(mutated_instructions) > mutation_no:
                mutation_no = len(mutated_instructions) - mutation_no

            if (
                not self.generate_answers
                and len(mutated_instructions[-mutation_no:]) > 0
            ):
                yield (
                    [
                        self.format_output(mutated_instruction)
                        for mutated_instruction in mutated_instructions[-mutation_no:]
                    ],
                    len(mutated_instructions) >= self.num_instructions,
                )

        self._logger.info(
            f"🎉 Finished evolving {len(mutated_instructions)} instructions!"
        )

        if self.generate_answers:
            self._logger.info(
                f"🧠 Generating answers for the {len(mutated_instructions)} evolved instructions!"
            )

            _mutated_instructions = [
                [{"role": "user", "content": instruction}]
                for instruction in mutated_instructions
            ]
            generated_answers = self.llm.generate(
                _mutated_instructions,
                **self.generation_kwargs,  # type: ignore
            )

            self._logger.info(
                f"🎉 Finished generating answers for the {len(mutated_instructions)} evolved instructions!"
            )

            yield (
                [
                    self.format_output(mutated_instruction, generated_answer)
                    for mutated_instruction, generated_answer in zip(
                        mutated_instructions, generated_answers
                    )
                ],
                True,
            )
