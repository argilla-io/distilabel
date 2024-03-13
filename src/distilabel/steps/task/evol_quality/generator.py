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
    pass
else:
    pass

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from distilabel.steps.task.base import GeneratorTask
from distilabel.steps.task.evol_quality.utils import MutationTemplates
from distilabel.steps.task.typing import ChatType
from pydantic import Field, PrivateAttr
from typing_extensions import override

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput


class EvolQualityGenerator(GeneratorTask):
    """
    Papers
        - What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning.
        - WizardLM: Empowering Large Language Models to Follow Complex Instructions
    Reference:
        - https://arxiv.org/abs/2312.15685
        - https://arxiv.org/abs/2304.12244
        - https://github.com/h2oai/h2o-wizardlm
    """

    min_length: RuntimeParameter[int] = Field(default=256)
    max_length: RuntimeParameter[int] = Field(default=1024)

    mutation_type: Union[str, None] = None
    _seed_texts: Optional[List[str]] = PrivateAttr(default_factory=list)
    _prompts: Optional[List[str]] = PrivateAttr(default_factory=list)

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["instruction", "response"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:  # type: ignore
        pass

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `instruction`, the `answer` if `generate_answers=True`
        and the `model_name`."""
        # TODO: having to define a `model_name` column every time as the `Task.outputs` is not ideal,
        # this could be handled always and the value could be included within the DAG validation when
        # a `Task` is used, since all the `Task` subclasses will have an `llm` with a `model_name` attr.
        return ["rewritten_response", "model_name"]

    def format_output(self, rewritten_response: str) -> Dict[str, Any]:  # type: ignore
        """The output is formatted as a dictionary with the `instruction`. The `model_name`
        will be automatically included."""
        if not self.output_mappings:
            self.output_mappings = {k: k for k in self.outputs}

        return {
            self.output_mappings["rewritten_response"]: rewritten_response,
            self.output_mappings["model_name"]: self.llm.model_name,
        }

    @override
    def process(self, inputs: dict) -> "GeneratorStepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        formatted_prompts = []

        def eliminate(repsonse):
            return repsonse

        enum_attributes = [
            member.name for member in MutationTemplates.__members__.values()
        ]

        for input in inputs:
            if self.mutation_type is None:
                mutation_type = np.random.choice(enum_attributes)
                prompt_template = MutationTemplates[mutation_type].value
            else:
                prompt_template = MutationTemplates[self.mutation_type].value

            prompt_template = prompt_template.replace("<PROMPT>", input["instruction"])
            prompt_template = prompt_template.replace("<RESPONSE>", input["response"])
            formatted_prompt = [{"role": "user", "content": prompt_template}]
            formatted_prompts.append(formatted_prompt)

        generated_prompts = self.llm.generate(
            formatted_prompts,
            **self.generation_kwargs,  # type: ignore
        )

        # TODO: think about how to deal with elimination step

        for generated_prompt in generated_prompts:
            yield self.format_output(generated_prompt)
