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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, PositiveInt

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import ChatType, FormattedInput
    from distilabel.steps.typing import StepOutput


class Magpie(Task):
    n_turns: Optional[RuntimeParameter[PositiveInt]] = Field(
        default=None,
        description="If provided, then the number of turns to generate for the conversation.",
    )

    @property
    def inputs(self) -> List[str]:
        return []

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return []

    @property
    def outputs(self) -> List[str]:
        if self.n_turns is None:
            return ["instruction"]

        return ["conversation"]

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        return {}

    def _prepare_inputs_for_instruction_generation(
        self, inputs: List[Dict[str, Any]]
    ) -> List["FormattedInput"]:
        return [
            [{"role": "system", "content": input["system_prompt"]}]
            if "system_prompt" in input
            else []
            for input in inputs
        ]

    def _format_instruction_generation_output(self, outputs: List["GenerateOutput"]):
        instructions = []
        for output in outputs:
            if output[0] is None:
                instructions.append({"instruction": None})
            else:
                parts = output[0].split("\n")
                instructions.append({"instruction": parts[0]})
        return instructions

    def process(self, inputs: StepInput) -> "StepOutput":
        inputs_for_instruction_generation = (
            self._prepare_inputs_for_instruction_generation(inputs=inputs)
        )
        outputs = self.llm.generate(
            inputs=inputs_for_instruction_generation, num_generations=1
        )
        instructions = self._format_instruction_generation_output(outputs=outputs)

        if self.n_turns is None:
            yield instructions
