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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import Field

from distilabel.llm.base import LLM
from distilabel.steps.base import RuntimeParameter, Step
from distilabel.steps.task.typing import ChatType
from distilabel.steps.typing import StepInput, StepOutput


class Task(Step, ABC):
    llm: LLM

    generation_kwargs: Optional[RuntimeParameter[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="The kwargs to be propagated to either `generate` or `agenerate`"
        " methods within each `LLM`. Note that these kwargs will be specific to each"
        " LLM, and while some as `temperature` may be present on each `LLM`, some others"
        " may not, so read the `LLM.{generate,agenerate}` signatures in advance to see"
        " which kwargs are available.",
    )

    def load(self) -> None:
        self.llm.load()  # type: ignore

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        pass

    def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
        formatted_inputs = [self.format_input(input) for input in inputs]
        outputs = self.llm.generate(formatted_inputs, **self.generation_kwargs)  # type: ignore
        formatted_outputs = [self.format_output(output) for output in outputs]  # type: ignore

        outputs: StepOutput = []  # type: ignore
        for input, formatted_output in zip(inputs, formatted_outputs):
            output = {k: v for k, v in input.items() if k in self.inputs}
            output.update(formatted_output)
            output["model_name"] = self.llm.model_name  # type: ignore
            outputs.append(output)  # type: ignore
        yield outputs  # type: ignore
