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
from typing import Any, Dict

from distilabel.llm.base import LLM
from distilabel.pipeline.step.base import Step
from distilabel.pipeline.step.task.typing import ChatType
from distilabel.pipeline.step.typing import StepInput, StepOutput


class Task(Step, ABC):
    llm: LLM

    def load(self) -> None:
        self.llm.load()  # type: ignore

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        pass

    def process(self, inputs: StepInput) -> StepOutput:
        formatted_inputs = [self.format_input(input) for input in inputs]
        outputs = self.llm.generate(formatted_inputs)  # type: ignore
        formatted_outputs = [self.format_output(output) for output in outputs]  # type: ignore

        outputs: StepOutput = []  # type: ignore
        for input, formatted_output in zip(inputs, formatted_outputs):
            output = {k: v for k, v in input.items() if k in self.inputs}
            output.update(formatted_output)
            output["model_name"] = self.llm.model_name  # type: ignore
            outputs.append(output)  # type: ignore
        yield outputs  # type: ignore
