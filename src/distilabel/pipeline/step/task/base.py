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
from distilabel.pipeline.step.base import Step
from distilabel.pipeline.step.task.typing import ChatType
from distilabel.pipeline.step.typing import RuntimeParameter, StepInput, StepOutput


class Task(Step, ABC):
    """Task is an abstract class that implements the `Step` interface and adds the
    `format_input` and `format_output` methods to format the inputs and outputs of the
    task. It also adds a `llm` attribute to be used as the LLM to generate the outputs.

    Args:
        llm: the `LLM` to be used to generate the outputs of the task.
        generation_kwargs: The kwargs to be propagated to either `generate` or
            `agenerate` methods within each `LLM`. Note that these kwargs will be
            specific to each LLM, and while some as `temperature` may be present on each
            `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures
            in advance to see which kwargs are available.
    """

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
        """Loads the LLM via the `LLM.load()` method (done for safer serialization)."""
        self.llm.load()  # type: ignore

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """Asbtract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates an OpenAI chat-like list of dicts."""
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        """Asbtract method to format the outputs of the task. It needs to receive an output
        as a string, and generates a Python dictionary with the outputs of the task."""
        pass

    def process(self, inputs: StepInput) -> StepOutput:
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
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
