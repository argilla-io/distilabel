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
from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import Field

from distilabel.llms.base import LLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import (
    GeneratorStep,
    Step,
    StepInput,
    _Step,
)
from distilabel.utils.dicts import combine_dicts

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput


class _Task(_Step, ABC):
    """_Task is an abstract class that implements the `_Step` interface and adds the
    `format_input` and `format_output` methods to format the inputs and outputs of the
    task. It also adds a `llm` attribute to be used as the LLM to generate the outputs.

    Args:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    llm: LLM

    group_generations: bool = False
    num_generations: RuntimeParameter[int] = Field(
        default=1, description="The number of generations to be produced per input."
    )

    def load(self) -> None:
        """Loads the LLM via the `LLM.load()` method (done for safer serialization)."""
        super().load()
        self.llm.load()

    @abstractmethod
    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Abstract method to format the outputs of the task. It needs to receive an output
        as a string, and generates a Python dictionary with the outputs of the task. In
        addition the `input` used to generate the output is also received just in case it's
        needed to be able to parse the output correctly.
        """
        pass

    def _format_outputs(
        self, outputs: "GenerateOutput", inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Formats the outputs of the task using the `format_output` method. If the output
        is `None` (i.e. the LLM failed to generate a response), then the outputs will be
        set to `None` as well.

        Args:
            outputs: The outputs of the LLM.
            inputs: The inputs used to generate the outputs.

        Returns:
            A list containing a dictionary with the outputs of the task for each input.
        """
        formatted_outputs = []
        for output, input in zip(outputs, inputs * len(outputs)):
            try:
                formatted_outputs.append(self.format_output(output, input))
            except Exception as e:
                self._logger.warning(  # type: ignore
                    f"Task '{self.name}' failed to format output: {e}. Using empty dict."  # type: ignore
                )
                formatted_outputs.append(self._outputs_empty_dict())
        return formatted_outputs

    def _outputs_empty_dict(self) -> Dict[str, None]:
        """Returns a dictionary with the outputs of the task set to `None`."""
        return {output: None for output in self.outputs}  # type: ignore


class Task(_Task, Step):
    """Task is a class that implements the `_Task` abstract class and adds the `Step`
    interface to be used as a step in the pipeline.

    Attributes:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    @abstractmethod
    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Abstract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates an OpenAI chat-like list of dicts."""
        pass

    def _format_inputs(self, inputs: List[Dict[str, Any]]) -> List["ChatType"]:
        """Formats the inputs of the task using the `format_input` method.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list containing the formatted inputs, which are `ChatType`-like following
            the OpenAI formatting.
        """
        return [self.format_input(input) for input in inputs]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            A list of Python dictionaries with the outputs of the task.
        """

        formatted_inputs = self._format_inputs(inputs)
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,  # type: ignore
            **self.llm.generation_kwargs,  # type: ignore
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs, inputs)

            if self.group_generations:
                combined = combine_dicts(*formatted_outputs)
                task_outputs.append(
                    {**input, "model_name": self.llm.model_name, **combined}
                )
                continue

            # Create a row per generation
            for formatted_output in formatted_outputs:
                task_outputs.append(
                    {**input, "model_name": self.llm.model_name, **formatted_output}
                )

        yield task_outputs


class GeneratorTask(_Task, GeneratorStep):
    """GeneratorTask is a class that implements the `_Task` abstract class and adds the
    `GeneratorStep` interface to be used as a step in the pipeline.

    Attributes:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    pass
