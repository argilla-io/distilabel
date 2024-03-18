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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field
from typing_extensions import override

from distilabel.llm.base import LLM
from distilabel.steps.base import GeneratorStep, RuntimeParameter, Step
from distilabel.utils.dicts import combine_dicts

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.task.typing import ChatType
    from distilabel.steps.typing import GeneratorStepOutput, StepOutput


class _Task(ABC):
    """_Task is an abstract class that implements the `Step` interface and adds the
    `format_input` and `format_output` methods to format the inputs and outputs of the
    task. It also adds a `llm` attribute to be used as the LLM to generate the outputs.

    Args:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
        generation_kwargs: The kwargs to be propagated to either `generate` or
            `agenerate` methods within each `LLM`. Note that these kwargs will be
            specific to each LLM, and while some as `temperature` may be present on each
            `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures
            in advance to see which kwargs are available.
    """

    llm: LLM

    group_generations: bool = False
    num_generations: RuntimeParameter[int] = Field(
        default=1, description="The number of generations to be produced per input."
    )
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
    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Asbtract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates an OpenAI chat-like list of dicts."""
        pass

    @abstractmethod
    def format_output(self, output: str) -> Dict[str, Any]:
        """Asbtract method to format the outputs of the task. It needs to receive an output
        as a string, and generates a Python dictionary with the outputs of the task."""
        pass

    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        formatted_inputs = self._format_inputs(inputs)
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,
            **self.generation_kwargs,  # type: ignore
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs)

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

    def _format_inputs(self, inputs: List[Dict[str, Any]]) -> List["ChatType"]:
        return [self.format_input(input) for input in inputs]

    def _format_outputs(self, outputs: List[Union[str, None]]) -> List[Dict[str, Any]]:
        """Formats the outputs of the task using the `format_output` method. If the output
        is `None` (i.e. the LLM failed to generate a response), then the outputs will be
        set to `None` as well.

        Args:
            outputs: The outputs of the LLM.

        Returns:
            A list containing a dictionary with the outputs of the task for each input.
        """
        return [
            (
                self.format_output(output)
                if output is not None
                else self._outputs_empty_dict()
            )
            for output in outputs
        ]

    def _outputs_empty_dict(self) -> Dict[str, None]:
        """Returns a dictionary with the outputs of the task set to `None`."""
        return {output: None for output in self.outputs}  # type: ignore


class Task(_Task, Step):
    """Task is a class that implements the `_Task` abstract class and adds the `Step`
    interface to be used as a step in the pipeline.

    Args:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
        generation_kwargs: The kwargs to be propagated to either `generate` or
            `agenerate` methods within each `LLM`. Note that these kwargs will be
            specific to each LLM, and while some as `temperature` may be present on each
            `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures
            in advance to see which kwargs are available.
    """

    pass


class GeneratorTask(_Task, GeneratorStep):
    """GeneratorTask is a class that implements the `_Task` abstract class and adds the
    `GeneratorStep` interface to be used as a step in the pipeline.

    Args:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
        generation_kwargs: The kwargs to be propagated to either `generate` or
            `agenerate` methods within each `LLM`. Note that these kwargs will be
            specific to each LLM, and while some as `temperature` may be present on each
            `LLM`, some others may not, so read the `LLM.{generate,agenerate}` signatures
            in advance to see which kwargs are available.
    """

    pass


class DataTask(_Task, GeneratorStep):
    """DataTask is a class that implements the `_Task` abstract class and adds the
    `GeneratorStep` interface to be used as a step in the pipeline.

    Args:
        data: The data to be used to generate the outputs of the task.
    """

    llm: LLM = None
    group_generations: Optional[bool] = None
    num_generations: Optional[int] = None
    generation_kwargs: Optional[dict] = None
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The data to be used to generate the outputs of the task.",
    )

    @override
    def load(self) -> None:
        pass

    @override
    def process(self) -> "GeneratorStepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """
        for entry in self.data:
            yield self.format_output(entry)

    @property
    def outputs(self) -> List[str]:
        """List of strings with the names of the columns that the step will produce as
        output.

        Returns:
            List of strings with the names of the columns that the step will produce as
            output.
        """
        return list(self.data[0].keys())

    def format_input(self, input: Dict[str, Any]) -> "ChatType":  # type: ignore
        pass

    def format_output(self, input: Dict[str, Any]) -> "ChatType":  # type: ignore
        pass
