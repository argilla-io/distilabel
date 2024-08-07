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
from typing_extensions import override

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.llms.base import LLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import (
    GeneratorStep,
    Step,
    StepInput,
    _Step,
)
from distilabel.utils.dicts import group_dicts

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput
    from distilabel.steps.typing import StepOutput


class _Task(_Step, ABC):
    """_Task is an abstract class that implements the `_Step` interface and adds the
    `format_input` and `format_output` methods to format the inputs and outputs of the
    task. It also adds a `llm` attribute to be used as the LLM to generate the outputs.

    Attributes:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        add_raw_output: whether to include a field with the raw output of the LLM in the
            `distilabel_metadata` field of the output. Can be helpful to not loose data
            with `Tasks` that need to format the output of the `LLM`. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    llm: LLM

    group_generations: bool = False
    add_raw_output: RuntimeParameter[bool] = Field(
        default=True,
        description=(
            "Whether to include the raw output of the LLM in the key `raw_output_<TASK_NAME>`"
            " of the `distilabel_metadata` dictionary output column"
        ),
    )
    num_generations: RuntimeParameter[int] = Field(
        default=1, description="The number of generations to be produced per input."
    )

    def load(self) -> None:
        """Loads the LLM via the `LLM.load()` method."""
        super().load()
        self.llm.load()

    @override
    def unload(self) -> None:
        """Unloads the LLM."""
        self._logger.debug("Executing task unload logic.")
        self.llm.unload()

    @abstractmethod
    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        """Abstract method to format the outputs of the task. It needs to receive an output
        as a string, and generates a Python dictionary with the outputs of the task. In
        addition the `input` used to generate the output is also received just in case it's
        needed to be able to parse the output correctly.
        """
        pass

    def _format_outputs(
        self,
        outputs: "GenerateOutput",
        input: Union[Dict[str, Any], None] = None,
    ) -> List[Dict[str, Any]]:
        """Formats the outputs of the task using the `format_output` method. If the output
        is `None` (i.e. the LLM failed to generate a response), then the outputs will be
        set to `None` as well.

        Args:
            outputs: The outputs (`n` generations) for the provided `input`.
            input: The input used to generate the output.

        Returns:
            A list containing a dictionary with the outputs of the task for each input.
        """
        inputs = [None] if input is None else [input]

        formatted_outputs = []
        for output, input in zip(outputs, inputs * len(outputs)):  # type: ignore
            try:
                formatted_output = self.format_output(output, input)
                formatted_output = self._maybe_add_raw_output(
                    formatted_output,
                    output,
                    add_raw_output=self.add_raw_output,  # type: ignore
                )
                formatted_outputs.append(formatted_output)
            except Exception as e:
                self._logger.warning(  # type: ignore
                    f"Task '{self.name}' failed to format output: {e}. Saving raw response."  # type: ignore
                )
                formatted_outputs.append(self._output_on_failure(output, input))
        return formatted_outputs

    def _output_on_failure(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """In case of failure to format the output, this method will return a dictionary including
        a new field `distilabel_meta` with the raw output of the LLM.
        """
        # Create a dictionary with the outputs of the task (every output set to None)
        outputs = {output: None for output in self.outputs}
        outputs["model_name"] = self.llm.model_name  # type: ignore
        outputs = self._maybe_add_raw_output(
            outputs,
            output,
            add_raw_output=self.add_raw_output,  # type: ignore
        )
        return outputs

    def _maybe_add_raw_output(
        self,
        output: Dict[str, Any],
        raw_output: Union[str, None],
        add_raw_output: bool = True,
    ) -> Dict[str, Any]:
        """Adds the raw output of the LLM to the output dictionary if `add_raw_output` is True."""
        if add_raw_output:
            meta = output.get(DISTILABEL_METADATA_KEY, {})
            meta[f"raw_output_{self.name}"] = raw_output
            output[DISTILABEL_METADATA_KEY] = meta
        return output


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
    def format_input(self, input: Dict[str, Any]) -> "FormattedInput":
        """Abstract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates an OpenAI chat-like list of dicts."""
        pass

    def _format_inputs(self, inputs: List[Dict[str, Any]]) -> List["FormattedInput"]:
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

        # `outputs` is a list containing a list of generations per input
        outputs = self.llm.generate(
            inputs=formatted_inputs,
            num_generations=self.num_generations,  # type: ignore
            **self.llm.get_generation_kwargs(),  # type: ignore
        )

        task_outputs = []
        for input, input_outputs in zip(inputs, outputs):
            formatted_outputs = self._format_outputs(input_outputs, input)

            if self.group_generations:
                combined = group_dicts(*formatted_outputs)
                task_outputs.append(
                    {**input, **combined, "model_name": self.llm.model_name}
                )
                continue

            # Create a row per generation
            for formatted_output in formatted_outputs:
                task_outputs.append(
                    {**input, **formatted_output, "model_name": self.llm.model_name}
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
