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

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.errors import DistilabelUserError
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.image_generation.base import ImageGenerationModel
from distilabel.models.llms.base import LLM
from distilabel.steps.base import (
    GeneratorStep,
    GlobalStep,
    Step,
    StepInput,
    _Step,
)
from distilabel.utils.dicts import group_dicts

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput, LLMStatistics
    from distilabel.steps.tasks.typing import ChatType, FormattedInput
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
    add_raw_input: RuntimeParameter[bool] = Field(
        default=True,
        description=(
            "Whether to include the raw input of the LLM in the key `raw_input_<TASK_NAME>`"
            " of the `distilabel_metadata` dictionary column"
        ),
    )
    num_generations: RuntimeParameter[int] = Field(
        default=1, description="The number of generations to be produced per input."
    )
    use_default_structured_output: bool = False

    _can_be_used_with_offline_batch_generation: bool = PrivateAttr(False)

    def model_post_init(self, __context: Any) -> None:
        if (
            self.llm.use_offline_batch_generation
            and not self._can_be_used_with_offline_batch_generation
        ):
            raise DistilabelUserError(
                f"`{self.__class__.__name__}` task cannot be used with offline batch generation"
                " feature.",
                page="sections/how_to_guides/advanced/offline-batch-generation",
            )

        super().model_post_init(__context)

    @property
    def is_global(self) -> bool:
        """Extends the `is_global` property to return `True` if the task is using the
        offline batch generation feature, otherwise it returns the value of the parent
        class property. `offline_batch_generation` requires to receive all the inputs
        at once, so for the `_BatchManager` this is a global step.

        Returns:
            Whether the task is a global step or not.
        """
        if self.llm.use_offline_batch_generation:
            return True

        return super().is_global

    def load(self) -> None:
        """Loads the LLM via the `LLM.load()` method."""
        super().load()
        self._set_default_structured_output()
        self.llm.load()

    @override
    def unload(self) -> None:
        """Unloads the LLM."""
        self._logger.debug("Executing task unload logic.")
        self.llm.unload()

    @override
    def impute_step_outputs(
        self, step_output: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Imputes the outputs of the task in case the LLM failed to generate a response.
        """
        result = []
        for row in step_output:
            data = row.copy()
            for output in self.get_outputs().keys():
                data[output] = None
            data = self._create_metadata(
                data,
                None,
                None,
                add_raw_output=self.add_raw_output,
                add_raw_input=self.add_raw_input,
            )
            result.append(data)
        return result

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
        repeate_inputs = len(outputs.get("generations"))
        outputs = normalize_statistics(outputs)

        for (output, stats), input in zip(
            iterate_generations_with_stats(outputs), inputs * repeate_inputs
        ):  # type: ignore
            try:
                # Extract the generations, and move the statistics to the distilabel_metadata,
                # to keep everything clean
                formatted_output = self.format_output(output, input)
                formatted_output = self._create_metadata(
                    formatted_output,
                    output,
                    input,
                    add_raw_output=self.add_raw_output,  # type: ignore
                    add_raw_input=self.add_raw_input,  # type: ignore
                    statistics=stats,
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
        outputs = self._create_metadata(
            outputs,
            output,
            input,
            add_raw_output=self.add_raw_output,  # type: ignore
            add_raw_input=self.add_raw_input,  # type: ignore
        )
        return outputs

    def _create_metadata(
        self,
        output: Dict[str, Any],
        raw_output: List[Union[str, None]],
        input: Union[str, None],
        add_raw_output: bool = True,
        add_raw_input: bool = True,
        statistics: Optional["LLMStatistics"] = None,
    ) -> Dict[str, Any]:
        """Adds the raw output and or the formatted input of the LLM to the output dictionary
        if `add_raw_output` is True or `add_raw_input` is True.

        Args:
            output:
                The output dictionary after formatting the output from the LLM,
                to add the raw output and or raw input.
            raw_output: The raw output of the LLM (the list of generations).
            input: The raw input of the LLM.
            add_raw_output: Whether to add the raw output to the output dictionary.
            add_raw_input: Whether to add the raw input to the output dictionary.
            statistics: The statistics generated by the LLM, which should contain at least
                the number of input and output tokens.
        """
        meta = output.get(DISTILABEL_METADATA_KEY, {})

        if add_raw_output:
            meta[f"raw_output_{self.name}"] = raw_output
        if add_raw_input:
            meta[f"raw_input_{self.name}"] = self.format_input(input) if input else None
        if statistics:
            meta[f"statistics_{self.name}"] = statistics
        if meta:
            output[DISTILABEL_METADATA_KEY] = meta

        return output

    def _set_default_structured_output(self) -> None:
        """Prepares the structured output to be set in the selected `LLM`.

        If the method `get_structured_output` returns None (the default), there's no need
        to set anything, as it doesn't apply.
        If the `use_default_structured_output` and there's no previous structured output
        set by hand, then decide the type of structured output to select depending on the
        `LLM` provider.
        """
        schema = self.get_structured_output()
        if not schema:
            return

        if self.use_default_structured_output and not self.llm.structured_output:
            # In case the default structured output is required, we have to set it before
            # the LLM is loaded
            from distilabel.models.llms import InferenceEndpointsLLM
            from distilabel.models.llms.base import AsyncLLM

            def check_dependency(module_name: str) -> None:
                if not importlib.util.find_spec(module_name):
                    raise ImportError(
                        f"`{module_name}` is not installed and is needed for the structured generation with this LLM."
                        f" Please install it using `pip install {module_name}`."
                    )

            dependency = "outlines"
            structured_output = {"schema": schema}
            if isinstance(self.llm, InferenceEndpointsLLM):
                structured_output.update({"format": "json"})
            # To determine instructor or outlines format
            elif isinstance(self.llm, AsyncLLM) and not isinstance(
                self.llm, InferenceEndpointsLLM
            ):
                dependency = "instructor"
                structured_output.update({"format": "json"})

            check_dependency(dependency)
            self.llm.structured_output = structured_output

    def get_structured_output(self) -> Union[Dict[str, Any], None]:
        """Returns the structured output for a task that implements one by default,
        must be overriden by subclasses of `Task`. When implemented, should be a json
        schema that enforces the response from the LLM so that it's easier to parse.
        """
        return None

    def _sample_input(self) -> "ChatType":
        """Returns a sample input to be used in the `print` method.
        Tasks that don't adhere to a format input that returns a map of the type
        str -> str should override this method to return a sample input.
        """
        return self.format_input(
            {input: f"<PLACEHOLDER_{input.upper()}>" for input in self.inputs}
        )

    def print(self, sample_input: Optional["ChatType"] = None) -> None:
        """Prints a sample input to the console using the `rich` library.
        Helper method to visualize the prompt of the task.

        Args:
            sample_input: A sample input to be printed. If not provided, a default will be
                generated using the `_sample_input` method, which can be overriden by
                subclasses. This should correspond to the same example you could pass to
                the `format_input` method.
                The variables be named <PLACEHOLDER_VARIABLE_NAME> by default.

        Examples:
            Print the URIAL prompt:

            ```python
            from distilabel.steps.tasks import URIAL
            from distilabel.models.llms.huggingface import InferenceEndpointsLLM

            # Consider this as a placeholder for your actual LLM.
            urial = URIAL(
                llm=InferenceEndpointsLLM(
                    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                ),
            )
            urial.load()
            urial.print()
            ╭─────────────────────────────────────── Prompt: URIAL  ────────────────────────────────────────╮
            │ ╭────────────────────────────────────── User Message ───────────────────────────────────────╮ │
            │ │ # Instruction                                                                             │ │
            │ │                                                                                           │ │
            │ │ Below is a list of conversations between a human and an AI assistant (you).               │ │
            │ │ Users place their queries under "# User:", and your responses are under  "# Assistant:".  │ │
            │ │ You are a helpful, respectful, and honest assistant.                                      │ │
            │ │ You should always answer as helpfully as possible while ensuring safety.                  │ │
            │ │ Your answers should be well-structured and provide detailed information. They should also │ │
            │ │ have an engaging tone.                                                                    │ │
            │ │ Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic,      │ │
            │ │ dangerous, or illegal content, even if it may be helpful.                                 │ │
            │ │ Your response must be socially responsible, and thus you can refuse to answer some        │ │
            │ │ controversial topics.                                                                     │ │
            │ │                                                                                           │ │
            │ │                                                                                           │ │
            │ │ # User:                                                                                   │ │
            │ │                                                                                           │ │
            │ │ <PLACEHOLDER_INSTRUCTION>                                                                 │ │
            │ │                                                                                           │ │
            │ │ # Assistant:                                                                              │ │
            │ ╰───────────────────────────────────────────────────────────────────────────────────────────╯ │
            ╰───────────────────────────────────────────────────────────────────────────────────────────────╯
            ```
        """
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        sample_input = sample_input or self._sample_input()

        panels = []
        for item in sample_input:
            content = Text.assemble((item.get("content", ""),))
            panel = Panel(
                content,
                title=f"[bold][magenta]{item.get('role', '').capitalize()} Message[/magenta][/bold]",
                border_style="light_cyan3",
            )
            panels.append(panel)

        # Create a group of panels
        # Wrap the group in an outer panel
        outer_panel = Panel(
            Group(*panels),
            title=f"[bold][magenta]Prompt: {type(self).__name__} [/magenta][/bold]",
            border_style="light_cyan3",
            expand=False,
        )
        console.print(outer_panel)


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

        # `outputs` is a dict containing the LLM outputs in the `generations`
        # key and the statistics in the `statistics` key
        outputs = self.llm.generate_outputs(
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
    """`GeneratorTask` is a class that implements the `_Task` abstract class and adds the
    `GeneratorStep` interface to be used as a step in the pipeline.

    Attributes:
        llm: the `LLM` to be used to generate the outputs of the task.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    pass


class GlobalTask(_Task, GlobalStep):
    """`GlobalTask` is a class that implements the `_Task` abstract class and adds the
    `GlobalStep` interface to be used as a step in the pipeline. It's generally used in
    combination with `LLM`s that can be used for offline batched inference.
    """

    pass


class ImageTask(_Task, Step):
    """`ImageTask` is a class that implements the `_Task` abstract class and adds the `Step`
    interface to be used as a step in the pipeline. It differs from the `Task` in that it's
    expected to work with `ImageGenerationModel`s instead of `LLM`s.

    Attributes:
        image_generation_model: the `ImageGenerationModel` to be used to generate the outputs.
        llm: This attribute is here to respect the `_Task` interface, but it's used internally only.
        group_generations: whether to group the `num_generations` generated per input in
            a list or create a row per generation. Defaults to `False`.
        num_generations: The number of generations to be produced per input.
    """

    llm: Union[LLM, ImageGenerationModel, None] = None
    image_generation_model: ImageGenerationModel

    def model_post_init(self, __context: Any) -> None:
        assert self.llm is None, (
            "`ImageTask` cannot use an `LLM` attribute given by the user, pass "
            "the `image_generation_model` attribute instead."
        )
        self.llm = self.image_generation_model
        # Call the post init from the Step, as we don't want to call specific behaviour
        # from the task, that may need to deal with specific attributes from the LLM
        # not in the ImageGenerationModel
        super(Step, self).model_post_init(__context)

    @abstractmethod
    def format_input(self, input: dict[str, any]) -> str:
        """Abstract method to format the inputs of the task. It needs to receive an input
        as a Python dictionary, and generates a string to be used as the prompt for the model."""
        pass

    def _format_inputs(self, inputs: list[dict[str, any]]) -> List["FormattedInput"]:
        """Formats the inputs of the task using the `format_input` method.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list containing the formatted inputs, which are `ChatType`-like following
            the OpenAI formatting.
        """
        return [self.format_input(input) for input in inputs]

    def _format_outputs(
        self,
        outputs: list[Union[str, None]],
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

        for output, input in zip(outputs, inputs):  # type: ignore
            try:
                formatted_output = self.format_output(output, input)
                formatted_output = self._create_metadata(
                    formatted_output,
                    output,
                    input,
                    add_raw_output=self.add_raw_output,  # type: ignore
                    add_raw_input=self.add_raw_input,  # type: ignore
                    statistics=None,
                )
                formatted_outputs.append(formatted_output)
            except Exception as e:
                self._logger.warning(  # type: ignore
                    f"Task '{self.name}' failed to format output: {e}. Saving raw response."  # type: ignore
                )
                formatted_outputs.append(self._output_on_failure(output, input))
        return formatted_outputs

    @abstractmethod
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the `ImageGenerationModel`.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            A list of Python dictionaries with the outputs of the task.
        """
        pass


def normalize_statistics(output: "GenerateOutput") -> "GenerateOutput":
    """Transforms the GenerateOutput statistics to have the same length as the generations.

    Args:
        data: A generate output that possibly has different lengths of statistics
            vs generations (due to num_generations=3 returning 3 generations, but
            for example the tokens are only counted once).

    Returns:
        Normalized statistics according to the generations length.

    Examples:
        ```python
        data = {
            "generations": ["text1", "text2", "text3", "text4"],
            "statistics": {"input_tokens": [1], "output_tokens": [1, 2, 3]}
        }
        normalize_statistics(data)
        data = {
            "generations": ["text1", "text2", "text3"],
            "statistics": {"input_tokens": [1, 1, 1], "output_tokens": [1, 2, 3]}
        }
        ```
    """
    statistics = output.get("statistics")
    if not statistics:
        return output
    gen_length = len(output["generations"])

    for stat_key, stat_values in output["statistics"].items():
        current_length = len(stat_values)

        if current_length < gen_length:
            # Calculate how many times to repeat the tokens
            repeats = gen_length // current_length
            remainder = gen_length % current_length

            # Create new list with repeated values
            new_values = stat_values * repeats + stat_values[:remainder]
            output["statistics"][stat_key] = new_values

    return output


def iterate_generations_with_stats(output: "GenerateOutput") -> "GenerateOutput":
    """Helper function to iterate together generations and statistics while
    processing them inside _format_outputs.

    Args:
        output: Output from the LLM.generate_outputs method.

    Yields:
        Iterator of generation and statistics paired.
    """
    for i, generation in enumerate(output["generations"]):
        # Create a new dictionary with the statistics for this index
        stats = {key: values[i] for key, values in output["statistics"].items()}

        yield generation, stats
