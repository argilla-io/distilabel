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

import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr
from typing_extensions import Annotated

from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)
from distilabel.utils.serialization import TYPE_INFO_KEY, _Serializable
from distilabel.utils.typing_ import is_parameter_annotated_with

if TYPE_CHECKING:
    from logging import Logger

    from distilabel.steps.typing import GeneratorStepOutput, StepOutput

DEFAULT_INPUT_BATCH_SIZE = 50


_STEP_INPUT_ANNOTATION = "distilabel_step_input"
StepInput = Annotated[List[Dict[str, Any]], _STEP_INPUT_ANNOTATION]
"""StepInput is just an `Annotated` alias of the typing `List[Dict[str, Any]]` with
extra metadata that allows `distilabel` to perform validations over the `process` step
method defined in each `Step`"""


class _Step(RuntimeParametersMixin, BaseModel, _Serializable, ABC):
    """Base class for the steps that can be included in a `Pipeline`.

    A `Step` is a class defining some processing logic. The input and outputs for this
    processing logic are lists of dictionaries with the same keys:

        ```python
        [
            {"column1": "value1", "column2": "value2", ...},
            {"column1": "value1", "column2": "value2", ...},
            {"column1": "value1", "column2": "value2", ...},
        ]
        ```

    The processing logic is defined in the `process` method, which depending on the
    number of previous steps, can receive more than one list of dictionaries, each with
    the output of the previous steps. In order to make `distilabel` know where the outputs
    from the previous steps are, the `process` function from each `Step` must have an argument
    or positional argument annotated with `StepInput`.

        ```python
        class StepWithOnePreviousStep(Step):
            def process(self, inputs: StepInput) -> StepOutput:
                yield [...]

        class StepWithSeveralPreviousStep(Step):
            # mind the * to indicate that the argument is a list of StepInput
            def process(self, inputs: *StepInput) -> StepOutput:
                yield [...]
        ```

    In order to perform static validations and to check that the chaining of the steps
    in the pipeline is valid, a `Step` must also define the `inputs` and `outputs`
    properties:

    - `inputs`: a list of strings with the names of the columns that the step needs as
        input. It can be an empty list if the step is a generator step.
    - `outputs`: a list of strings with the names of the columns that the step will
        produce as output.

    Optionally, a `Step` can override the `load` method to perform any initialization
    logic before the `process` method is called. For example, to load an LLM, stablish a
    connection to a database, etc.

    Finally, the `Step` class inherits from `pydantic.BaseModel`, so attributes can be easily
    defined, validated, serialized and included in the `__init__` method of the step.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_default=True, validate_assignment=True
    )

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    pipeline: Annotated[Any, Field(exclude=True, repr=False)] = None
    input_mappings: Dict[str, str] = {}
    output_mappings: Dict[str, str] = {}

    _built_from_decorator: bool = PrivateAttr(default=False)
    _logger: Union["Logger", None] = PrivateAttr(...)

    def model_post_init(self, __context: Any) -> None:
        from distilabel.pipeline.base import _GlobalPipelineManager

        super().model_post_init(__context)

        if self.pipeline is None:
            self.pipeline = _GlobalPipelineManager.get_pipeline()

        if self.pipeline is None:
            raise ValueError(
                f"Step '{self.name}' hasn't received a pipeline, and it hasn't been"
                " created within a `Pipeline` context. Please, use"
                " `with Pipeline() as pipeline:` and create the step within the context."
            )

        self.pipeline._add_step(self)

    def connect(
        self, step: "_Step", input_mappings: Union[Dict[str, str], None] = None
    ) -> None:
        """Connects the current step to another step in the pipeline, which means that
        the output of this step will be the input of the other step.

        Args:
            step: The step to connect to.
            input_mappings: A dictionary with the mapping of the columns from the output
                of the current step to the input of the other step. If `None`, the
                columns will be mapped by name. This is useful when the names of the
                output columns of the current step are different from the names of the
                input columns of the other step. Defaults to `None`.
        """
        if input_mappings is not None:
            step.input_mappings = input_mappings
        self.pipeline._add_edge(self.name, step.name)  # type: ignore

    def load(self) -> None:
        """Method to perform any initialization logic before the `process` method is
        called. For example, to load an LLM, stablish a connection to a database, etc.
        """
        self._logger = logging.getLogger(f"distilabel.step.{self.name}")

    @property
    def is_generator(self) -> bool:
        """Whether the step is a generator step or not.

        Returns:
            `True` if the step is a generator step, `False` otherwise.
        """
        return isinstance(self, GeneratorStep)

    @property
    def is_global(self) -> bool:
        """Whether the step is a global step or not.

        Returns:
            `True` if the step is a global step, `False` otherwise.
        """
        return isinstance(self, GlobalStep)

    @property
    def is_normal(self) -> bool:
        """Whether the step is a normal step or not.

        Returns:
            `True` if the step is a normal step, `False` otherwise.
        """
        return not self.is_generator and not self.is_global

    @property
    def inputs(self) -> List[str]:
        """List of strings with the names of the columns that the step needs as input.

        Returns:
            List of strings with the names of the columns that the step needs as input.
        """
        return []

    @property
    def outputs(self) -> List[str]:
        """List of strings with the names of the columns that the step will produce as
        output.

        Returns:
            List of strings with the names of the columns that the step will produce as
            output.
        """
        return []

    @cached_property
    def process_parameters(self) -> List[inspect.Parameter]:
        """Returns the parameters of the `process` method of the step.

        Returns:
            The parameters of the `process` method of the step.
        """
        return list(inspect.signature(self.process).parameters.values())  # type: ignore

    def has_multiple_inputs(self) -> bool:
        """Whether the `process` method of the step receives more than one input or not
        i.e. has a `*` argument annotated with `StepInput`.

        Returns:
            `True` if the `process` method of the step receives more than one input,
            `False` otherwise.
        """
        return any(
            param.kind == param.VAR_POSITIONAL for param in self.process_parameters
        )

    def get_process_step_input(self) -> Union[inspect.Parameter, None]:
        """Returns the parameter of the `process` method of the step annotated with
        `StepInput`.

        Returns:
            The parameter of the `process` method of the step annotated with `StepInput`,
            or `None` if there is no parameter annotated with `StepInput`.

        Raises:
            TypeError: If the step has more than one parameter annotated with `StepInput`.
        """
        step_input_parameter = None
        for parameter in self.process_parameters:
            if is_parameter_annotated_with(parameter, _STEP_INPUT_ANNOTATION):
                if step_input_parameter is not None:
                    raise TypeError(
                        f"Step '{self.name}' should have only one parameter with type"
                        " hint `StepInput`."
                    )
                step_input_parameter = parameter
        return step_input_parameter

    def verify_inputs_mappings(self) -> None:
        """Verifies that the `inputs_mappings` of the step are valid i.e. the input
        columns exist in the inputs of the step.

        Raises:
            ValueError: If the `inputs_mappings` of the step are not valid.
        """
        if not self.input_mappings:
            return

        for input in self.input_mappings:
            if input not in self.inputs:
                raise ValueError(
                    f"The input column '{input}' doesn't exist in the inputs of the"
                    f" step '{self.name}'. Inputs of the step are: {self.inputs}."
                    " Please, review the `inputs_mappings` argument of the step."
                )

    def verify_outputs_mappings(self) -> None:
        """Verifies that the `outputs_mappings` of the step are valid i.e. the output
        columns exist in the outputs of the step.

        Raises:
            ValueError: If the `outputs_mappings` of the step are not valid.
        """
        if not self.output_mappings:
            return

        for output in self.output_mappings:
            if output not in self.outputs:
                raise ValueError(
                    f"The output column '{output}' doesn't exist in the outputs of the"
                    f" step '{self.name}'. Outputs of the step are: {self.outputs}."
                    " Please, review the `outputs_mappings` argument of the step."
                )

    def get_inputs(self) -> List[str]:
        """Gets the inputs of the step after the `input_mappings`. This method is meant
        to be used to run validations on the inputs of the step.

        Returns:
            The inputs of the step after the `input_mappings`.
        """
        return [self.input_mappings.get(input, input) for input in self.inputs]

    def get_outputs(self) -> List[str]:
        """Gets the outputs of the step after the `outputs_mappings`. This method is
        meant to be used to run validations on the outputs of the step.

        Returns:
            The outputs of the step after the `outputs_mappings`.
        """
        return [self.output_mappings.get(output, output) for output in self.outputs]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_Step":
        """Create a Step from a dict containing the serialized data.

        Needs the information from the step and the Pipeline it belongs to.

        Note:
            It's intended for internal use.

        Args:
            data: dictionary containing the serialized data from a `Step` and the
                `Pipeline` it belongs to.

        Returns:
            A `Step` instance.
        """
        # Remove the "type_info" to avoid errors on instantiation
        _data = data.copy()
        if TYPE_INFO_KEY in _data.keys():
            _data.pop(TYPE_INFO_KEY)

        # Before passing the data to instantiate the general step, we have to instantiate
        # some of the internal objects. For the moment we only take into account the LLM,
        # we should take care if we update any of the objects.
        if llm := _data.get("llm"):
            from distilabel.utils.serialization import _get_class

            nested_cls = _get_class(**llm.pop(TYPE_INFO_KEY))
            # Load the LLM and update the _data inplace
            nested_cls = nested_cls(**llm)
            _data.update({"llm": nested_cls})

        # Enums need a specific restoring process
        for k, v in _data.items():
            if isinstance(v, dict) and "_type" in v and v["_type"] == "enum":
                _data[k] = Enum(v["_name"], v["_values"], type=eval(v["_enum_type"]))

        # Every step needs the pipeline, and the remaining arguments are general
        step = cls(**_data)

        return step

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        dump = super()._model_dump(obj, **kwargs)
        dump["runtime_parameters_info"] = self.get_runtime_parameters_info()
        return dump


class Step(_Step, ABC):
    """Base class for the steps that can be included in a `Pipeline`.

    Attributes:
        input_batch_size: The number of rows that will contain the batches processed by
            the step. Defaults to `50`.

    Runtime parameters:
        - `input_batch_size`: The number of rows that will contain the batches processed
            by the step. Defaults to `50`.
    """

    input_batch_size: RuntimeParameter[PositiveInt] = Field(
        default=DEFAULT_INPUT_BATCH_SIZE,
        description="The number of rows that will contain the batches processed by the"
        " step.",
    )

    @abstractmethod
    def process(self, *inputs: StepInput) -> "StepOutput":
        """Method that defines the processing logic of the step. It should yield the
        output rows.

        Args:
            *inputs: An argument used to receive the outputs of the previous steps. The
                number of arguments depends on the number of previous steps. It doesn't
                need to be an `*args` argument, it can be a regular argument annotated
                with `StepInput` if the step has only one previous step.
        """
        pass

    def process_applying_mappings(self, *args: List[Dict[str, Any]]) -> "StepOutput":
        """Runs the `process` method of the step applying the `input_mappings` to the input
        rows and the `outputs_mappings` to the output rows. This is the function that
        should be used to run the processing logic of the step.

        Yields:
            The output rows.
        """

        inputs = self._apply_input_mappings(args) if self.input_mappings else args

        # If the `Step` was built using the `@step` decorator, then we need to pass
        # the runtime parameters as kwargs, so they can be used within the processing
        # function
        generator = (
            self.process(*inputs)
            if not self._built_from_decorator
            else self.process(*inputs, **self._runtime_parameters)
        )

        for output_rows in generator:
            yield [
                {
                    # Apply output mapping and revert input mapping
                    self.output_mappings.get(k, None)
                    or self.input_mappings.get(k, None)
                    or k: v
                    for k, v in row.items()
                }
                for row in output_rows
            ]

    def _revert_input_mappings(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Reverts the `input_mappings` of the step to the input row.

        Args:
            input: The input row.

        Returns:
            The input row with the `input_mappings` reverted.
        """
        return {self.input_mappings.get(k, k): v for k, v in input.items()}

    def _apply_input_mappings(
        self, inputs: Tuple[List[Dict[str, Any]], ...]
    ) -> List[List[Dict[str, Any]]]:
        """Applies the `input_mappings` to the input rows.

        Args:
            inputs: The input rows.

        Returns:
            The input rows with the `input_mappings` applied.
        """
        reverted_input_mappings = {v: k for k, v in self.input_mappings.items()}

        return [
            [
                {reverted_input_mappings.get(k, k): v for k, v in row.items()}
                for row in row_inputs
            ]
            for row_inputs in inputs
        ]


class GeneratorStep(_Step, ABC):
    """A special kind of `Step` that is able to generate data i.e. it doesn't receive
    any input from the previous steps.

    Attributes:
        batch_size: The number of rows that will contain the batches generated by the
            step. Defaults to `50`.

    Runtime parameters:
        - `batch_size`: The number of rows that will contain the batches generated by
            the step. Defaults to `50`.
    """

    batch_size: RuntimeParameter[int] = Field(
        default=50,
        description="The number of rows that will contain the batches generated by the"
        " step.",
    )

    @abstractmethod
    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        """Method that defines the generation logic of the step. It should yield the
        output rows and a boolean indicating if it's the last batch or not.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Yields:
            The output rows and a boolean indicating if it's the last batch or not.
        """
        pass

    def process_applying_mappings(self, offset: int = 0) -> "GeneratorStepOutput":
        """Runs the `process` method of the step applying the `outputs_mappings` to the
        output rows. This is the function that should be used to run the generation logic
        of the step.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Yields:
            The output rows and a boolean indicating if it's the last batch or not.
        """

        # If the `Step` was built using the `@step` decorator, then we need to pass
        # the runtime parameters as `kwargs`, so they can be used within the processing
        # function
        generator = (
            self.process(offset=offset)
            if not self._built_from_decorator
            else self.process(offset=offset, **self._runtime_parameters)
        )

        for output_rows, last_batch in generator:
            yield (
                [
                    {self.output_mappings.get(k, k): v for k, v in row.items()}
                    for row in output_rows
                ],
                last_batch,
            )


class GlobalStep(Step, ABC):
    """A special kind of `Step` which it's `process` method receives all the data processed
    by their previous steps at once, instead of receiving it in batches. This kind of steps
    are useful when the processing logic requires to have all the data at once, for example
    to train a model, to perform a global aggregation, etc.
    """

    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return []
