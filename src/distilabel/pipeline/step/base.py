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
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr
from typing_extensions import Annotated, get_args, get_origin

from distilabel.pipeline.base import BasePipeline, _GlobalPipelineManager
from distilabel.pipeline.logging import get_logger
from distilabel.pipeline.serialization import _Serializable
from distilabel.pipeline.step.typing import StepInput

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from distilabel.pipeline.step.typing import GeneratorStepOutput, StepOutput

DEFAULT_INPUT_BATCH_SIZE = 50

_T = TypeVar("_T")


class _DefaultValueRuntimeParameter:
    pass


_DEFAULT_VALUE_RUNTIME_PARAMETER = _DefaultValueRuntimeParameter()
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"

RuntimeParameter = Annotated[
    _T,
    Field(default=_DEFAULT_VALUE_RUNTIME_PARAMETER),
    _RUNTIME_PARAMETER_ANNOTATION,
]
"""Used to mark the attributes of a `Step` as a runtime parameter."""


class _Step(BaseModel, _Serializable, ABC):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    pipeline: Annotated[
        Union[BasePipeline, None], Field(exclude=True, repr=False)
    ] = None
    input_mappings: Dict[str, str] = {}
    output_mappings: Dict[str, str] = {}

    _runtime_parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _values: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger = PrivateAttr(get_logger("step"))

    def model_post_init(self, _: Any) -> None:
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
        pass

    def _set_runtime_parameters(self, runtime_parameters: Dict[str, Any]) -> None:
        """Sets the runtime parameters of the step.

        Args:
            runtime_parameters: A dictionary with the runtime parameters for the step.
        """
        self._runtime_parameters = runtime_parameters
        for name, value in runtime_parameters.items():
            setattr(self, name, value)

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

    @property
    def runtime_parameters_names(self) -> Dict[str, bool]:
        """Returns a dictionary containing the name of the runtime parameters of the step
        as keys and whether the parameter is required or not as values.

        Returns:
            A dictionary containing the name of the runtime parameters of the step as keys
            and whether the parameter is required or not as values.
        """

        runtime_parameters = {}

        for name, info in self.model_fields.items():
            is_runtime_param, is_optional = _is_runtime_parameter(info)
            if is_runtime_param:
                runtime_parameters[name] = is_optional

        return runtime_parameters

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
            if _is_step_input(parameter) and step_input_parameter is not None:
                raise TypeError(
                    f"Step '{self.name}' should have only one parameter with type hint `StepInput`."
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
            data (Dict[str, Any]): Dict containing the serialized data from a Step and the
                Pipeline it belongs to.

        Returns:
            step (Step): Instance of the Step.
        """
        if not (pipe := _GlobalPipelineManager.get_pipeline()):
            raise ValueError("A Step must be initialized in the context of a Pipeline.")

        # Remove the _type_info_ to avoid errors on instantiation
        _data = data.copy()
        if "_type_info_" in _data.keys():
            _data.pop("_type_info_")

        # Before passing the data to instantiate the general step, we have to instantiate some of the internal objects.
        # For the moment we only take into account the LLM, we should take care if we update any of the objects.
        if llm := _data.get("llm"):
            from distilabel.utils.serialization import _get_class

            nested_cls = _get_class(**llm.pop("_type_info_"))
            # Load the LLM and update the _data inplace
            nested_cls = nested_cls(**llm)
            _data.update({"llm": nested_cls})
        # Every step needs the pipeline, and the remaining arguments are general
        step = cls(pipeline=pipe, **_data)

        return step


class Step(_Step, ABC):
    input_batch_size: PositiveInt = DEFAULT_INPUT_BATCH_SIZE

    @abstractmethod
    def process(self, *inputs: StepInput) -> "StepOutput":
        pass

    def process_applying_mappings(self, *args: List[Dict[str, Any]]) -> "StepOutput":
        """Runs the `process` method of the step applying the `input_mappings` to the input
        rows and the `outputs_mappings` to the output rows. This is the function that
        should be used to run the processing logic of the step.

        Yields:
            The output rows.
        """

        inputs = self._apply_input_mappings(args) if self.input_mappings else args

        for output_rows in self.process(*inputs):
            yield [
                {
                    # Apply output mapping and revert input mapping
                    self.input_mappings.get(self.output_mappings.get(k, k), k): v
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
    """

    batch_size: int = 50

    @abstractmethod
    def process(self) -> "GeneratorStepOutput":
        """Method that defines the generation logic of the step. It should yield the
        output rows and a boolean indicating if it's the last batch or not."""
        pass

    def process_applying_mappings(self, *args: "StepInput") -> "GeneratorStepOutput":
        """Runs the `process` method of the step applying the `outputs_mappings` to the
        output rows. This is the function that should be used to run the generation logic
        of the step.

        Yields:
            The output rows and a boolean indicating if it's the last batch or not.
        """

        for output_rows, last_batch in self.process(*args):
            yield (
                [
                    {self.output_mappings.get(k, k): v for k, v in row.items()}
                    for row in output_rows
                ],
                last_batch,
            )


class GlobalStep(_Step, ABC):
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


def _is_step_input(parameter: inspect.Parameter) -> bool:
    """Check if the parameter has type hint `StepInput`.

    Args:
        parameter: The parameter to check.

    Returns:
        `True` if the parameter has type hint `StepInput`, `False` otherwise.
    """
    return (
        get_origin(parameter.annotation) is Annotated
        and get_args(parameter.annotation)[-1] == "StepInput"
    )


def _is_runtime_parameter(field: "FieldInfo") -> Tuple[bool, bool]:
    """Check if a `pydantic.BaseModel` field is a `RuntimeParameter` and if it's optional
    i.e. providing a value for the field in `Pipeline.run` is optional.

    Args:
        field: The info of the field of the `pydantic.BaseModel` to check.

    Returns:
        A tuple with two booleans. The first one indicates if the field is a
        `RuntimeParameter` or not, and the second one indicates if the field is optional
        or not.
    """
    # Case 1: `runtime_param: RuntimeParameter[int]`
    # Mandatory runtime parameter that needs to be provided when running the pipeline
    if _RUNTIME_PARAMETER_ANNOTATION in field.metadata:
        is_optional = field.default is not _DEFAULT_VALUE_RUNTIME_PARAMETER
        return True, is_optional

    # Case 2: `runtime_param: Union[RuntimeParameter[int], None] = None`
    # Optional runtime parameter that doesn't need to be provided when running the pipeline
    type_args = get_args(field.annotation)
    for arg in type_args:
        is_runtime_param = (
            get_origin(arg) is Annotated
            and get_args(arg)[-1] == _RUNTIME_PARAMETER_ANNOTATION
        )
        if is_runtime_param:
            is_optional = (
                get_origin(field.annotation) is Union and type(None) in type_args
            )
            return True, is_optional

    return False, False
