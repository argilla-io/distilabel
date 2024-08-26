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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Type,
    Union,
    overload,
)

from pydantic import create_model

from distilabel.mixins.runtime_parameters import _RUNTIME_PARAMETER_ANNOTATION
from distilabel.steps.base import (
    _STEP_INPUT_ANNOTATION,
    GeneratorStep,
    GlobalStep,
    Step,
)
from distilabel.utils.typing_ import is_parameter_annotated_with

if TYPE_CHECKING:
    from distilabel.steps.base import _Step
    from distilabel.steps.typing import GeneratorStepOutput, StepColumns, StepOutput

_STEP_MAPPING = {
    "normal": Step,
    "global": GlobalStep,
    "generator": GeneratorStep,
}

ProcessingFunc = Callable[..., Union["StepOutput", "GeneratorStepOutput"]]


@overload
def step(
    inputs: Union["StepColumns", None] = None,
    outputs: Union["StepColumns", None] = None,
    step_type: Literal["normal"] = "normal",
) -> Callable[..., Type["Step"]]: ...


@overload
def step(
    inputs: Union["StepColumns", None] = None,
    outputs: Union["StepColumns", None] = None,
    step_type: Literal["global"] = "global",
) -> Callable[..., Type["GlobalStep"]]: ...


@overload
def step(
    inputs: None = None,
    outputs: Union["StepColumns", None] = None,
    step_type: Literal["generator"] = "generator",
) -> Callable[..., Type["GeneratorStep"]]: ...


def step(
    inputs: Union["StepColumns", None] = None,
    outputs: Union["StepColumns", None] = None,
    step_type: Literal["normal", "global", "generator"] = "normal",
) -> Callable[..., Type["_Step"]]:
    """Creates an `Step` from a processing function.

    Args:
        inputs: a list containing the name of the inputs columns/keys or a dictionary
            where the keys are the columns and the values are booleans indicating whether
            the column is required or not, that are required by the step. If not provided
            the default will be an empty list `[]` and it will be assumed that the step
            doesn't need any specific columns. Defaults to `None`.
        outputs: a list containing the name of the outputs columns/keys or a dictionary
            where the keys are the columns and the values are booleans indicating whether
            the column will be generated or not. If not provided the default will be an
            empty list `[]` and it will be assumed that the step doesn't need any specific
            columns. Defaults to `None`.
        step_type: the kind of step to create. Valid choices are: "normal" (`Step`),
            "global" (`GlobalStep`) or "generator" (`GeneratorStep`). Defaults to
            `"normal"`.

    Returns:
        A callable that will generate the type given the processing function.

    Example:

    ```python
    # Normal step
    @step(inputs=["instruction"], outputs=["generation"])
    def GenerationStep(inputs: StepInput, dummy_generation: RuntimeParameter[str]) -> StepOutput:
        for input in inputs:
            input["generation"] = dummy_generation
        yield inputs

    # Global step
    @step(inputs=["instruction"], step_type="global")
    def FilteringStep(inputs: StepInput, max_length: RuntimeParameter[int] = 256) -> StepOutput:
        yield [
            input
            for input in inputs
            if len(input["instruction"]) <= max_length
        ]

    # Generator step
    @step(outputs=["num"], step_type="generator")
    def RowGenerator(num_rows: RuntimeParameter[int] = 500) -> GeneratorStepOutput:
        data = list(range(num_rows))
        for i in range(0, len(data), 100):
            last_batch = i + 100 >= len(data)
            yield [{"num": num} for num in data[i : i + 100]], last_batch
    ```
    """

    inputs = inputs or []
    outputs = outputs or []

    def decorator(func: ProcessingFunc) -> Type["_Step"]:
        if step_type not in _STEP_MAPPING:
            raise ValueError(
                f"Invalid step type '{step_type}'. Please, review the '{func.__name__}'"
                " function decorated with the `@step` decorator and provide a valid"
                " `step_type`. Valid choices are: 'normal', 'global' or 'generator'."
            )

        BaseClass = _STEP_MAPPING[step_type]

        signature = inspect.signature(func)

        runtime_parameters = {
            name: (
                param.annotation,
                param.default if param.default != param.empty else None,
            )
            for name, param in signature.parameters.items()
        }

        runtime_parameters = {}
        step_input_parameter = None
        for name, param in signature.parameters.items():
            if is_parameter_annotated_with(param, _RUNTIME_PARAMETER_ANNOTATION):
                runtime_parameters[name] = (
                    param.annotation,
                    param.default if param.default != param.empty else None,
                )

            if not step_type == "generator" and is_parameter_annotated_with(
                param, _STEP_INPUT_ANNOTATION
            ):
                if step_input_parameter is not None:
                    raise ValueError(
                        f"Function '{func.__name__}' has more than one parameter annotated"
                        f" with `StepInput`. Please, review the '{func.__name__}' function"
                        " decorated with the `@step` decorator and provide only one"
                        " argument annotated with `StepInput`."
                    )
                step_input_parameter = param

        RuntimeParametersModel = create_model(  # type: ignore
            "RuntimeParametersModel",
            **runtime_parameters,  # type: ignore
        )

        def inputs_property(self) -> List[str]:
            return inputs

        def outputs_property(self) -> List[str]:
            return outputs

        def process(
            self, *args: Any, **kwargs: Any
        ) -> Union["StepOutput", "GeneratorStepOutput"]:
            return func(*args, **kwargs)

        return type(  # type: ignore
            func.__name__,
            (
                BaseClass,
                RuntimeParametersModel,
            ),
            {
                "process": process,
                "inputs": property(inputs_property),
                "outputs": property(outputs_property),
                "__module__": func.__module__,
                "__doc__": func.__doc__,
                "_built_from_decorator": True,
                # Override the `get_process_step_input` method to return the parameter
                # of the original function annotated with `StepInput`.
                "get_process_step_input": lambda self: step_input_parameter,
            },
        )

    return decorator
