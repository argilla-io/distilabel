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
from typing import TYPE_CHECKING, Callable, List, Literal, Type, TypeVar, Union

from pydantic import create_model

from distilabel.steps.base import (
    _RUNTIME_PARAMETER_ANNOTATION,
    GlobalStep,
    Step,
)
from distilabel.utils.typing import is_parameter_annotated_with

if TYPE_CHECKING:
    from distilabel.steps.base import _Step
    from distilabel.steps.typing import GeneratorStepOutput, StepOutput

ProcessingFunc = TypeVar(
    "ProcessingFunc", bound=Callable[..., Union["StepOutput", "GeneratorStepOutput"]]
)


def step(
    inputs: Union[List[str], None] = None,
    outputs: Union[List[str], None] = None,
    step_type: Literal["normal", "global", "generator"] = "normal",
) -> Callable[..., Type["_Step"]]:
    """Creates an `Step` from a processing function.

    Args:
        inputs: a list containing the name of the inputs columns/keys expected by this step.
            If not provided the default will be an empty list `[]` and it will be assumed
            that the step doesn't need any spefic columns. Defaults to `None`.
        outputs: a list containing the name of the outputs columns/keys that the step
            will generate. If not provided the default will be an empty list `[]` and it
            will be assumed that the step doesn't need any spefic columns. Defaults to
            `None`.
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
    @step(inputs=["instruction"], outputs=["generation"], step_type="generator")
    def RowGenerator(inputs: StepInput, num_rows: RuntimeParameter[int] = 500) -> GeneratorStepOutput:
        data = list(range(num_rows))
        for i in range(0, len(data), 100):
            last_batch = i + 100 >= len(data)
            yield data[i:i + 100], last_batch
    ```
    """
    inputs = inputs or []
    outputs = outputs or []

    def decorator(
        func: Callable[..., Union["StepOutput", "GeneratorStepOutput"]],
    ) -> Type["_Step"]:
        signature = inspect.signature(func)

        runtime_parameters = {
            name: (
                param.annotation,
                param.default if param.default != param.empty else None,
            )
            for name, param in signature.parameters.items()
            if is_parameter_annotated_with(param, _RUNTIME_PARAMETER_ANNOTATION)
        }

        RuntimeParametersModel = create_model(  # type: ignore
            "RuntimeParametersModel",
            **runtime_parameters,  # type: ignore
        )

        BaseClass = Step if step_type == "normal" else GlobalStep

        def inputs_property(self) -> List[str]:
            return inputs

        def outputs_property(self) -> List[str]:
            return outputs

        return type(  # type: ignore
            func.__name__,
            (
                BaseClass,
                RuntimeParametersModel,
            ),
            {
                "process": func,
                "inputs": property(inputs_property),
                "outputs": property(outputs_property),
                "__module__": func.__module__,
                "__doc__": func.__doc__,
                "_built_from_decorator": True,
            },
        )

    return decorator
