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

from typing import TYPE_CHECKING, Callable, List, TypeVar, Union

if TYPE_CHECKING:
    from distilabel.steps.base import _Step

S = TypeVar("S", bound="_Step")


def requirements(requirements: Union[List[str]]) -> Callable[[S], S]:
    """Decorator to add requirements to a Step.

    When creating a custom step for a Pipeline that requires additional packages to be installed,
    (in case you want to distribute the pipeline) you can use this decorator to add the requirements.

    Args:
        requirements: List of requirements to be added to the step.

    Returns:
        The step with the requirements added.

    Example:

        ```python
        @requirements(["my_library>=1.0.1"])
        class CustomStep(Step):
            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                for input in inputs:
                    input["response"] = "unit test"
                yield inputs
        ```
    """

    def decorator(step: S) -> S:
        step.requirements = requirements
        return step

    return decorator
