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

# - Try to import the function from a given module
# - If function, try to import it and run it
# - If fails, track the error message, and return it

from typing import TYPE_CHECKING, Callable, Union

import orjson
from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.steps.base import Step, StepInput
from distilabel.steps.tasks.apigen.utils import (
    execute_from_response,
    load_module_from_path,
)

if TYPE_CHECKING:
    from types import ModuleType

    from distilabel.steps.typing import StepColumns, StepOutput


class APIGenExecutionChecker(Step):
    """
    Implements a CodeAgent?
    # TODO: Maybe the implementation from does the job here?
    # https://huggingface.co/docs/transformers/en/agents#code-agent

    # NOTE: In load() we may need to add 'pip install transformers[agents]'

    The answer is a list of dictionaries with the following keys:
    pass

    Attributes:
        libpath (str): The path to the library where we will retrieve the functions.
    """

    libpath: str = Field(
        default=...,
        description="The path to the library where we will retrieve the functions.",
    )

    _toolbox: Union["ModuleType", None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the library where the functions will be extracted from."""
        # TODO: Place a user error in case the path is not valid, to show what this really is.
        self._toolbox = load_module_from_path(self.libpath)

    def unload(self) -> None:
        # TODO: Unload the variables setting them to None.
        self._toolbox = None

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task are those found in the original dataset."""
        return ["answers"]

    @property
    def outputs(self) -> "StepColumns":
        """The outputs are the columns required by `APIGenGenerator` task."""
        return ["keep_row_after_execution_check", "reason"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        """Checks the answer to see if it can be executed.
        Captures the possible errors and returns them.

        If a single example is provided, it is copied to avoid raising an error.

        Args:
            inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        outputs = []
        for input in inputs:
            answers = orjson.loads(input["answers"])
            _output = []
            for answer in answers:
                function_name = answer.get("name", None)
                arguments = answer.get("arguments", None)

                function: Callable = getattr(self._toolbox, function_name)
                execution = execute_from_response(function, arguments)
                _output.append(
                    {
                        "keep": execution["keep"],
                        "reason": execution["error"],
                    }
                )
            # We only consider a good response if all the answers were executed successfully, but keep the reasons
            # for further review if needed.
            outputs.append(
                {
                    "keep_row_after_execution_check": all(
                        o["keep"] is True for o in _output
                    ),
                    "reason": [o["reason"] for o in _output],
                }
            )
        yield outputs
