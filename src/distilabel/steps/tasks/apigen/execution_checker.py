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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

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
    """Executes the generated function calls.

    This step checks if a given answer from a model as generated by `APIGenGenerator`
    can be executed against the given library (given by `libpath`, which is a string
    pointing to a python .py file with functions).

    Attributes:
        libpath: The path to the library where we will retrieve the functions.
            It can also point to a folder with the functions. In this case, the folder
            layout should be a folder with .py files, each containing a single function,
            the name of the function being the same as the filename.

    Input columns:
        - answers (`str`): List with arguments to be passed to the function,
            dumped as a string from a list of dictionaries. Should be loaded using
            `json.loads`.

    Output columns:
        - keep_row_after_execution_check (`bool`): Whether the function should be kept or not.
        - execution_result (`str`): The result from executing the function.

    Categories:
        - filtering
        - execution

    References:
        - [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)
        - [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

    Examples:
        Execute a function from a given library with the answer from an LLM:

        ```python
        from distilabel.steps.tasks import APIGenExecutionChecker

        # For the libpath you can use as an example the file at the tests folder:
        # ../distilabel/tests/unit/steps/tasks/apigen/_sample_module.py
        task = APIGenExecutionChecker(
            libpath="../distilabel/tests/unit/steps/tasks/apigen/_sample_module.py",
        )
        task.load()

        res = next(
            task.process(
                [
                    {
                        "answers": [
                            {
                                "arguments": {
                                    "initial_velocity": 0.2,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "final_velocity",
                            }
                        ],
                    }
                ]
            )
        )
        res
        #[{'answers': [{'arguments': {'initial_velocity': 0.2, 'acceleration': 0.1, 'time': 0.5}, 'name': 'final_velocity'}], 'keep_row_after_execution_check': True, 'execution_result': ['0.25']}]
        ```
    """

    libpath: str = Field(
        default=...,
        description=(
            "The path to the library where we will retrieve the functions, "
            "or a folder with python files named the same as the functions they contain.",
        ),
    )

    _toolbox: Union["ModuleType", None] = PrivateAttr(None)

    def load(self) -> None:
        """Loads the library where the functions will be extracted from."""
        super().load()
        if Path(self.libpath).suffix == ".py":
            self._toolbox = load_module_from_path(self.libpath)

    def unload(self) -> None:
        self._toolbox = None

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task are those found in the original dataset."""
        return ["answers"]

    @property
    def outputs(self) -> "StepColumns":
        """The outputs are the columns required by `APIGenGenerator` task."""
        return ["keep_row_after_execution_check", "execution_result"]

    def _get_function(self, function_name: str) -> Callable:
        """Retrieves the function from the toolbox.

        Args:
            function_name: The name of the function to retrieve.

        Returns:
            Callable: The function to be executed.
        """
        if self._toolbox:
            return getattr(self._toolbox, function_name, None)
        try:
            toolbox = load_module_from_path(
                str(Path(self.libpath) / f"{function_name}.py")
            )
            return getattr(toolbox, function_name, None)
        except FileNotFoundError:
            return None
        except Exception as e:
            self._logger.warning(f"Error loading function '{function_name}': {e}")
            return None

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
        for input in inputs:
            output = []
            if input["answers"]:
                answers = json.loads(input["answers"])
            else:
                input.update(
                    **{
                        "keep_row_after_execution_check": False,
                        "execution_result": ["No answers were provided."],
                    }
                )
                continue
            for answer in answers:
                if answer is None:
                    output.append(
                        {
                            "keep": False,
                            "execution_result": "Nothing was generated for this answer.",
                        }
                    )
                    continue

                function_name = answer.get("name", None)
                arguments = answer.get("arguments", None)

                function = self._get_function(function_name)

                if function is None:
                    output.append(
                        {
                            "keep": False,
                            "execution_result": f"Function '{function_name}' not found.",
                        }
                    )
                else:
                    # TODO: Based on the signature, try to cast the values before passing them to the function.
                    execution = execute_from_response(function, arguments)
                    output.append(
                        {
                            "keep": execution["keep"],
                            "execution_result": execution["execution_result"],
                        }
                    )
            # We only consider a good response if all the answers were executed successfully,
            # but keep the reasons for further review if needed.
            input.update(
                **{
                    "keep_row_after_execution_check": all(
                        o["keep"] is True for o in output
                    ),
                    "execution_result": [o["execution_result"] for o in output],
                }
            )

        yield inputs
