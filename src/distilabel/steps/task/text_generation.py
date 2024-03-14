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

from typing import Any, Dict, List

from distilabel.steps.task.base import Task
from distilabel.steps.task.typing import ChatType


class TextGeneration(Task):
    """TextGeneration is a pre-defined task that defines the `instruction` as the input
    and `generation` as the output. This task is used to generate text based on the input
    instruction. The model_name is also returned as part of the output in order to enhance it.

    Columns:

    - `input`: instruction
    - `output`: generation, model_name
    """

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {"role": "user", "content": input[self.inputs[0]]},
        ]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(self, output: str, input: Dict[str, Any]) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {self.outputs[0]: output}
