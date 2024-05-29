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

import warnings
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import StructuredInput


class StructuredGeneration(Task):
    use_system_prompt: bool = False

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `instruction` and the `grammar`.
        Optionally, if the `use_system_prompt` flag is set to True, then the
        `system_prompt` will be used too."""
        columns = ["instruction", "grammar"]
        if self.use_system_prompt:
            columns = ["system_prompt"] + columns
        return columns

    def format_input(self, input: Dict[str, Any]) -> StructuredInput:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        if not isinstance(input["instruction"], str):
            raise ValueError(
                f"Input `instruction` must be a string. Got: {input['instruction']}."
            )

        messages = [{"role": "user", "content": input["instruction"]}]
        if self.use_system_prompt:
            if "system_prompt" in input:
                messages.insert(
                    0, {"role": "system", "content": input["system_prompt"]}
                )
            else:
                warnings.warn(
                    "`use_system_prompt` is set to `True`, but no `system_prompt` in input batch, so it will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        return (messages, input.get("grammar", None))  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}
