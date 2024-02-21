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

from distilabel.pipeline.step.task.base import Task
from distilabel.pipeline.step.task.types import ChatType


class TextGeneration(Task):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": input[self.inputs[0]]},
        ]

    @property
    def outputs(self) -> List[str]:
        return ["generation"]

    def format_output(self, output: str) -> Dict[str, Any]:
        return {self.outputs[0]: output}
