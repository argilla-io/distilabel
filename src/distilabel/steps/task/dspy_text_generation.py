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

from typing import Any, Dict

import dspy
from dspy.signatures.signature import signature_to_template

from distilabel.steps.task.text_generation import TextGeneration
from distilabel.steps.task.typing import ChatType


class DSPyProgram(TextGeneration):
    program: dspy.Predict

    @property
    def demos(self):
        return self.program.demos

    @property
    def template(self):
        return signature_to_template(self.program.signature)

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        prompt = self.render_prompt(input["instruction"])
        return [
            {"role": "user", "content": prompt},
        ]

    def render_prompt(self, input: str) -> str:
        example = dspy.Example(
            **{
                "question": input,
            }
        )
        example.demos = self.demos
        return self.template(example)
