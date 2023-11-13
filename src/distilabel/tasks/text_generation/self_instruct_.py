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
#
# WARNING: THIS FILE NAME HAS BEEN PREPENDED WITH AN UNDERSCORE TO AVOID
# ANY POTENTIAL CONFLICT / COLLISSION WITH THE `self_instruct` PYTHON PACKAGE.

from typing import Dict, List

from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

_SELF_INSTRUCT_TEMPLATE = get_template("self-instruct.jinja2")


class SelfInstructTask(TextGenerationTask):
    system_prompt: str = (
        "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks."
        "You are given a task description and a set of instructions for how to write the prompts for a specific AI application."
    )
    application_description: str = "AI assistant"
    num_instructions: int = 5

    __jinja2_template__: str = _SELF_INSTRUCT_TEMPLATE

    def generate_prompt(self, input: str) -> Prompt:
        render_kwargs = {
            "application_description": self.application_description,
            "num_instructions": self.num_instructions,
            "input": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        return {"generations": output.split("\n")}
