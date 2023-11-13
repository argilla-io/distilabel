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

from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

_LLAMA2_TEXT_GENERATION_TEMPLATE = get_template("llama2-generation.jinja2")


class Llama2TextGenerationTask(TextGenerationTask):
    __jinja2_template__: str = _LLAMA2_TEXT_GENERATION_TEMPLATE

    def generate_prompt(self, input: str) -> str:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(
                system_prompt=self.system_prompt, input=input
            ),
        ).format_as(
            "llama2"
        )  # type: ignore
