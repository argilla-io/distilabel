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

from typing import List
import random
from distilabel.tasks.base import Task, get_template
from distilabel.tasks.text_generation.principles import Principles

_LLAMA2_TEXT_GENERATION_TEMPLATE = get_template("llama2-generation.jinja2")


class PrincipledGenerationTask(Task):
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )

    principles: List[List[str]] = [
        Principles.harmlessness, 
        Principles.helpfulness, 
        Principles.truthfulness,
        Principles.honesty,
        Principles.verbalized_calibration
    ]

    __jinja2_template__: str = _LLAMA2_TEXT_GENERATION_TEMPLATE

    def generate_prompt(self, input: str) -> str:
        principle = random.choice(self.principles)
        principle = "\n".join(principle)
        system_prompt = f"{self.system_prompt}\n{principle}"
        return self.template.render(system_prompt=system_prompt, input=input)

    def parse_output(self, output: str) -> dict[str, str]:
        return {"generations": output}

    @property
    def input_args_names(self) -> list[str]:
        return ["input"]

    @property
    def output_args_names(self) -> list[str]:
        return ["generations"]
