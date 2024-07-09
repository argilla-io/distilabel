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

import copy
from typing import TYPE_CHECKING, Dict, Final, Literal, Union

import jinja2
from pydantic import BaseModel, PrivateAttr
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import StandardInput


MagpieAvailableTemplates = Literal[
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]


class MagpieChatTemplate(TypedDict):
    chat_template: str
    generate_instruction: str
    generate_instruction_with_system_prompt: str


MAGPIE_TEMPLATES: Final[Dict["MagpieAvailableTemplates", "MagpieChatTemplate"]] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
        "generate_instruction": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "generate_instruction_with_system_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
        "generate_instruction": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "generate_instruction_with_system_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    },
}


class MagpieChatTemplateMixin(BaseModel):
    model: str
    use_magpie_template: bool = False
    template: Union[MagpieChatTemplate, None] = None

    _chat_template: jinja2.Template = PrivateAttr(default=None)

    def load(self) -> None:
        if not self.use_magpie_template:
            return

        if self.template is None:
            self.template = MAGPIE_TEMPLATES[
                "meta-llama/Meta-Llama-3-8B-Instruct"
            ].copy()
            self._chat_template = jinja2.Template(self.template["chat_template"])

    def prepare_input(self, input: "StandardInput") -> Union[str, None]:
        if not self.use_magpie_template:
            return None

        assert self.template

        if len(input) == 0:
            return self.template["generate_instruction"]

        if len(input) == 1 and input[0]["role"] == "system":
            template = copy.copy(
                self.template["generate_instruction_with_system_prompt"]
            )
            return template.format(system_prompt=input[0]["content"])

        # TODO: case there are messages

        return None
