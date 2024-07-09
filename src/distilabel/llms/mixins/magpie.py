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

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import StandardInput


class MagpieChatTemplateMixin(BaseModel):
    model: str
    use_magpie_template: bool = False
    # TODO: harcoded to llama 3
    pre_query_template: str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    )

    def apply_pre_query_template(self, prompt: str, input: "StandardInput") -> str:
        if not self.use_magpie_template or input[-1]["role"] == "assistant":
            return prompt
        return prompt + self.pre_query_template
