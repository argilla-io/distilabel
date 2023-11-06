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

from typing import List, Literal, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ChatCompletion(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


# TODO: add more output formats
# TODO: move this file outside as `prompt.py` or something more meaningful
class Prompt(BaseModel):
    system_prompt: str
    formatted_prompt: str

    def format_as(
        self, format: Literal["openai", "llama2"]
    ) -> Union[str, List[ChatCompletion]]:
        if format == "openai":
            return [
                ChatCompletion(
                    role="system",
                    content=self.system_prompt,
                ),
                ChatCompletion(role="user", content=self.formatted_prompt),
            ]
        elif format == "llama2":
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}<</SYS>>\n\n{self.formatted_prompt} [/INST]"
        else:
            raise ValueError(
                f"Format {format} not supported, please provide a custom `formatting_fn`."
            )
