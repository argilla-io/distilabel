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

from dataclasses import dataclass
from typing import List, Literal, Union

from typing_extensions import TypedDict


class ChatCompletion(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


SupportedFormats = Literal["default", "openai", "llama2", "chatml", "zephyr"]


@dataclass
class Prompt:
    system_prompt: str
    formatted_prompt: str

    def format_as(self, format: SupportedFormats) -> Union[str, List[ChatCompletion]]:
        if format == "default":
            return f"{self.system_prompt}\n{self.formatted_prompt}"
        elif format == "openai":
            return [
                ChatCompletion(
                    role="system",
                    content=self.system_prompt,
                ),
                ChatCompletion(role="user", content=self.formatted_prompt),
            ]
        elif format == "llama2":
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}<</SYS>>\n\n{self.formatted_prompt} [/INST]"
        elif format == "chatml":
            return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{self.formatted_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif format == "zephyr":
            return f"<|system|>\n{self.system_prompt}</s>\n<|user|>\n{self.formatted_prompt}</s>\n<|assistant|>\n"
        else:
            raise ValueError(
                f"Format {format} not supported, please provide a custom `prompt_formatting_fn`"
                " or use any of the available formats: openai, llama2, chatml, zephyr"
            )
