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

import re
from dataclasses import dataclass
from typing import Dict, List, Union

from distilabel.tasks.base import Task
from distilabel.tasks.prompt import ChatCompletion, SupportedFormats


@dataclass
class Chat:
    messages: Union[str, List[ChatCompletion]]

    def format_as(
        self, input_format: SupportedFormats, output_format: SupportedFormats
    ) -> Union[str, List[ChatCompletion]]:
        # First we convert the `messages` into a list of `ChatCompletion` objects, if necessary.
        # Unless the input and output formats are both the same, so we just need to append the
        # last message to the str or list.
        messages = None
        if input_format == "openai":
            messages = self.messages
        elif input_format == "chatml" and isinstance(self.messages, str):
            pattern = r"\<\|im_start\|\>(\bsystem\b|\buser\b|\bassistant\b)\s+(.*?)\<\|im_end\|\>"
            turns = re.findall(pattern, self.messages)
            messages = [{"role": turn[0], "content": turn[1]} for turn in turns]
        elif input_format == "llama2" and isinstance(self.messages, str):
            pattern = r"<s>\[INST\](?: <<SYS>>\n(.*?)\n<</SYS>>\n\n)?(.*?)\s\[\/INST\]\s(.*?)<\/s>"
            turns = re.findall(pattern, self.messages)
            messages = []
            for turn in turns:
                for role, content in zip(["system", "user", "assistant"], turn):
                    if content:
                        messages.append({"role": role, "content": content})
        else:
            raise ValueError(
                f"`input_format={input_format}` of `type={type(input_format)}` not supported,"
                " please provide a custom `prompt_formatting_fn` or use any of the available"
                f" formats: {SupportedFormats}"
            )

        # Then we format it as the specified output format.
        if output_format == "openai":
            return messages  # type: ignore
        else:
            raise ValueError(
                f"`output_format={output_format}` of `type={type(output_format)}` not supported,"
                " please provide a custom `prompt_formatting_fn` or use any of the available"
                f" formats: {SupportedFormats}"
            )


@dataclass
class ChatTask(Task):
    @property
    def input_args_names(self) -> List[str]:
        return ["messages"]

    @property
    def output_args_names(self) -> List[str]:
        return ["follow_up"]

    def generate_prompt(self, messages: Union[str, List[ChatCompletion]]) -> Chat:
        return messages  # type: ignore

    def parse_output(self, output: str) -> Dict[str, ChatCompletion]:
        return {"follow_up": ChatCompletion(role="assistant", content=output)}
