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

import os
from typing import Optional

from openai import OpenAI

from distilabel.pipeline.llm.base import LLM
from distilabel.pipeline.step.task.types import ChatType


class OpenAILLM(LLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or "gpt-3.5-turbo"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if self.api_key is None:
            raise ValueError("You must provide an API key to use OpenAI.")

    def load(self) -> None:
        self.client = OpenAI(api_key=self.api_key, max_retries=6)

    def format_input(self, input: ChatType) -> ChatType:
        return input

    def generate(
        self,
        input: ChatType,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        chat_completions = self.client.chat.completions.create(
            messages=input,  # type: ignore
            model=self.model,
            max_new_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            timeout=50,
        )
        return chat_completions.choices[0].message.content  # type: ignore
