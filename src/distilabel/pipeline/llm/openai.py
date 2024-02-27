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
from typing import TYPE_CHECKING, List, Optional, Union

from openai import OpenAI
from pydantic import PrivateAttr, SecretStr, field_validator

from distilabel.pipeline.llm.base import LLM

if TYPE_CHECKING:
    from distilabel.pipeline.step.task.typing import ChatType


# TODO: OpenAI client can be used for AnyScale, TGI, vLLM, etc.
# https://github.com/vllm-project/vllm/blob/main/examples/openai_chatcompletion_client.py
class OpenAILLM(LLM):
    model: str = "gpt-3.5-turbo"
    api_key: Optional[SecretStr] = None

    _client: Optional["OpenAI"] = PrivateAttr(...)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_none(cls, v: Union[SecretStr, None]) -> SecretStr:
        v = v or os.getenv("OPENAI_API_KEY", None)  # type: ignore
        if v is None:
            raise ValueError("You must provide an API key to use OpenAI.")
        if not isinstance(v, SecretStr):
            v = SecretStr(v)
        return v

    def load(self) -> None:
        self._client = OpenAI(api_key=self.api_key.get_secret_value(), max_retries=6)  # type: ignore

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        inputs: List["ChatType"],
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[str]:
        outputs = []
        for input in inputs:
            chat_completions = self._client.chat.completions.create(  # type: ignore
                messages=input,  # type: ignore
                model=self.model,
                max_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
                timeout=50,
            )
            outputs.append(chat_completions.choices[0].message.content)  # type: ignore
        return outputs
