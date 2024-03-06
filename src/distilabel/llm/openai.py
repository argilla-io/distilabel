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
from typing import TYPE_CHECKING, Optional, Union

from openai import AsyncOpenAI
from pydantic import Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import Annotated

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


# TODO: OpenAI client can be used for AnyScale, TGI, vLLM, etc.
# https://github.com/vllm-project/vllm/blob/main/examples/openai_chatcompletion_client.py
class OpenAILLM(AsyncLLM):
    """OpenAI LLM implementation running the Async API client.

    Args:
        model: the model name to use for the LLM e.g. "gpt-3.5-turbo", "gpt-4", etc.
        api_key: the API key to authenticate the requests to the OpenAI API.
    """

    model: str = "gpt-3.5-turbo"
    api_key: Annotated[Optional[SecretStr], Field(validate_default=True)] = None

    _aclient: Optional["AsyncOpenAI"] = PrivateAttr(...)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_none(cls, v: Union[str, SecretStr, None]) -> SecretStr:
        """Ensures that either the `api_key` or the environment variable `OPENAI_API_KEY` are set.

        Additionally, the `api_key` when provided is casted to `pydantic.SecretStr` to prevent it
        from being leaked and/or included within the logs or the serialization of the object.
        """
        v = v or os.getenv("OPENAI_API_KEY", None)  # type: ignore
        if v is None:
            raise ValueError("You must provide an API key to use OpenAI.")
        if not isinstance(v, SecretStr):
            v = SecretStr(v)
        return v

    def load(self) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""
        self._aclient = AsyncOpenAI(
            api_key=self.api_key.get_secret_value(),  # type: ignore
            max_retries=6,
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(
        self,
        input: "ChatType",
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """Generates a response asynchronously, using the OpenAI Async API."""
        completion = await self._aclient.chat.completions.create(  # type: ignore
            messages=input,  # type: ignore
            model=self.model,
            max_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            timeout=50,
        )
        return completion.choices[0].message.content  # type: ignore
