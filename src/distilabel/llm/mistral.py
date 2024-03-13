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

import asyncio
import os
from typing import TYPE_CHECKING, Any, List, Optional, Union

from mistralai.async_client import MistralAsyncClient
from pydantic import Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import Annotated

from distilabel.llm.base import AsyncLLM
from distilabel.utils.itertools import grouper

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


class MistralLLM(AsyncLLM):
    """Mistral LLM implementation running the async API client.

    Args:
        api_key: the API key to authenticate the requests to the Mistral API.
        model: the model name to use for the LLM e.g. "mistral-tiny", "mistral-large", etc.
        endpoint: the endpoint to use for the Mistral API. Defaults to "https://api.mistral.ai".
        max_retries: the maximum number of retries to attempt when a request fails. Defaults to 5.next
        timeout: the maximum time in seconds to wait for a response. Defaults to 120.next
        max_concurrent_requests: the maximum number of concurrent requests to send. Defaults to 64.
    """

    api_key: Annotated[Optional[SecretStr], Field(validate_default=True)] = None
    model: str = "mistral-medium"
    endpoint: str = "https://api.mistral.ai"
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64

    _aclient: Optional["MistralAsyncClient"] = PrivateAttr(...)

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_none(cls, v: Union[str, SecretStr, None]) -> SecretStr:
        """Ensures that either the `api_key` or the environment variable `MISTRAL_API_KEY` are set.

        Additionally, the `api_key` when provided is casted to `pydantic.SecretStr` to prevent it
        from being leaked and/or included within the logs or the serialization of the object.
        """
        v = v or os.getenv("MISTRAL_API_KEY", None)  # type: ignore
        if v is None:
            raise ValueError("You must provide an API key to use Mistral.")
        if not isinstance(v, SecretStr):
            v = SecretStr(v)
        return v

    def load(self) -> None:
        """Loads the `MistralAsyncClient` client to benefit from async requests."""
        self._aclient = MistralAsyncClient(
            api_key=self.api_key.get_secret_value(),  # type: ignore
            endpoint=self.endpoint,
            max_retries=self.max_retries,
            timeout=self.timeout,
            max_concurrent_requests=self.max_concurrent_requests,
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    # TODO: add `num_generations` parameter once Mistral client allows `n` parameter
    async def agenerate(  # type: ignore
        self,
        input: "ChatType",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[Union[str, None]]:
        """Generates a response asynchronously, using the Mistral Async API."""
        completion = await self._aclient.chat(  # type: ignore
            messages=input,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        generations = []
        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using MistralAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations

    # TODO: remove this function once Mistral client allows `n` parameter
    def generate(
        self, inputs: List["ChatType"], num_generations: int = 1, **kwargs: Any
    ) -> List[List[Union[str, None]]]:
        async def agenerate(
            inputs: List["ChatType"], **kwargs: Any
        ) -> List[Union[str, None]]:
            tasks = [
                asyncio.create_task(self.agenerate(input=input, **kwargs))
                for input in inputs
                for _ in range(num_generations)
            ]
            return [outputs[0] for outputs in await asyncio.gather(*tasks)]

        results = asyncio.run(agenerate(inputs, **kwargs))
        return list(grouper(results, n=num_generations, incomplete="ignore"))
