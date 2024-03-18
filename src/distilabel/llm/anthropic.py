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

from pydantic import PrivateAttr, SecretStr

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from distilabel.steps.task.typing import ChatType

SamplingParams = None


class AnthropicLLM(AsyncLLM):
    """Anthropic LLM implementation running the Async API client.

    TO DO: update the docstring to include the new parameters.
    Args:
        model: the model name to use for the LLM e.g. "notus", etc.
        host: the host to use for the LLM e.g. "https://ollama.com".
        max_retries: the maximum number of retries for the LLM.
        timeout: the timeout for the LLM.

    """

    model = str
    api_key = Optional[SecretStr] = os.getenv("ANTHROPIC_API_KEY", None)  # type: ignore
    http_client: Union[str, None] = None
    max_retries = int = 5

    _aclient: Optional["AsyncAnthropic"] = PrivateAttr(...)

    def load(self, api_key: Optional[str] = None) -> None:
        """Loads the `AsyncAnthropic` client to use the Anthropic async API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError as ie:
            raise ImportError(
                "Anthropic Python client is not installed. Please install it using"
                " `pip install anthropic`."
            ) from ie

        self.api_key = self._handle_api_key_value(
            self_value=self.api_key, load_value=api_key, env_var="ANTHROPIC_API_KEY"
        )

        self._aclient = AsyncAnthropic(
            api_key=self.api_key.get_secret_value(),
            max_retries=6,
            http_client=self.http_client,
            # TO DO: add more: tiemout, follow_redirects
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(
        self,
        input: "ChatType",
        stream: bool = False,
        max_tokens: int = 128,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
    ) -> str:
        """Generates a response asynchronously, using the Anthropic Async API definition.
        TO DO: define defaults and add docstring.
        Args:
            input (ChatType): _description_
            stream (bool, optional): _description_. Defaults to False.

        Returns:
            str: _description_
        """
        completion = await self._aclient.messages.create(  # type: ignore
            model=self.model,
            messages=input,
            stream=stream,
            format=format,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return completion.content


# client = AsyncAnthropic(
#     # This is the default and can be omitted
#     api_key=os.environ.get("ANTHROPIC_API_KEY"),
# )
# async def main() -> None:
#     message = await client.messages.create(
#         max_tokens=1024,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Hello, Claude",
#             }
#         ],
#         model="claude-3-opus-20240229",
#     )
#     print(message.content)


# asyncio.run(main())
