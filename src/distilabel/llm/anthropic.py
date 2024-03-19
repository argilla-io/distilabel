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

from pydantic import PrivateAttr, SecretStr

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from distilabel.llm.typing import GenerateOutput
    from distilabel.steps.task.typing import ChatType


class AnthropicLLM(AsyncLLM):
    """Anthropic LLM implementation running the Async API client.

    Args:
        model: the model name to use for the LLM.
        api_key: the API key to authenticate the requests to the Anthropic API.
        base_url: the base URL to use for the Anthropic API. Defaults to "https://api.anthropic.com".
        http_client: the HTTP client to use for the Anthropic API. Defaults to None.
        timeout: the maximum time in seconds to wait for a response. Defaults to 600.0.
        max_retries: the maximum number of retries for the LLM. Defaults to 2.
    """

    model: str = "claude-3-opus-20240229"
    api_key: Optional[SecretStr] = os.getenv("ANTHROPIC_API_KEY", None)  # type: ignore
    base_url: str = "https://api.anthropic.com"
    timeout: float = 600.0
    http_client: Union[str, None] = None
    max_retries: int = 2

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
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=self.http_client,
            max_retries=self.max_retries,
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(  # type: ignore
        self,
        input: "ChatType",
        system: str = "",
        num_generations: int = 1,
        max_tokens: int = 128,
        stop_sequences: List[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> "GenerateOutput":
        """Generates a response asynchronously, using the [Anthropic Async API definition](https://github.com/anthropics/anthropic-sdk-python).

        Args:
            input: a single input in chat format to generate responses for.
            system: the system prompt to use for the generation. No existing 'system' role. Defaults to `""`.
            num_generations: the number of generations to create per input. Defaults to `1`.
            max_tokens: the maximum number of new tokens that the model will generate. Defaults to `128`.
            stop_sequences: custom text sequences that will cause the model to stop generating. Defaults to None.
            temperature: the temperature to use for the generation. Set only if top_p is None. Defaults to `1.0`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        generations = []
        for _ in range(num_generations):
            completion = await self._aclient.messages.create(
                model=self.model,
                system=system,
                messages=input,
                max_tokens=max_tokens,
                stream=False,
                stop_sequences=stop_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generations.append(completion.content[0].text)
        return generations
