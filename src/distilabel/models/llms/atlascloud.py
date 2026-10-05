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
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field, PrivateAttr, SecretStr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.openai import OpenAILLM

if TYPE_CHECKING:
    from distilabel.typing import GenerateOutput

_ATLASCLOUD_API_KEY_ENV_VAR_NAME = "ATLASCLOUD_API_KEY"


class AtlasCloudLLM(OpenAILLM):
    """Atlas Cloud LLM implementation running the async API client of OpenAI.

    Attributes:
        model: the model name to use for the LLM e.g. "deepseek-ai/DeepSeek-V3.1-Terminus".
            Supported models can be found [here](https://atlascloud.ai/models).
        base_url: the base URL to use for the Atlas Cloud API can be set with `ATLASCLOUD_BASE_URL`.
            Defaults to `None` which means that the value set for the environment variable
            `ATLASCLOUD_BASE_URL` will be used, or "https://api.atlascloud.ai/v1" if not set.
        api_key: the API key to authenticate the requests to the Atlas Cloud API. Defaults to `None`
            which means that the value set for the environment variable `ATLASCLOUD_API_KEY` will be
            used, or `None` if not set.
        _api_key_env_var: the name of the environment variable to use for the API key. It
            is meant to be used internally.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import AtlasCloudLLM

        llm = AtlasCloudLLM(model="deepseek-ai/DeepSeek-V3.1-Terminus", api_key="api.key")

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "ATLASCLOUD_BASE_URL", "https://api.atlascloud.ai/v1"
        ),
        description="The base URL to use for the Atlas Cloud API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_ATLASCLOUD_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Atlas Cloud API.",
    )

    _api_key_env_var: str = PrivateAttr(_ATLASCLOUD_API_KEY_ENV_VAR_NAME)

    def load(self) -> None:
        """Loads the client and strips the log-probability parameters.

        Atlas Cloud rejects `logprobs` with a 400 even when it is sent as
        `False`, which is what `OpenAILLM` passes on every chat request. The
        parameters are removed at the client boundary so the inherited
        request-building code can stay untouched.
        """
        super().load()
        self._aclient = _StripLogprobs(self._aclient)


class _StripLogprobs:
    """Proxies an `AsyncOpenAI` client, dropping `logprobs` / `top_logprobs`."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def chat(self) -> Any:
        return _StripLogprobsChat(self._client.chat)


class _StripLogprobsChat:
    def __init__(self, chat: Any) -> None:
        self._chat = chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)

    @property
    def completions(self) -> Any:
        return _StripLogprobsCompletions(self._chat.completions)


class _StripLogprobsCompletions:
    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)

    async def create(self, **kwargs: Any) -> Any:
        kwargs.pop("logprobs", None)
        kwargs.pop("top_logprobs", None)
        return await self._completions.create(**kwargs)
