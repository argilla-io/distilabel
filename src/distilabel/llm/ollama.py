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

from typing import TYPE_CHECKING, Literal, Optional, Union

from ollama import AsyncClient
from pydantic import PrivateAttr

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from ollama import Options

    from distilabel.steps.task.typing import ChatType


class OllamalLLM(AsyncLLM):
    """Mistral LLM implementation running the Async API client.

    Args:
        model: the model name to use for the LLM e.g. "notus", etc.
        host: the host to use for the LLM e.g. "https://ollama.com".
        max_retries: the maximum number of retries for the LLM.
        timeout: the timeout for the LLM.

    """

    model: str = "notus"
    host: Union[str, None] = None
    timeout: int = 120
    follow_redirects: bool = True

    _aclient: Optional["AsyncClient"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncClient` and `Client` for Ollama."""
        self._aclient = AsyncClient(
            host=self.host,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(
        self,
        inputs: "ChatType",
        stream: bool = False,
        format: Literal["", "json"] = "",
        options: Union["Options", None] = None,
    ) -> str:
        """Generates a response asynchronously, using the Ollama Async API definition."""
        completion = await self._aclient.chat(  # type: ignore
            model=self.model,
            messages=inputs,
            stream=stream,
            format=format,
            options=options,
        )

        return completion["message"]["content"]  # type: ignore
