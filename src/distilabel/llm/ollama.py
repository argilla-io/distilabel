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

from typing import TYPE_CHECKING, List, Literal, Optional, Union

from pydantic import PrivateAttr

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from ollama import AsyncClient, Options

    from distilabel.steps.task.typing import ChatType


class OllamaLLM(AsyncLLM):
    """Ollama LLM implementation running the Async API client.

    Args:
        model: the model name to use for the LLM e.g. "notus", etc.
        host: the host to use for the LLM e.g. "https://ollama.com".
        max_retries: the maximum number of retries for the LLM.
        timeout: the timeout for the LLM.

    """

    model: str
    host: Union[str, None] = None
    timeout: int = 120
    follow_redirects: bool = True

    _aclient: Optional["AsyncClient"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncClient` to use Ollama async API."""
        try:
            from ollama import AsyncClient

            self._aclient = AsyncClient(
                host=self.host,
                timeout=self.timeout,
                follow_redirects=self.follow_redirects,
            )
        except ImportError as e:
            raise ImportError(
                "Ollama Python client is not installed. Please install it using"
                " `pip install ollama`."
            ) from e

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(
        self,
        input: "ChatType",
        num_generations: int = 1,
        format: Literal["", "json"] = "",
        options: Union["Options", None] = None,
        keep_alive: Union[bool, None] = None,
    ) -> List[str]:
        """
        Generates a response asynchronously, using the [Ollama Async API definition](https://github.com/ollama/ollama-python).

        Args:
            input: the input to use for the generation.
            num_generations: the number of generations to produce. Defaults to `1`.
            format: the format to use for the generation. Defaults to "".
            options: the options to use for the generation. Defaults to `None`.
            keep_alive: whether to keep the connection alive. Defaults to `None`.

        Returns:
            A list of strings as completion for the given input.
        """
        generations = []
        for _ in range(num_generations):
            completion = await self._aclient.chat(
                model=self.model,
                messages=input,
                stream=False,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )
            generations.append(completion["message"]["content"])

        return generations
