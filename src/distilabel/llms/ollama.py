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

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union

from pydantic import Field, PrivateAttr, validate_call
from typing_extensions import TypedDict

from distilabel.llms.base import AsyncLLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from ollama import AsyncClient


# Copied from `ollama._types.Options`
class Options(TypedDict, total=False):
    # load time options
    numa: bool
    num_ctx: int
    num_batch: int
    num_gqa: int
    num_gpu: int
    main_gpu: int
    low_vram: bool
    f16_kv: bool
    logits_all: bool
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    embedding_only: bool
    rope_frequency_base: float
    rope_frequency_scale: float
    num_thread: int

    # runtime options
    num_keep: int
    seed: int
    num_predict: int
    top_k: int
    top_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    temperature: float
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_newline: bool
    stop: Sequence[str]


class OllamaLLM(AsyncLLM):
    """Ollama LLM implementation running the Async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "notus".
        host: the Ollama server host.
        timeout: the timeout for the LLM. Defaults to `120`.
        _aclient: the `AsyncClient` to use for the Ollama API. It is meant to be used internally.
            Set in the `load` method.

    Runtime parameters:
        - `host`: the Ollama server host.
        - `timeout`: the client timeout for the Ollama API. Defaults to `120`.
    """

    model: str
    host: Optional[RuntimeParameter[str]] = Field(
        default=None, description="The host of the Ollama API."
    )
    timeout: RuntimeParameter[int] = Field(
        default=120, description="The timeout for the Ollama API."
    )
    follow_redirects: bool = True

    _aclient: Optional["AsyncClient"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncClient` to use Ollama async API."""
        super().load()

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

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        num_generations: int = 1,
        format: Literal["", "json"] = "",
        # TODO: include relevant options from `Options` in `agenerate` method.
        options: Union[Options, None] = None,
        keep_alive: Union[bool, None] = None,
    ) -> List[str]:
        """
        Generates a response asynchronously, using the [Ollama Async API definition](https://github.com/ollama/ollama-python).

        Args:
            input: the input to use for the generation.
            num_generations: the number of generations to produce. Defaults to `1`.
            format: the format to use for the generation. Defaults to `""`.
            options: the options to use for the generation. Defaults to `None`.
            keep_alive: whether to keep the connection alive. Defaults to `None`.

        Returns:
            A list of strings as completion for the given input.
        """
        generations = []
        # TODO: remove this for-loop and override the `generate` method
        for _ in range(num_generations):
            completion = await self._aclient.chat(  # type: ignore
                model=self.model,
                messages=input,  # type: ignore
                stream=False,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )
            # TODO: improve error handling
            generations.append(completion["message"]["content"])

        return generations
