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
import sys
from typing import TYPE_CHECKING, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import PrivateAttr, SecretStr, model_validator

from distilabel.llm.base import AsyncLLM

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from distilabel.llm.typing import GenerateOutput
    from distilabel.steps.task.typing import ChatType


class OpenAILLM(AsyncLLM):
    """OpenAI LLM implementation running the async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "gpt-3.5-turbo", "gpt-4", etc.
            Supported models can be found [here](https://platform.openai.com/docs/guides/text-generation).
        base_url: the base URL to use for the OpenAI API requests. Defaults to `None`, which means that
            the value set for the environment variable `OPENAI_BASE_URL` will be used, or "https://api.openai.com/v1"
            if not set.
        api_key: the API key to authenticate the requests to the OpenAI API. Defaults to `None` which
            means that the value set for the environment variable `OPENAI_API_KEY` will be used, or
            `None` if not set.
    """

    model: str
    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = None

    _base_url_env_var: str = PrivateAttr(default="OPENAI_BASE_URL")
    _default_base_url: str = PrivateAttr("https://api.openai.com/v1")
    _api_key_env_var: str = PrivateAttr(default="OPENAI_API_KEY")
    _aclient: Optional["AsyncOpenAI"] = PrivateAttr(...)

    @model_validator(mode="after")
    def base_url_provided_or_env_var(self) -> Self:
        if self.base_url is None:
            self.base_url = os.getenv(self._base_url_env_var, self._default_base_url)
        if self.api_key is None:
            self.api_key = os.getenv(self._api_key_env_var, None)  # type: ignore
        if self.api_key is not None and isinstance(self.api_key, str):
            self.api_key = SecretStr(self.api_key)
        return self

    def load(self, api_key: Optional[Union[str, SecretStr]] = None) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""

        try:
            from openai import AsyncOpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install openai`."
            ) from ie

        # TODO: this may not be needed at the end, and a simple `self.api_key = self.api_key or api_key`
        # may do the work.
        self.api_key = self._handle_api_key_value(
            self_value=self.api_key,
            load_value=api_key,
            env_var=self._env_var,  # type: ignore
        )

        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=6,
        )

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    async def agenerate(  # type: ignore
        self,
        input: "ChatType",
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> "GenerateOutput":
        """Generates `num_generations` responses for the given input using the OpenAI async
        client.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximun number of new tokens that the model will generate.
                Defaults to `128`.
            frequence_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            top_k: the top-k value to use for the generation. Defaults to `0`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        completion = await self._aclient.chat.completions.create(  # type: ignore
            messages=input,  # type: ignore
            model=self.model,
            max_tokens=max_new_tokens,
            n=num_generations,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            timeout=50,
        )
        generations = []
        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(
                    f"Received no response using OpenAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations
