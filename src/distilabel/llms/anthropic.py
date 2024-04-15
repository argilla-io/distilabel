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
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from httpx import AsyncClient
from pydantic import Field, PrivateAttr, SecretStr, validate_call
from typing_extensions import override

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.itertools import grouper

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


_ANTHROPIC_API_KEY_ENV_VAR_NAME = "ANTHROPIC_API_KEY"


class AnthropicLLM(AsyncLLM):
    """Anthropic LLM implementation running the Async API client.

    Attributes:
        model: the name of the model to use for the LLM e.g. "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", etc. Available models can be checked here:
            [Anthropic: Models overview](https://docs.anthropic.com/claude/docs/models-overview).
        api_key: the API key to authenticate the requests to the Anthropic API. If not provided,
            it will be read from `ANTHROPIC_API_KEY` environment variable.
        base_url: the base URL to use for the Anthropic API. Defaults to `None` which means
            that `https://api.anthropic.com` will be used internally.
        timeout: the maximum time in seconds to wait for a response. Defaults to `600.0`.
        max_retries: The maximum number of times to retry the request before failing. Defaults
            to `6`.
        http_client: if provided, an alternative HTTP client to use for calling Anthropic
            API. Defaults to `None`.
        _api_key_env_var: the name of the environment variable to use for the API key. It
            is meant to be used internally.
        _aclient: the `AsyncAnthropic` client to use for the Anthropic API. It is meant
            to be used internally. Set in the `load` method.

    Runtime parameters:
        - `api_key`: the API key to authenticate the requests to the Anthropic API. If not
            provided, it will be read from `ANTHROPIC_API_KEY` environment variable.
        - `base_url`: the base URL to use for the Anthropic API. Defaults to `"https://api.anthropic.com"`.
        - `timeout`: the maximum time in seconds to wait for a response. Defaults to `600.0`.
        - `max_retries`: the maximum number of times to retry the request before failing.
            Defaults to `6`.
    """

    model: str
    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        ),
        description="The base URL to use for the Anthropic API.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_ANTHROPIC_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Anthropic API.",
    )
    timeout: RuntimeParameter[float] = Field(
        default=600.0,
        description="The maximum time in seconds to wait for a response from the API.",
    )
    max_retries: RuntimeParameter[int] = Field(
        default=6,
        description="The maximum number of times to retry the request to the API before"
        " failing.",
    )
    http_client: Optional[AsyncClient] = Field(default=None, exclude=True)

    _api_key_env_var: str = PrivateAttr(default=_ANTHROPIC_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncAnthropic"] = PrivateAttr(...)

    def _check_model_exists(self) -> None:
        """Checks if the specified model exists in the available models."""
        from anthropic import AsyncAnthropic

        annotation = get_type_hints(AsyncAnthropic().messages.create).get("model", None)
        models = [
            value
            for type_ in get_args(annotation)
            if get_origin(type_) is Literal
            for value in get_args(type_)
        ]

        if self.model not in models:
            raise ValueError(
                f"Model {self.model} does not exist among available models. "
                f"The available models are {', '.join(models)}"
            )

    def load(self) -> None:
        """Loads the `AsyncAnthropic` client to use the Anthropic async API."""
        super().load()

        try:
            from anthropic import AsyncAnthropic
        except ImportError as ie:
            raise ImportError(
                "Anthropic Python client is not installed. Please install it using"
                " `pip install anthropic`."
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )

        self._check_model_exists()

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

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        max_tokens: int = 128,
        stop_sequences: Union[List[str], None] = None,
        temperature: float = 1.0,
        top_p: Union[float, None] = None,
        top_k: Union[int, None] = None,
    ) -> GenerateOutput:
        """Generates a response asynchronously, using the [Anthropic Async API definition](https://github.com/anthropics/anthropic-sdk-python).

        Args:
            input: a single input in chat format to generate responses for.
            max_tokens: the maximum number of new tokens that the model will generate. Defaults to `128`.
            stop_sequences: custom text sequences that will cause the model to stop generating. Defaults to `NOT_GIVEN`.
            temperature: the temperature to use for the generation. Set only if top_p is None. Defaults to `1.0`.
            top_p: the top-p value to use for the generation. Defaults to `NOT_GIVEN`.
            top_k: the top-k value to use for the generation. Defaults to `NOT_GIVEN`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        from anthropic._types import NOT_GIVEN

        completion = await self._aclient.messages.create(  # type: ignore
            model=self.model,
            system=(
                input.pop(0)["content"]
                if input and input[0]["role"] == "system"
                else NOT_GIVEN
            ),
            messages=input,  # type: ignore
            max_tokens=max_tokens,
            stream=False,
            stop_sequences=NOT_GIVEN if stop_sequences is None else stop_sequences,
            temperature=temperature,
            top_p=NOT_GIVEN if top_p is None else top_p,
            top_k=NOT_GIVEN if top_k is None else top_k,
        )
        generations = []
        if (content := completion.content[0].text) is None:
            self._logger.warning(
                f"Received no response using Anthropic client (model: '{self.model}')."
                f" Finish reason was: {completion.stop_reason}"
            )
        generations.append(content)
        return generations

    # TODO: remove this function once Anthropic client allows `n` parameter
    @override
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`.
        """

        async def agenerate(
            inputs: List["ChatType"], **kwargs: Any
        ) -> "GenerateOutput":
            """Internal function to parallelize the asynchronous generation of responses."""
            tasks = [
                asyncio.create_task(self.agenerate(input=input, **kwargs))
                for input in inputs
                for _ in range(num_generations)
            ]
            return [outputs[0] for outputs in await asyncio.gather(*tasks)]

        outputs = self.event_loop.run_until_complete(agenerate(inputs, **kwargs))
        return list(grouper(outputs, n=num_generations, incomplete="ignore"))
