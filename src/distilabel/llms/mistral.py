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
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import Field, PrivateAttr, SecretStr, validate_call
from typing_extensions import override

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.itertools import grouper

if TYPE_CHECKING:
    from mistralai.async_client import MistralAsyncClient


_MISTRALAI_API_KEY_ENV_VAR_NAME = "MISTRAL_API_KEY"


class MistralLLM(AsyncLLM):
    """Mistral LLM implementation running the async API client.

    Attributes:
        model: the model name to use for the LLM e.g. "mistral-tiny", "mistral-large", etc.
        endpoint: the endpoint to use for the Mistral API. Defaults to "https://api.mistral.ai".
        api_key: the API key to authenticate the requests to the Mistral API. Defaults to `None` which
            means that the value set for the environment variable `OPENAI_API_KEY` will be used, or
            `None` if not set.
        max_retries: the maximum number of retries to attempt when a request fails. Defaults to `5`.
        timeout: the maximum time in seconds to wait for a response. Defaults to `120`.
        max_concurrent_requests: the maximum number of concurrent requests to send. Defaults
            to `64`.
        _api_key_env_var: the name of the environment variable to use for the API key. It is meant to
            be used internally.
        _aclient: the `MistralAsyncClient` to use for the Mistral API. It is meant to be used internally.
            Set in the `load` method.

    Runtime parameters:
        - `api_key`: the API key to authenticate the requests to the Mistral API.
        - `max_retries`: the maximum number of retries to attempt when a request fails.
            Defaults to `5`.
        - `timeout`: the maximum time in seconds to wait for a response. Defaults to `120`.
        - `max_concurrent_requests`: the maximum number of concurrent requests to send.
            Defaults to `64`.
    """

    model: str
    endpoint: str = "https://api.mistral.ai"
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_MISTRALAI_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Mistral API.",
    )
    max_retries: RuntimeParameter[int] = Field(
        default=6,
        description="The maximum number of times to retry the request to the API before"
        " failing.",
    )
    timeout: RuntimeParameter[int] = Field(
        default=120,
        description="The maximum time in seconds to wait for a response from the API.",
    )
    max_concurrent_requests: RuntimeParameter[int] = Field(
        default=64, description="The maximum number of concurrent requests to send."
    )

    _api_key_env_var: str = PrivateAttr(_MISTRALAI_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["MistralAsyncClient"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `MistralAsyncClient` client to benefit from async requests."""
        super().load()

        try:
            from mistralai.async_client import MistralAsyncClient
        except ImportError as ie:
            raise ImportError(
                "MistralAI Python client is not installed. Please install it using"
                " `pip install mistralai`."
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )

        self._aclient = MistralAsyncClient(
            api_key=self.api_key.get_secret_value(),
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
    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for the given input using the MistralAI async
        client.

        Args:
            input: a single input in chat format to generate responses for.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        completion = await self._aclient.chat(  # type: ignore
            messages=input,
            model=self.model,
            temperature=temperature,
            max_tokens=max_new_tokens,
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
