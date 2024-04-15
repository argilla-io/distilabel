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
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import Field, PrivateAttr, SecretStr, validate_call
from typing_extensions import override

from distilabel.llms.base import AsyncLLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.itertools import grouper

if TYPE_CHECKING:
    from cohere import AsyncClient, ChatMessage

    from distilabel.llms.typing import GenerateOutput

_COHERE_API_KEY_ENV_VAR_NAME = "COHERE_API_KEY"


class CohereLLM(AsyncLLM):
    """Cohere API implementation using the async client for concurrent text generation.


    Attributes:
        model: the name of the model from the Cohere API to use for the generation.
        base_url: the base URL to use for the Cohere API requests. Defaults to
            `"https://api.cohere.ai/v1"`.
        api_key: the API key to authenticate the requests to the Cohere API. Defaults to
            the value of the `COHERE_API_KEY` environment variable.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        client_name: the name of the client to use for the API requests. Defaults to
            `"distilabel"`.
        _ChatMessage: the `ChatMessage` class from the `cohere` package.
        _aclient: the `AsyncClient` client from the `cohere` package.

    Runtime parameters:
        - `base_url`: the base URL to use for the Cohere API requests. Defaults to
            `"https://api.cohere.ai/v1"`.
        - `api_key`: the API key to authenticate the requests to the Cohere API. Defaults
            to the value of the `COHERE_API_KEY` environment variable.
        - `timeout`: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        - `client_name`: the name of the client to use for the API requests. Defaults to
            `"distilabel"`.
    """

    model: str
    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "COHERE_BASE_URL", "https://api.cohere.ai/v1"
        ),
        description="The base URL to use for the Cohere API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_COHERE_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Cohere API.",
    )
    timeout: RuntimeParameter[int] = Field(
        default=120,
        description="The maximum time in seconds to wait for a response from the API.",
    )
    client_name: RuntimeParameter[str] = Field(
        default="distilabel",
        description="The name of the client to use for the API requests.",
    )

    _ChatMessage: Type["ChatMessage"] = PrivateAttr(...)
    _aclient: "AsyncClient" = PrivateAttr(...)

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    def load(self) -> None:
        """Loads the `AsyncClient` client from the `cohere` package."""

        super().load()

        try:
            from cohere import AsyncClient, ChatMessage
        except ImportError as ie:
            raise ImportError(
                "The `cohere` package is required to use the `CohereLLM` class."
            ) from ie

        self._ChatMessage = ChatMessage

        self._aclient = AsyncClient(
            api_key=self.api_key.get_secret_value(),  # type: ignore
            client_name=self.client_name,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _format_chat_to_cohere(
        self, input: "ChatType"
    ) -> Tuple[Union[str, None], List["ChatMessage"], str]:
        """Formats the chat input to the Cohere Chat API conversational format.

        Args:
            input: The chat input to format.

        Returns:
            A tuple containing the system, chat history, and message.
        """
        system = None
        message = None
        chat_history = []
        for item in input:
            role = item["role"]
            content = item["content"]
            if role == "system":
                system = content
            elif role == "user":
                message = content
            elif role == "assistant":
                if message is None:
                    raise ValueError(
                        "An assistant message but be preceded by a user message."
                    )
                chat_history.append(self._ChatMessage(role="USER", message=message))  # type: ignore
                chat_history.append(self._ChatMessage(role="CHATBOT", message=content))
                message = None

        if message is None:
            raise ValueError("The chat input must end with a user message.")

        return system, chat_history, message

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        seed: Optional[float] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        raw_prompting: Optional[bool] = None,
    ) -> Union[str, None]:
        """Generates a response from the LLM given an input.

        Args:
            input: a single input in chat format to generate responses for.
            temperature: the temperature to use for the generation. Defaults to `None`.
            max_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `None`.
            k: the number of highest probability vocabulary tokens to keep for the generation.
                Defaults to `None`.
            p: the nucleus sampling probability to use for the generation. Defaults to
                `None`.
            seed: the seed to use for the generation. Defaults to `None`.
            stop_sequences: a list of sequences to use as stopping criteria for the generation.
                Defaults to `None`.
            frequency_penalty: the frequency penalty to use for the generation. Defaults
                to `None`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `None`.
            raw_prompting: a flag to use raw prompting for the generation. Defaults to
                `None`.

        Returns:
            The generated response from the Cohere API model.
        """
        system, chat_history, message = self._format_chat_to_cohere(input)

        response = await self._aclient.chat(  # type: ignore
            message=message,
            model=self.model,
            preamble=system,
            chat_history=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            k=k,
            p=p,
            seed=seed,
            stop_sequences=stop_sequences,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            raw_prompting=raw_prompting,
        )

        if (text := response.text) == "":
            self._logger.warning(
                f"Received no response using Cohere client (model: '{self.model}')."
                f" Finish reason was: {response.finish_reason}"
            )
            return None

        return text

    @override
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        """Method to generate a list of responses asynchronously, returning the output
        synchronously awaiting for the response of each input sent to `agenerate`."""

        async def agenerate(
            inputs: List["ChatType"], **kwargs: Any
        ) -> "GenerateOutput":
            """Internal function to parallelize the asynchronous generation of responses."""
            tasks = [
                asyncio.create_task(self.agenerate(input=input, **kwargs))
                for input in inputs
                for _ in range(num_generations)
            ]
            return await asyncio.gather(*tasks)

        outputs = self.event_loop.run_until_complete(agenerate(inputs, **kwargs))
        return list(grouper(outputs, n=num_generations, incomplete="ignore"))
