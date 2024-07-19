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

from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import FormattedInput, InstructorStructuredOutputType

if TYPE_CHECKING:
    from openai import AsyncOpenAI


_01ai_API_KEY_ENV_VAR_NAME = "01AI_API_KEY"


class OneAI(AsyncLLM):
    """WIP"""

    model: str
    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "01AI_BASE_URL", "https://api.openai.com/v1"
        ),
        description="The base URL to use for the 01AI API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_01ai_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the 01ai API.",
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
    structured_output: Optional[RuntimeParameter[InstructorStructuredOutputType]] = (
        Field(
            default=None,
            description="The structured output format to use across all the generations.",
        )
    )

    _api_key_env_var: str = PrivateAttr(_01ai_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncOpenAI"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""
        super().load()

        try:
            from openai import AsyncOpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed. Please install it using"
                " `pip install openai`."
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )

        self._aclient = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
        )

        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")  # type: ignore
            if structured_output := result.get("structured_output"):
                self.structured_output = structured_output

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[str] = None,
    ) -> GenerateOutput:
        """Generates `num_generations` responses for the given input using the OpenAI async
        client.

        Args:
            input: a single input in chat format to generate responses for.
            num_generations: the number of generations to create per input. Defaults to
                `1`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            stop: a string or a list of strings to use as a stop sequence for the generation.
                Defaults to `None`.
            response_format: the format of the response to return. Must be one of
                "text" or "json". Read the documentation [here](https://platform.openai.com/docs/guides/text-generation/json-mode)
                for more information on how to use the JSON model from OpenAI. Defaults to `text`.

        Note:
            If response_format

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """

        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,
                client=self._aclient,
                framework="openai",
            )
            self._aclient = result.get("client")

        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,  # type: ignore
            "model": self.model,
            "max_tokens": max_new_tokens,
            "n": num_generations,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "timeout": 50,
        }

        if response_format is not None:
            if response_format not in ["text", "json", "json_object"]:
                raise ValueError(
                    f"Invalid response format '{response_format}'. Must be either 'text'"
                    " or 'json'."
                )

            if response_format == "json":
                response_format = "json_object"

            kwargs["response_format"] = response_format

        if structured_output:
            kwargs = self._prepare_kwargs(kwargs, structured_output)

        generations = []
        completion = await self._aclient.chat.completions.create(**kwargs)  # type: ignore

        if structured_output:
            generations.append(completion.model_dump_json())
            return generations

        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(  # type: ignore
                    f"Received no response using OpenAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations
