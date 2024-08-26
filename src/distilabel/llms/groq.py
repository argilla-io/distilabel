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
from typing import TYPE_CHECKING, Optional

from pydantic import Field, PrivateAttr, SecretStr, validate_call

from distilabel.llms.base import AsyncLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.steps.base import RuntimeParameter
from distilabel.steps.tasks.typing import (
    FormattedInput,
    InstructorStructuredOutputType,
)

if TYPE_CHECKING:
    from groq import AsyncGroq


_GROQ_API_BASE_URL_ENV_VAR_NAME = "GROQ_BASE_URL"
_GROQ_API_KEY_ENV_VAR_NAME = "GROQ_API_KEY"


class GroqLLM(AsyncLLM):
    """Groq API implementation using the async client for concurrent text generation.

    Attributes:
        model: the name of the model from the Groq API to use for the generation.
        base_url: the base URL to use for the Groq API requests. Defaults to
            `"https://api.groq.com"`.
        api_key: the API key to authenticate the requests to the Groq API. Defaults to
            the value of the `GROQ_API_KEY` environment variable.
        max_retries: the maximum number of times to retry the request to the API before
            failing. Defaults to `2`.
        timeout: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.
        structured_output: a dictionary containing the structured output configuration configuration
            using `instructor`. You can take a look at the dictionary structure in
            `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.
        _api_key_env_var: the name of the environment variable to use for the API key.
        _aclient: the `AsyncGroq` client from the `groq` package.

    Runtime parameters:
        - `base_url`: the base URL to use for the Groq API requests. Defaults to
            `"https://api.groq.com"`.
        - `api_key`: the API key to authenticate the requests to the Groq API. Defaults to
            the value of the `GROQ_API_KEY` environment variable.
        - `max_retries`: the maximum number of times to retry the request to the API before
            failing. Defaults to `2`.
        - `timeout`: the maximum time in seconds to wait for a response from the API. Defaults
            to `120`.

    Examples:
        Generate text:

        ```python
        from distilabel.llms import GroqLLM

        llm = GroqLLM(model="llama3-70b-8192")

        llm.load()

        # Call the model
        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

        Generate structured data:

        ```python
        from pydantic import BaseModel
        from distilabel.llms import GroqLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = GroqLLM(
            model="llama3-70b-8192",
            api_key="api.key",
            structured_output={"schema": User}
        )

        llm.load()

        output = llm.generate(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
    """

    model: str

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            _GROQ_API_BASE_URL_ENV_VAR_NAME, "https://api.groq.com"
        ),
        description="The base URL to use for the Groq API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_GROQ_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Groq API.",
    )
    max_retries: RuntimeParameter[int] = Field(
        default=2,
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

    _num_generations_param_supported = False

    _api_key_env_var: str = PrivateAttr(_GROQ_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncGroq"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `AsyncGroq` client to benefit from async requests."""
        super().load()

        try:
            from groq import AsyncGroq
        except ImportError as ie:
            raise ImportError(
                "Groq Python client is not installed. Please install it using"
                ' `pip install groq` or from the extras as `pip install "distilabel[groq]"`.'
            ) from ie

        if self.api_key is None:
            raise ValueError(
                f"To use `{self.__class__.__name__}` an API key must be provided via `api_key`"
                f" attribute or runtime parameter, or set the environment variable `{self._api_key_env_var}`."
            )

        self._aclient = AsyncGroq(
            base_url=self.base_url,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
        )

        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="groq",
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
        seed: Optional[int] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[str] = None,
    ) -> "GenerateOutput":
        """Generates `num_generations` responses for the given input using the Groq async
        client.

        Args:
            input: a single input in chat format to generate responses for.
            seed: the seed to use for the generation. Defaults to `None`.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            temperature: the temperature to use for the generation. Defaults to `0.1`.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            stop: the stop sequence to use for the generation. Defaults to `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.

        References:
            - https://console.groq.com/docs/text-chat
        """
        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,
                client=self._aclient,
                framework="groq",
            )
            self._aclient = result.get("client")

        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,  # type: ignore
            "model": self.model,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "stream": False,
            "stop": stop,
        }
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
                    f"Received no response using the Groq client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return generations
