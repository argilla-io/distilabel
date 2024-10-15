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
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.typing import (
    FormattedInput,
    InstructorStructuredOutputType,
)

if TYPE_CHECKING:
    from mistralai import Mistral


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
        structured_output: a dictionary containing the structured output configuration configuration
            using `instructor`. You can take a look at the dictionary structure in
            `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.
        _api_key_env_var: the name of the environment variable to use for the API key. It is meant to
            be used internally.
        _aclient: the `Mistral` to use for the Mistral API. It is meant to be used internally.
            Set in the `load` method.

    Runtime parameters:
        - `api_key`: the API key to authenticate the requests to the Mistral API.
        - `max_retries`: the maximum number of retries to attempt when a request fails.
            Defaults to `5`.
        - `timeout`: the maximum time in seconds to wait for a response. Defaults to `120`.
        - `max_concurrent_requests`: the maximum number of concurrent requests to send.
            Defaults to `64`.

    Examples:
        Generate text:

        ```python
        from distilabel.llms import MistralLLM

        llm = MistralLLM(model="open-mixtral-8x22b")

        llm.load()

        # Call the model
        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

        Generate structured data:

        ```python
        from pydantic import BaseModel
        from distilabel.llms import MistralLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = MistralLLM(
            model="open-mixtral-8x22b",
            api_key="api.key",
            structured_output={"schema": User}
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
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
    structured_output: Optional[RuntimeParameter[InstructorStructuredOutputType]] = (
        Field(
            default=None,
            description="The structured output format to use across all the generations.",
        )
    )

    _num_generations_param_supported = False

    _api_key_env_var: str = PrivateAttr(_MISTRALAI_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["Mistral"] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the `Mistral` client to benefit from async requests."""
        super().load()

        try:
            from mistralai import Mistral
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

        self._aclient = Mistral(
            api_key=self.api_key.get_secret_value(),
            endpoint=self.endpoint,
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,  # type: ignore
            max_concurrent_requests=self.max_concurrent_requests,  # type: ignore
        )

        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="mistral",
            )
            self._aclient = result.get("client")  # type: ignore
            if structured_output := result.get("structured_output"):
                self.structured_output = structured_output

    @property
    def model_name(self) -> str:
        """Returns the model name used for the LLM."""
        return self.model

    # TODO: add `num_generations` parameter once Mistral client allows `n` parameter
    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: FormattedInput,
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
        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,
                client=self._aclient,
                framework="mistral",
            )
            self._aclient = result.get("client")

        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,  # type: ignore
            "model": self.model,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        generations = []
        if structured_output:
            kwargs = self._prepare_kwargs(kwargs, structured_output)
            # TODO: This should work just with the _aclient.chat method, but it's not working.
            # We need to check instructor and see if we can create a PR.
            completion = await self._aclient.chat.completions.create(**kwargs)  # type: ignore
        else:
            # completion = await self._aclient.chat(**kwargs)  # type: ignore
            completion = await self._aclient.chat.complete_async(**kwargs)  # type: ignore

        if structured_output:
            raw_response = completion._raw_response
            return {
                "generations": [completion.model_dump_json()],
                "statistics": {
                    "input_tokens": raw_response.usage.prompt_tokens,
                    "output_tokens": raw_response.usage.completion_tokens,
                },
            }

        for choice in completion.choices:
            if (content := choice.message.content) is None:
                self._logger.warning(  # type: ignore
                    f"Received no response using MistralAI client (model: '{self.model}')."
                    f" Finish reason was: {choice.finish_reason}"
                )
            generations.append(content)
        return {
            "generations": generations,
            "statistics": {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            },
        }
