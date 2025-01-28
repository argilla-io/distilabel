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
from typing import (
    TYPE_CHECKING,
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

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.base import AsyncLLM
from distilabel.models.llms.utils import prepare_output
from distilabel.typing import (
    FormattedInput,
    GenerateOutput,
    InstructorStructuredOutputType,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    from pydantic import BaseModel

    from distilabel.typing import LLMStatistics


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
        structured_output: a dictionary containing the structured output configuration configuration
            using `instructor`. You can take a look at the dictionary structure in
            `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.
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

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import AnthropicLLM

        llm = AnthropicLLM(model="claude-3-opus-20240229", api_key="api.key")

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```

        Generate structured data:

        ```python
        from pydantic import BaseModel
        from distilabel.models.llms import AnthropicLLM

        class User(BaseModel):
            name: str
            last_name: str
            id: int

        llm = AnthropicLLM(
            model="claude-3-opus-20240229",
            api_key="api.key",
            structured_output={"schema": User}
        )

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
        ```
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
    structured_output: Optional[RuntimeParameter[InstructorStructuredOutputType]] = (
        Field(
            default=None,
            description="The structured output format to use across all the generations.",
        )
    )

    _num_generations_param_supported = False

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
                " `pip install 'distilabel[anthropic]'`."
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
        if self.structured_output:
            result = self._prepare_structured_output(
                structured_output=self.structured_output,
                client=self._aclient,
                framework="anthropic",
            )
            self._aclient = result.get("client")
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

        structured_output = None
        if isinstance(input, tuple):
            input, structured_output = input
            result = self._prepare_structured_output(
                structured_output=structured_output,
                client=self._aclient,
                framework="anthropic",
            )
            self._aclient = result.get("client")

        if structured_output is None and self.structured_output is not None:
            structured_output = self.structured_output

        kwargs = {
            "messages": input,  # type: ignore
            "model": self.model,
            "system": (
                input.pop(0)["content"]
                if input and input[0]["role"] == "system"
                else NOT_GIVEN
            ),
            "max_tokens": max_tokens,
            "stream": False,
            "stop_sequences": NOT_GIVEN if stop_sequences is None else stop_sequences,
            "temperature": temperature,
            "top_p": NOT_GIVEN if top_p is None else top_p,
            "top_k": NOT_GIVEN if top_k is None else top_k,
        }

        if structured_output:
            kwargs = self._prepare_kwargs(kwargs, structured_output)

        completion: Union["Message", "BaseModel"] = await self._aclient.messages.create(
            **kwargs
        )  # type: ignore
        if structured_output:
            # raw_response = completion._raw_response
            return prepare_output(
                [completion.model_dump_json()],
                **self._get_llm_statistics(completion._raw_response),
            )

        if (content := completion.content[0].text) is None:
            self._logger.warning(
                f"Received no response using Anthropic client (model: '{self.model}')."
                f" Finish reason was: {completion.stop_reason}"
            )
        return prepare_output([content], **self._get_llm_statistics(completion))

    @staticmethod
    def _get_llm_statistics(completion: "Message") -> "LLMStatistics":
        return {
            "input_tokens": [completion.usage.input_tokens],
            "output_tokens": [completion.usage.output_tokens],
        }
