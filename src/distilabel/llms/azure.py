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

from pydantic import Field, PrivateAttr, SecretStr
from typing_extensions import override

from distilabel.llms.openai import OpenAILLM
from distilabel.mixins.runtime_parameters import RuntimeParameter

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI

_AZURE_OPENAI_ENDPOINT_ENV_VAR_NAME = "AZURE_OPENAI_ENDPOINT"
_AZURE_OPENAI_API_KEY_ENV_VAR_NAME = "AZURE_OPENAI_API_KEY"


class AzureOpenAILLM(OpenAILLM):
    """Azure OpenAI LLM implementation running the async API client of OpenAI because of
    duplicate API behavior, but with Azure-specific parameters.

    Attributes:
        model: the model name to use for the LLM i.e. the name of the Azure deployment.
        base_url: the base URL to use for the Azure OpenAI API can be set with `AZURE_OPENAI_ENDPOINT`.
            Defaults to `None` which means that the value set for the environment variable
            `AZURE_OPENAI_ENDPOINT` will be used, or `None` if not set.
        api_key: the API key to authenticate the requests to the Azure OpenAI API. Defaults to `None`
            which means that the value set for the environment variable `AZURE_OPENAI_API_KEY` will be
            used, or `None` if not set.
        api_version: the API version to use for the Azure OpenAI API. Defaults to `None` which means
            that the value set for the environment variable `OPENAI_API_VERSION` will be used, or
            `None` if not set.
    """

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(_AZURE_OPENAI_ENDPOINT_ENV_VAR_NAME),
        description="The base URL to use for the Azure OpenAI API requests i.e. the Azure OpenAI endpoint.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_AZURE_OPENAI_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Azure OpenAI API.",
    )

    api_version: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_VERSION"),
        description="The API version to use for the Azure OpenAI API.",
    )

    _base_url_env_var: str = PrivateAttr(_AZURE_OPENAI_ENDPOINT_ENV_VAR_NAME)
    _api_key_env_var: str = PrivateAttr(_AZURE_OPENAI_API_KEY_ENV_VAR_NAME)
    _aclient: Optional["AsyncAzureOpenAI"] = PrivateAttr(...)  # type: ignore

    @override
    def load(self) -> None:
        """Loads the `AsyncAzureOpenAI` client to benefit from async requests."""
        super().load()

        try:
            from openai import AsyncAzureOpenAI
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

        # TODO: May be worth adding the AD auth too? Also the `organization`?
        self._aclient = AsyncAzureOpenAI(  # type: ignore
            azure_endpoint=self.base_url,  # type: ignore
            azure_deployment=self.model,
            api_version=self.api_version,
            api_key=self.api_key.get_secret_value(),
            max_retries=self.max_retries,  # type: ignore
            timeout=self.timeout,
        )
