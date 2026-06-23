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
from typing import Optional

from pydantic import Field, PrivateAttr, SecretStr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.openai import OpenAILLM

_OPENROUTER_API_KEY_ENV_VAR_NAME = "OPENROUTER_API_KEY"


class OpenRouterLLM(OpenAILLM):
    """OpenRouter LLM implementation running the async API client of OpenAI.

    Attributes:
        model: the model name to use for the LLM, e.g., `google/gemma-7b-it`. See the
            supported models under the "Models" section
            [here](https://openrouter.ai/models?fmt=cards&supported_parameters=tools).
        base_url: the base URL to use for the OpenRouter API requests. Defaults to `None`, which
            means that the value set for the environment variable `OPENROUTER_BASE_URL` will be used, or
            "https://api.endpoints.anyscale.com/v1" if not set.
        api_key: the API key to authenticate the requests to the OpenRouter API. Defaults to `None` which
            means that the value set for the environment variable `OPENROUTER_API_KEY` will be used, or
            `None` if not set.
        _api_key_env_var: the name of the environment variable to use for the API key.
            It is meant to be used internally.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import OpenRouterLLM

        llm = OpenRouterLLM(model="google/gemma-7b-it", api_key="api.key")

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        description="The base URL to use for the OpenRouter API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_OPENROUTER_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the OpenRouter API.",
    )

    _api_key_env_var: str = PrivateAttr(_OPENROUTER_API_KEY_ENV_VAR_NAME)
