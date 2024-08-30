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

from distilabel.llms.openai import OpenAILLM
from distilabel.mixins.runtime_parameters import RuntimeParameter

_TOGETHER_API_KEY_ENV_VAR_NAME = "TOGETHER_API_KEY"


class TogetherLLM(OpenAILLM):
    """TogetherLLM LLM implementation running the async API client of OpenAI.

    Attributes:
        model: the model name to use for the LLM e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1".
            Supported models can be found [here](https://api.together.xyz/models).
        base_url: the base URL to use for the Together API can be set with `TOGETHER_BASE_URL`.
            Defaults to `None` which means that the value set for the environment variable
            `TOGETHER_BASE_URL` will be used, or "https://api.together.xyz/v1" if not set.
        api_key: the API key to authenticate the requests to the Together API. Defaults to `None`
            which means that the value set for the environment variable `TOGETHER_API_KEY` will be
            used, or `None` if not set.
        _api_key_env_var: the name of the environment variable to use for the API key. It
            is meant to be used internally.

    Examples:
        Generate text:

        ```python
        from distilabel.llms import AnyscaleLLM

        llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="api.key")

        llm.load()

        output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "TOGETHER_BASE_URL", "https://api.together.xyz/v1"
        ),
        description="The base URL to use for the Together API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_TOGETHER_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Together API.",
    )

    _api_key_env_var: str = PrivateAttr(_TOGETHER_API_KEY_ENV_VAR_NAME)
