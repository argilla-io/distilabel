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

_CHEAPER_INFERENCE_API_KEY_ENV_VAR_NAME = "CHEAPER_INFERENCE_API_KEY"


class CheaperInferenceLLM(OpenAILLM):
    """Cheaper Inference LLM implementation running the async API client of OpenAI.

    Attributes:
        model: the model name to use for the LLM, e.g., `gpt-5.4-mini`. See the
            supported models [here](https://cheaperinference.com/docs).
        base_url: the base URL to use for the Cheaper Inference API requests. Defaults to `None`,
            which means that the value set for the environment variable `CHEAPER_INFERENCE_BASE_URL`
            will be used, or "https://api.cheaperinference.com/v1" if not set.
        api_key: the API key to authenticate the requests to the Cheaper Inference API. Defaults to
            `None` which means that the value set for the environment variable
            `CHEAPER_INFERENCE_API_KEY` will be used, or `None` if not set.
        _api_key_env_var: the name of the environment variable to use for the API key.
            It is meant to be used internally.

    Examples:
        Generate text:

        ```python
        from distilabel.models.llms import CheaperInferenceLLM

        llm = CheaperInferenceLLM(model="gpt-5.4-mini", api_key="api.key")

        llm.load()

        output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
        ```
    """

    base_url: Optional[RuntimeParameter[str]] = Field(
        default_factory=lambda: os.getenv(
            "CHEAPER_INFERENCE_BASE_URL", "https://api.cheaperinference.com/v1"
        ),
        description="The base URL to use for the Cheaper Inference API requests.",
    )
    api_key: Optional[RuntimeParameter[SecretStr]] = Field(
        default_factory=lambda: os.getenv(_CHEAPER_INFERENCE_API_KEY_ENV_VAR_NAME),
        description="The API key to authenticate the requests to the Cheaper Inference API.",
    )

    _api_key_env_var: str = PrivateAttr(_CHEAPER_INFERENCE_API_KEY_ENV_VAR_NAME)
