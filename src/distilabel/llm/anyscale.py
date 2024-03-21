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

from pydantic import PrivateAttr, SecretStr

from distilabel.llm.openai import OpenAILLM


class AnyscaleLLM(OpenAILLM):
    """Anyscale LLM implementation running the async API client of OpenAI because of
    duplicate API behavior.

    Attributes:
        model: the model name to use for the LLM, e.g., `google/gemma-7b-it`. See the
            supported models under the "Text Generation -> Supported Models" section
            [here](https://docs.endpoints.anyscale.com/).
        base_url: the base URL to use for the Anyscale API can be set with `ANYSCALE_BASE_URL`.
            Defaults to the value set for the environment variable `ANYSCALE_BASE_URL`, or
            "https://api.endpoints.anyscale.com/v1" if not set.
        api_key: the API key to authenticate the requests to the Anyscale API. Defaults to the
            value set for the environment variable `ANYSCALE_API_KEY`, or `None` if not set.

    """

    base_url: Optional[str] = os.getenv(
        "ANYSCALE_BASE_URL", "https://api.endpoints.anyscale.com/v1"
    )
    api_key: Optional[SecretStr] = os.getenv("ANYSCALE_API_KEY", None)  # type: ignore

    _env_var: Optional[str] = PrivateAttr(default="ANYSCALE_API_KEY")
