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

from typing import Optional

from distilabel.llm.openai import OpenAILLM


class TogetherLLM(OpenAILLM):
    """
    TogetherLLM LLM implementation running the async API client of OpenAI because of duplicate API behavior.

    Attributes:
        model: the model name to use for the LLM e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1". Supported models can be found [here](https://api.together.xyz/models).
        base_url: the base URL to use for the Together API can be set with `OPENAI_BASE_URL`. Default is "https://api.together.xyz/v1".
        api_key: the API key to authenticate the requests to the Together API. Can be set with `TOGETHER_API_KEY`. Default is `None`.
    """

    base_url: Optional[str] = "https://api.together.xyz/v1"

    def load(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        """Loads the `AsyncOpenAI` client to benefit from async requests."""

        try:
            from openai import AsyncOpenAI
        except ImportError as ie:
            raise ImportError(
                "OpenAI Python client is not installed which is required for `TogetherLLM`. Please install it using"
                " `pip install openai`."
            ) from ie

        self.api_key = self._handle_api_key_value(
            self_value=self.api_key, load_value=api_key, env_var="TOGETHER_API_KEY"
        )
        self.base_url = base_url or self.base_url

        self._aclient = AsyncOpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
            max_retries=6,
        )
