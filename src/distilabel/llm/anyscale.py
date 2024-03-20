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


from distilabel.llm.openai import OpenAILLM


class AnyscaleLLM(OpenAILLM):
    """
    Anyscale LLM implementation running the async API client of OpenAI because of duplicate API behavior.

    Attributes:
        model: the model name to use for the LLM, e.g., `google/gemma-7b-it`. [Supported models](https://docs.endpoints.anyscale.com/text-generation/supported-models/google-gemma-7b-it).
        base_url: the base URL to use for the Anyscale API can be set with `OPENAI_BASE_URL`. Default is "https://api.endpoints.anyscale.com/v1".
        api_key: the API key to authenticate the requests to the Anyscale API. Can be set with `OPENAI_API_KEY`. Default is `None`.
    """

    base_url: str = "https://api.endpoints.anyscale.com/v1"
