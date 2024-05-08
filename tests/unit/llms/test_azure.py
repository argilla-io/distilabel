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
from unittest import mock

from distilabel.llms.azure import AzureOpenAILLM


class TestAzureOpenAILLM:
    model_id: str = "gpt-4"
    base_url: str = "https://example-resource.azure.openai.com/"
    api_version: str = "preview"

    def test_azure_openai_llm(self) -> None:
        llm = AzureOpenAILLM(
            model=self.model_id,
            base_url=self.base_url,
            api_key="api.key",  # type: ignore
            api_version=self.api_version,
        )

        assert isinstance(llm, AzureOpenAILLM)
        assert llm.model_name == self.model_id
        assert llm.base_url == self.base_url
        assert llm.api_key.get_secret_value() == "api.key"  # type: ignore
        assert llm.api_version == self.api_version

    def test_azure_openai_llm_env_vars(self) -> None:
        from distilabel.llms.azure import (
            _AZURE_OPENAI_API_KEY_ENV_VAR_NAME,
            _AZURE_OPENAI_ENDPOINT_ENV_VAR_NAME,
        )

        with mock.patch.dict(os.environ, clear=True):
            os.environ[_AZURE_OPENAI_API_KEY_ENV_VAR_NAME] = "another.api.key"
            os.environ[_AZURE_OPENAI_ENDPOINT_ENV_VAR_NAME] = self.base_url
            os.environ["OPENAI_API_VERSION"] = self.api_version

            llm = AzureOpenAILLM(model=self.model_id)

            assert isinstance(llm, AzureOpenAILLM)
            assert llm.model_name == self.model_id
            assert llm.base_url == self.base_url
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore
            assert llm.api_version == self.api_version

    def test_serialization(self) -> None:
        llm = AzureOpenAILLM(
            model=self.model_id, base_url=self.base_url, api_version=self.api_version
        )

        _dump = {
            "generation_kwargs": {},
            "model": "gpt-4",
            "base_url": "https://example-resource.azure.openai.com/",
            "max_retries": 6,
            "timeout": 120,
            "api_version": "preview",
            "structured_output": None,
            "type_info": {"module": "distilabel.llms.azure", "name": "AzureOpenAILLM"},
        }
        assert llm.dump() == _dump
        assert isinstance(AzureOpenAILLM.from_dict(_dump), AzureOpenAILLM)
