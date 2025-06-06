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

from distilabel.models.llms.openrouter import OpenRouterLLM


class TestOpenRouterLLM:
    model_id: str = "deepseek/deepseek-chat-v3-0324:free"

    def test_anyscale_llm(self) -> None:
        llm = OpenRouterLLM(model=self.model_id, api_key="api.key")  # type: ignore

        assert isinstance(llm, OpenRouterLLM)
        assert llm.model_name == self.model_id

    def test_anyscale_llm_env_vars(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["OPENROUTER_API_KEY"] = "another.api.key"
            os.environ["OPENROUTER_BASE_URL"] = "https://example.com"

            llm = OpenRouterLLM(model=self.model_id)

            assert isinstance(llm, OpenRouterLLM)
            assert llm.model_name == self.model_id
            assert llm.base_url == "https://example.com"
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

    def test_serialization(self) -> None:
        llm = OpenRouterLLM(model=self.model_id)

        _dump = {
            "model": self.model_id,
            "generation_kwargs": {},
            "max_retries": 6,
            "default_headers": None,
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": 120,
            "structured_output": None,
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.models.llms.openrouter",
                "name": "OpenRouterLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(OpenRouterLLM.from_dict(_dump), OpenRouterLLM)
