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

from distilabel.llm.together import TogetherLLM


class TestTogetherLLM:
    model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def test_together_llm(self) -> None:
        llm = TogetherLLM(model=self.model_id, api_key="api.key")  # type: ignore

        assert isinstance(llm, TogetherLLM)
        assert llm.model_name == self.model_id

    def test_together_llm_env_vars(self) -> None:
        os.environ["TOGETHER_API_KEY"] = "another.api.key"
        os.environ["TOGETHER_BASE_URL"] = "https://example.com"

        llm = TogetherLLM(model=self.model_id)

        assert isinstance(llm, TogetherLLM)
        assert llm.model_name == self.model_id
        assert llm.base_url == "https://example.com"
        assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

        del os.environ["TOGETHER_API_KEY"]
        del os.environ["TOGETHER_BASE_URL"]

    def test_serialization(self) -> None:
        os.environ["TOGETHER_API_KEY"] = "api.key"
        llm = TogetherLLM(model=self.model_id)

        _dump = {
            "model": self.model_id,
            "base_url": "https://api.together.xyz/v1",
            "type_info": {
                "module": "distilabel.llm.together",
                "name": "TogetherLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(TogetherLLM.from_dict(_dump), TogetherLLM)
