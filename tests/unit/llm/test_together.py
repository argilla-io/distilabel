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
from unittest.mock import MagicMock, patch

from distilabel.llm.together import TogetherLLM


@patch("openai.AsyncOpenAI")
class TestTogetherLLM:
    def test_llm(self, mock_openai: MagicMock) -> None:
        llm = TogetherLLM(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="api.key"
        )  # type: ignore
        assert isinstance(llm, TogetherLLM)
        assert llm.model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def test_serialization(self, mock_openai: MagicMock) -> None:
        os.environ["TOGETHER_API_KEY"] = "api.key"
        llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")  # type: ignore

        _dump = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "base_url": "https://api.together.xyz/v1",
            "type_info": {
                "module": "distilabel.llm.together",
                "name": "TogetherLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(TogetherLLM.from_dict(_dump), TogetherLLM)
