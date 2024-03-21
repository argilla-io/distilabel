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

from distilabel.llm.anyscale import AnyscaleLLM

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


class TestAnyscaleLLM:
    def test_anyscale_llm(self) -> None:
        llm = AnyscaleLLM(model=MODEL, api_key="api.key")
        assert isinstance(llm, AnyscaleLLM)
        assert llm.model_name == MODEL

    def test_serialization(self) -> None:
        os.environ["OPENAI_API_KEY"] = "api.key"
        llm = AnyscaleLLM(model=MODEL)

        _dump = {
            "model": MODEL,
            "base_url": "https://api.endpoints.anyscale.com/v1",
            "type_info": {
                "module": "distilabel.llm.anyscale",
                "name": "AnyscaleLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(AnyscaleLLM.from_dict(_dump), AnyscaleLLM)
