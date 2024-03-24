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

import importlib
import os
from typing import Generator
from unittest import mock

import pytest
from distilabel.llm import anyscale


@pytest.fixture
def reload_anyscale_module() -> Generator[None, None, None]:
    importlib.reload(anyscale)
    yield


@pytest.mark.usefixtures("reload_anyscale_module")
class TestAnyscaleLLM:
    model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def test_anyscale_llm(self) -> None:
        llm = anyscale.AnyscaleLLM(model=self.model_id, api_key="api.key")  # type: ignore

        assert isinstance(llm, anyscale.AnyscaleLLM)
        assert llm.model_name == self.model_id

    def test_anyscale_llm_env_vars(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["ANYSCALE_API_KEY"] = "another.api.key"
            os.environ["ANYSCALE_BASE_URL"] = "https://example.com"

            importlib.reload(anyscale)

            llm = anyscale.AnyscaleLLM(model=self.model_id)

            assert isinstance(llm, anyscale.AnyscaleLLM)
            assert llm.model_name == self.model_id
            assert llm.base_url == "https://example.com"
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

    def test_serialization(self) -> None:
        llm = anyscale.AnyscaleLLM(model=self.model_id)

        _dump = {
            "model": self.model_id,
            "generation_kwargs": {},
            "base_url": "https://api.endpoints.anyscale.com/v1",
            "type_info": {
                "module": "distilabel.llm.anyscale",
                "name": "AnyscaleLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(anyscale.AnyscaleLLM.from_dict(_dump), anyscale.AnyscaleLLM)
