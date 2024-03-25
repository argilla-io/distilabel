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
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest

try:
    from distilabel.llm.mistral import MistralLLM
except ImportError:
    pass


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="`mistralai` requires Python 3.9 or higher"
)
@patch("mistralai.async_client.MistralAsyncClient")
class TestMistralLLM:
    def test_mistral_llm(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        assert isinstance(llm, MistralLLM)
        assert llm.model_name == "mistral-tiny"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        llm._aclient = mock_mistral

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat = AsyncMock(return_value=mocked_completion)

        await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_generate(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        llm._aclient = mock_mistral

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        llm.generate(
            inputs=[
                [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    },
                ]
            ]
        )

    def test_serialization(self, mock_mistral: MagicMock) -> None:
        os.environ["MISTRAL_API_KEY"] = "api.key"
        llm = MistralLLM(model="mistral-tiny")  # type: ignore

        _dump = {
            "model": "mistral-tiny",
            "endpoint": "https://api.mistral.ai",
            "generation_kwargs": {},
            "max_retries": 5,
            "timeout": 120,
            "max_concurrent_requests": 64,
            "type_info": {
                "module": "distilabel.llm.mistral",
                "name": "MistralLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(MistralLLM.from_dict(_dump), MistralLLM)
