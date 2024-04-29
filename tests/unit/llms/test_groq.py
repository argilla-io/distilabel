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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest
from distilabel.llms.groq import GroqLLM


@patch("groq._client.AsyncGroq")
class TestGroqLLM:
    def test_mistral_llm(self, mock_groq: MagicMock) -> None:
        llm = GroqLLM(model="llama3-70b-8192", api_key="api.key")  # type: ignore
        assert isinstance(llm, GroqLLM)  # type: ignore
        assert llm.model_name == "llama3-70b-8192"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_groq: MagicMock) -> None:
        llm = GroqLLM(model="llama3-70b-8192", api_key="api.key")  # type: ignore
        llm._aclient = mock_groq

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

        assert await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        ) == [" Aenean hendrerit aliquam velit. ..."]

    @pytest.mark.asyncio
    async def test_generate(self, mock_groq: MagicMock) -> None:
        llm = GroqLLM(model="llama3-70b-8192", api_key="api.key")  # type: ignore
        llm._aclient = mock_groq

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        assert llm.generate(
            inputs=[
                [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    },
                ]
            ]
        ) == [(" Aenean hendrerit aliquam velit. ...",)]

    def test_serialization(self, mock_groq: MagicMock) -> None:
        os.environ["GROQ_API_KEY"] = "api.key"
        llm = GroqLLM(model="llama3-70b-8192")

        _dump = {
            "model": "llama3-70b-8192",
            "base_url": "https://api.groq.com",
            "generation_kwargs": {},
            "max_retries": 2,
            "timeout": 120,
            "type_info": {
                "module": "distilabel.llms.groq",
                "name": "GroqLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(GroqLLM.from_dict(_dump), GroqLLM)  # type: ignore
