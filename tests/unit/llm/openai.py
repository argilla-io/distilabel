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

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest
from distilabel.llm.openai import OpenAILLM


@patch("openai.AsyncOpenAI")
class TestOpenAILLM:
    def test_openai_llm(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model="gpt-4", api_key="api.key")  # type: ignore
        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model="gpt-4", api_key="api.key")  # type: ignore
        llm._aclient = mock_openai

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

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
    async def test_generate(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model="gpt-4", api_key="api.key")  # type: ignore
        llm._aclient = mock_openai

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

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

    def test_serialization(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model="gpt-4", api_key="api.key")  # type: ignore

        _dump = {
            "model": "gpt-4",
            "_type_info_": {
                "module": "distilabel.llm.openai",
                "name": "OpenAILLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(OpenAILLM.from_dict(_dump), OpenAILLM)
