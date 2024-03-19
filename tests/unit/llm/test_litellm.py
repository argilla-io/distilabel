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
from distilabel.llm.litellm import LitellmLLM


@patch("litellm.acompletion")
class TestOpenAILLM:
    def test_litellm_llm(self, mock_litellm: MagicMock) -> None:
        llm = LitellmLLM(model="gpt-4", api_key="api.key")  # type: ignore
        assert isinstance(llm, LitellmLLM)
        assert llm.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_litellm: MagicMock) -> None:
        llm = LitellmLLM(model="gpt-4", api_key="api.key")  # type: ignore
        llm._aclient = mock_litellm

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
    async def test_generate(self, mock_litellm: MagicMock) -> None:
        llm = LitellmLLM(model="gpt-4", api_key="api.key")  # type: ignore
        llm._aclient = mock_litellm

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

    def test_serialization(self, mock_litellm: MagicMock) -> None:
        os.environ["OPENAI_API_KEY"] = "api.key"
        llm = LitellmLLM(model="gpt-4")  # type: ignore

        _dump = {
            "model": "gpt-4",
            "type_info": {
                "module": "distilabel.llm.litellm",
                "name": "LitellmLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(LitellmLLM.from_dict(_dump), LitellmLLM)
