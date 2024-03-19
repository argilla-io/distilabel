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
from distilabel.llm.anthropic import AnthropicLLM


@patch("anthropic.AsyncAnthropic")
class TestAnthropicLLM:
    def test_anthropic_llm(self, mock_anthropic: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore
        assert isinstance(llm, AnthropicLLM)
        assert llm.model_name == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_anthropic: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229", api_key="api.key")  # type: ignore
        llm._aclient = mock_anthropic

        mocked_completion = Mock()
        mocked_completion.content = [Mock(text="Aenean hendrerit aliquam velit...")]

        llm._aclient.messages.create = AsyncMock(return_value=mocked_completion)

        await llm.agenerate(
            system="",
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_generate(self, mock_anthropic: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore
        llm._aclient = mock_anthropic

        mocked_completion = Mock()
        mocked_completion.content = [Mock(text="Aenean hendrerit aliquam velit...")]

        llm._aclient.messages.create = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        llm.generate(
            system="",
            inputs=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ],
        )

    def test_serialization(self, mock_anthropic: MagicMock) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "api.key"
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore

        _dump = {
            "base_url": "https://api.anthropic.com",
            "http_client": None,
            "max_retries": 2,
            "model": "claude-3-opus-20240229",
            "timeout": 600.0,
            "type_info": {
                "module": "distilabel.llm.anthropic",
                "name": "AnthropicLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(AnthropicLLM.from_dict(_dump), AnthropicLLM)
