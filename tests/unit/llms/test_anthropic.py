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
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest

from distilabel.llms.anthropic import AnthropicLLM

from .utils import DummyUserDetail


@patch("anthropic.AsyncAnthropic")
class TestAnthropicLLM:
    def test_anthropic_llm(self, _: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore
        assert isinstance(llm, AnthropicLLM)
        assert llm.model_name == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_anthropic: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229", api_key="api.key")  # type: ignore
        llm._aclient = mock_anthropic

        mocked_completion = Mock(
            content=[Mock(text="Aenean hendrerit aliquam velit...")],
            usage=Mock(input_tokens=100, output_tokens=100),
        )

        llm._aclient.messages.create = AsyncMock(return_value=mocked_completion)

        result = await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": ["Aenean hendrerit aliquam velit..."],
            "statistics": {"input_tokens": 100, "output_tokens": 100},
        }

    @pytest.mark.asyncio
    async def test_agenerate_structured(self, mock_openai: MagicMock) -> None:
        llm = AnthropicLLM(
            model="claude-3-opus-20240229",
            api_key="api.key",
            structured_output={
                "schema": DummyUserDetail,
                "mode": "tool_call",
                "max_retries": 1,
            },
        )  # type: ignore
        llm._aclient = mock_openai

        mocked_usage = MagicMock(
            usage=MagicMock(input_tokens=100, output_tokens=100),
        )
        sample_user = DummyUserDetail(
            name="John Doe", age=30, _raw_response=mocked_usage
        )
        llm._aclient.messages.create = AsyncMock(return_value=sample_user)

        generation = await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert generation == {
            "generations": [sample_user.model_dump_json()],
            "statistics": {
                "input_tokens": 100,
                "output_tokens": 100,
            },
        }

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="`mistralai` requires Python 3.9 or higher"
    )
    @pytest.mark.asyncio
    async def test_generate(self, mock_anthropic: MagicMock) -> None:
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore
        llm._aclient = mock_anthropic

        mocked_completion = Mock(
            content=[Mock(text="Aenean hendrerit aliquam velit...")],
            usage=Mock(input_tokens=100, output_tokens=100),
        )

        llm._aclient.messages.create = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        result = llm.generate(
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
        assert result == [
            {
                "generations": ["Aenean hendrerit aliquam velit..."],
                "statistics": {"input_tokens": 100, "output_tokens": 100},
            }
        ]

    @pytest.mark.parametrize(
        "structured_output, dump",
        [
            (
                None,
                {
                    "base_url": "https://api.anthropic.com",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "model": "claude-3-opus-20240229",
                    "timeout": 600.0,
                    "structured_output": None,
                    "type_info": {
                        "module": "distilabel.llms.anthropic",
                        "name": "AnthropicLLM",
                    },
                },
            ),
            (
                {
                    "schema": DummyUserDetail.model_json_schema(),
                    "mode": "tool_call",
                    "max_retries": 1,
                },
                {
                    "base_url": "https://api.anthropic.com",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "model": "claude-3-opus-20240229",
                    "timeout": 600.0,
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "mode": "tool_call",
                        "max_retries": 1,
                    },
                    "type_info": {
                        "module": "distilabel.llms.anthropic",
                        "name": "AnthropicLLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self, _: MagicMock, structured_output: Dict[str, Any], dump: Dict[str, Any]
    ) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "api.key"
        llm = AnthropicLLM(model="claude-3-opus-20240229")  # type: ignore

        _dump = {
            "base_url": "https://api.anthropic.com",
            "generation_kwargs": {},
            "max_retries": 6,
            "model": "claude-3-opus-20240229",
            "timeout": 600.0,
            "structured_output": None,
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.llms.anthropic",
                "name": "AnthropicLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(AnthropicLLM.from_dict(_dump), AnthropicLLM)
