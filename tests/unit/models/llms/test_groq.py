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

from distilabel.models.llms.groq import GroqLLM

from .utils import DummyUserDetail


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
            choices=[
                Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))
            ],
            usage=Mock(prompt_tokens=100, completion_tokens=100),
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
        ) == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {"input_tokens": [100], "output_tokens": [100]},
        }

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="`groq` requires Python 3.9 or higher"
    )
    @pytest.mark.asyncio
    async def test_agenerate_structured(self, mock_openai: MagicMock) -> None:
        llm = GroqLLM(
            model="llama3-70b-8192",
            api_key="api.key",
            structured_output={
                "schema": DummyUserDetail,
                "mode": "tool_call",
                "max_retries": 1,
            },
        )  # type: ignore
        llm._aclient = mock_openai

        mocked_usage = MagicMock(
            usage=MagicMock(prompt_tokens=100, completion_tokens=100),
        )
        sample_user = DummyUserDetail(
            name="John Doe", age=30, _raw_response=mocked_usage
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=sample_user)

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
            "statistics": {"input_tokens": [100], "output_tokens": [100]},
        }

    @pytest.mark.asyncio
    async def test_generate(self, mock_groq: MagicMock) -> None:
        llm = GroqLLM(model="llama3-70b-8192", api_key="api.key")  # type: ignore
        llm._aclient = mock_groq

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content="Aenean hendrerit aliquam velit..."))],
            usage=Mock(prompt_tokens=100, completion_tokens=100),
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
        ) == [
            {
                "generations": ["Aenean hendrerit aliquam velit..."],
                "statistics": {"input_tokens": [100], "output_tokens": [100]},
            }
        ]

    @pytest.mark.parametrize(
        "structured_output, dump",
        [
            (
                None,
                {
                    "model": "llama3-70b-8192",
                    "base_url": "https://api.groq.com",
                    "generation_kwargs": {},
                    "max_retries": 2,
                    "timeout": 120,
                    "structured_output": None,
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.models.llms.groq",
                        "name": "GroqLLM",
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
                    "model": "llama3-70b-8192",
                    "base_url": "https://api.groq.com",
                    "generation_kwargs": {},
                    "max_retries": 2,
                    "timeout": 120,
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "mode": "tool_call",
                        "max_retries": 1,
                    },
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.models.llms.groq",
                        "name": "GroqLLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self, _: MagicMock, structured_output: Dict[str, Any], dump: Dict[str, Any]
    ) -> None:
        os.environ["GROQ_API_KEY"] = "api.key"
        llm = GroqLLM(model="llama3-70b-8192", structured_output=structured_output)

        assert llm.dump() == dump
        assert isinstance(GroqLLM.from_dict(dump), GroqLLM)  # type: ignore
