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

from .utils import DummyUserDetail

try:
    from distilabel.llms.mistral import MistralLLM
except ImportError:
    MistralLLM = None


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="`mistralai` requires Python 3.9 or higher"
)
@patch("mistralai.async_client.MistralAsyncClient")
class TestMistralLLM:
    def test_mistral_llm(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        assert isinstance(llm, MistralLLM)  # type: ignore
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
    async def test_agenerate_structured(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(
            model="mistral-tiny",
            api_key="api.key",
            structured_output={
                "schema": DummyUserDetail,
                "mode": "tool_call",
                "max_retries": 1,
            },
        )  # type: ignore
        llm._aclient = mock_mistral

        sample_user = DummyUserDetail(name="John Doe", age=30)

        llm._aclient.chat.completions.create = AsyncMock(return_value=sample_user)
        # This should work just with the _aclient.chat method once it's fixed in instructor, and
        # then in our code.
        # llm._aclient.chat = AsyncMock(return_value=sample_user)

        generation = await llm.agenerate(
            input=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert generation[0] == sample_user.model_dump_json()

    @pytest.mark.asyncio
    async def test_generate(self, mock_mistral: MagicMock) -> None:
        llm = MistralLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        llm._aclient = mock_mistral

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat = Mock(
            complete_async=AsyncMock(return_value=mocked_completion)
        )

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

    @pytest.mark.parametrize(
        "structured_output, dump",
        [
            (
                None,
                {
                    "model": "mistral-tiny",
                    "endpoint": "https://api.mistral.ai",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "timeout": 120,
                    "max_concurrent_requests": 64,
                    "structured_output": None,
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.llms.mistral",
                        "name": "MistralLLM",
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
                    "model": "mistral-tiny",
                    "endpoint": "https://api.mistral.ai",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "timeout": 120,
                    "max_concurrent_requests": 64,
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "mode": "tool_call",
                        "max_retries": 1,
                    },
                    "jobs_ids": None,
                    "offline_batch_generation_block_until_done": None,
                    "use_offline_batch_generation": False,
                    "type_info": {
                        "module": "distilabel.llms.mistral",
                        "name": "MistralLLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self, _: MagicMock, structured_output: Dict[str, Any], dump: Dict[str, Any]
    ) -> None:
        os.environ["MISTRAL_API_KEY"] = "api.key"
        llm = MistralLLM(model="mistral-tiny")  # type: ignore

        _dump = {
            "model": "mistral-tiny",
            "endpoint": "https://api.mistral.ai",
            "generation_kwargs": {},
            "max_retries": 6,
            "timeout": 120,
            "max_concurrent_requests": 64,
            "structured_output": None,
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.llms.mistral",
                "name": "MistralLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(MistralLLM.from_dict(_dump), MistralLLM)  # type: ignore
