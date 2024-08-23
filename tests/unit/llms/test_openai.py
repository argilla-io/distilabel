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
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest

from distilabel.llms.openai import OpenAILLM

from .utils import DummyUserDetail


@patch("openai.AsyncOpenAI")
class TestOpenAILLM:
    model_id: str = "gpt-4"

    def test_openai_llm(self, _: MagicMock) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore

        assert isinstance(llm, OpenAILLM)
        assert llm.model_name == self.model_id

    def test_openai_llm_env_vars(self, _: MagicMock) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["OPENAI_API_KEY"] = "another.api.key"
            os.environ["OPENAI_BASE_URL"] = "https://example.com"

            llm = OpenAILLM(model=self.model_id)

            assert isinstance(llm, OpenAILLM)
            assert llm.model_name == self.model_id
            assert llm.base_url == "https://example.com"
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
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
    async def test_agenerate_structured(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(
            model=self.model_id,
            api_key="api.key",
            structured_output={
                "schema": DummyUserDetail,
                "mode": "tool_call",
                "max_retries": 1,
            },
        )  # type: ignore
        llm._aclient = mock_openai

        sample_user = DummyUserDetail(name="John Doe", age=30)

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
        assert generation[0] == sample_user.model_dump_json()

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="`mistralai` requires Python 3.9 or higher"
    )
    @pytest.mark.asyncio
    async def test_generate(self, mock_openai: MagicMock) -> None:
        llm = OpenAILLM(model=self.model_id, api_key="api.key")  # type: ignore
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

        with pytest.raises(ValueError):
            llm.generate(
                inputs=[
                    [
                        {"role": "system", "content": ""},
                        {
                            "role": "user",
                            "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                        },
                    ]
                ],
                response_format="unkown_format",
            )

    @pytest.mark.parametrize(
        "structured_output, dump",
        [
            (
                None,
                {
                    "model": "gpt-4",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 120,
                    "structured_output": None,
                    "type_info": {
                        "module": "distilabel.llms.openai",
                        "name": "OpenAILLM",
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
                    "model": "gpt-4",
                    "generation_kwargs": {},
                    "max_retries": 6,
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 120,
                    "structured_output": {
                        "schema": DummyUserDetail.model_json_schema(),
                        "mode": "tool_call",
                        "max_retries": 1,
                    },
                    "type_info": {
                        "module": "distilabel.llms.openai",
                        "name": "OpenAILLM",
                    },
                },
            ),
        ],
    )
    def test_serialization(
        self, _: MagicMock, structured_output: Dict[str, Any], dump: Dict[str, Any]
    ) -> None:
        llm = OpenAILLM(model=self.model_id, structured_output=structured_output)

        assert llm.dump() == dump
        assert isinstance(OpenAILLM.from_dict(dump), OpenAILLM)
