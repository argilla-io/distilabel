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
from unittest import mock

import nest_asyncio
import pytest
from distilabel.llms.cohere import CohereLLM


@mock.patch("cohere.AsyncClient")
class TestCohereLLM:
    def test_cohere_llm(self, _: mock.MagicMock) -> None:
        llm = CohereLLM(model="command-r", api_key="api.key")

        assert isinstance(llm, CohereLLM)
        assert llm.model_name == "command-r"

    def test_cohere_llm_env_vars(self, _: mock.MagicMock) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["COHERE_API_KEY"] = "another.api.key"
            os.environ["COHERE_BASE_URL"] = "https://example.com"

            llm = CohereLLM(model="command-r")

            assert isinstance(llm, CohereLLM)
            assert llm.model_name == "command-r"
            assert llm.base_url == "https://example.com"
            assert llm.api_key.get_secret_value() == "another.api.key"  # type: ignore

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_async_client: mock.MagicMock) -> None:
        llm = CohereLLM(model="command-r")
        llm._aclient = mock_async_client  # type: ignore

        mocked_completion = mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content=" Aenean hendrerit aliquam velit. ...")
                )
            ]
        )
        llm._aclient.chat = mock.AsyncMock(return_value=mocked_completion)

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
    async def test_generate(self, mock_async_client: mock.MagicMock) -> None:
        llm = CohereLLM(model="command-r")
        llm._aclient = mock_async_client  # type: ignore

        mocked_completion = mock.Mock(
            choices=[
                mock.Mock(
                    message=mock.Mock(content=" Aenean hendrerit aliquam velit. ...")
                )
            ]
        )
        llm._aclient.chat = mock.AsyncMock(return_value=mocked_completion)

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

    def test_serialization(self, _: mock.MagicMock) -> None:
        llm = CohereLLM(model="command-r")

        dump = {
            "model": "command-r",
            "generation_kwargs": {},
            "base_url": "https://api.cohere.ai/v1",
            "timeout": 120,
            "client_name": "distilabel",
            "structured_output": None,
            "type_info": {
                "module": "distilabel.llms.cohere",
                "name": "CohereLLM",
            },
        }

        assert llm.dump() == dump
        assert isinstance(CohereLLM.from_dict(dump), CohereLLM)
