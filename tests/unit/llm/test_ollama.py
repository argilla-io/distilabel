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

try:
    from distilabel.llm.ollama import OllamalLLM
except ImportError:
    pass


@patch("ollama.AsyncClient")
class TestOllamaLLM:
    def test_ollama_llm(self, mock_ollama: MagicMock) -> None:
        llm = OllamalLLM(model="notus")  # type: ignore
        assert isinstance(llm, OllamalLLM)
        assert llm.model_name == "notus"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_ollama: MagicMock) -> None:
        llm = OllamalLLM(model="notus")  # type: ignore
        llm._aclient = mock_ollama

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat = AsyncMock(return_value=mocked_completion)

        await llm.agenerate(
            inputs=[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_generate(self, mock_ollama: MagicMock) -> None:
        llm = OllamalLLM(model="mistral-tiny", api_key="api.key")  # type: ignore
        llm._aclient = mock_ollama

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

    def test_serialization(self, mock_ollama: MagicMock) -> None:
        llm = OllamalLLM(model="notus")  # type: ignore

        _dump = {
            "model": "notus",
            "host": None,
            "timeout": 120,
            "follow_redirects": True,
            "type_info": {
                "module": "distilabel.llm.ollama",
                "name": "OllamalLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(OllamalLLM.from_dict(_dump), OllamalLLM)
