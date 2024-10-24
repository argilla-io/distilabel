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

from distilabel.models.llms.litellm import LiteLLM


@pytest.fixture(params=["mistral/mistral-tiny", "gpt-4"])
def model(request) -> list:
    return request.param


@patch("litellm.acompletion")
class TestLiteLLM:
    def test_litellm_llm(self, _: MagicMock, model: str) -> None:
        llm = LiteLLM(model=model)  # type: ignore
        assert isinstance(llm, LiteLLM)
        assert llm.model_name == model

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_litellm: MagicMock, model: str) -> None:
        llm = LiteLLM(model=model)  # type: ignore
        llm._aclient = mock_litellm

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient = AsyncMock(return_value=mocked_completion)

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
    async def test_generate(self, mock_litellm: MagicMock, model: str) -> None:
        llm = LiteLLM(model=model)  # type: ignore
        llm._aclient = mock_litellm

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient = AsyncMock(return_value=mocked_completion)

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

    def test_serialization(self, _: MagicMock, model: str) -> None:
        llm = LiteLLM(model=model)  # type: ignore

        _dump = {
            "model": model,
            "verbose": False,
            "structured_output": None,
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.llms.litellm",
                "name": "LiteLLM",
            },
            "generation_kwargs": {},
        }

        assert llm.dump() == _dump
        assert isinstance(LiteLLM.from_dict(_dump), LiteLLM)
