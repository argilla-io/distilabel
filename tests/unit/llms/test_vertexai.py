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
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    Part,
)

from distilabel.llms.vertexai import VertexAILLM


@patch("vertexai.generative_models.GenerativeModel.generate_content_async")
class TestVertexAILLM:
    def test_openai_llm(self, _: MagicMock) -> None:
        llm = VertexAILLM(model="gemini-1.0-pro")
        assert isinstance(llm, VertexAILLM)
        assert llm.model_name == "gemini-1.0-pro"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_generative_model: MagicMock) -> None:
        llm = VertexAILLM(model="gemini-1.0-pro")
        llm._aclient = mock_generative_model
        llm._part_class = Part  # type: ignore
        llm._content_class = Content  # type: ignore
        llm._generation_config_class = GenerationConfig

        mocked_completion = Mock(
            candidates=[Mock(text=" Aenean hendrerit aliquam velit. ...")],
            usage_metadata=Mock(prompt_token_count=10, candidates_token_count=10),
        )
        llm._aclient.generate_content_async = AsyncMock(return_value=mocked_completion)

        with pytest.raises(
            ValueError, match="`VertexAILLM only supports the roles 'user' or 'model'."
        ):
            await llm.agenerate(
                input=[
                    {"role": "system", "content": ""},
                    {
                        "role": "test",
                        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    },
                ]
            )

        result = await llm.agenerate(
            input=[
                {"role": "model", "content": ""},
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {"input_tokens": [10], "output_tokens": [10]},
        }

    @pytest.mark.asyncio
    async def test_generate(self, mock_generative_model: MagicMock) -> None:
        llm = VertexAILLM(model="gemini-1.0-pro")
        llm._aclient = mock_generative_model
        llm._part_class = Part  # type: ignore
        llm._content_class = Content  # type: ignore
        llm._generation_config_class = GenerationConfig

        mocked_completion = Mock(
            candidates=[Mock(text=" Aenean hendrerit aliquam velit. ...")],
            usage_metadata=Mock(prompt_token_count=10, candidates_token_count=10),
        )
        llm._aclient.generate_content_async = AsyncMock(return_value=mocked_completion)

        nest_asyncio.apply()

        with pytest.raises(
            ValueError, match="`VertexAILLM only supports the roles 'user' or 'model'."
        ):
            llm.generate(
                inputs=[
                    [
                        {"role": "system", "content": ""},
                        {
                            "role": "test",
                            "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                        },
                    ],
                ]
            )

        result = llm.generate(
            inputs=[
                [
                    {"role": "model", "content": "I am a model."},
                    {
                        "role": "user",
                        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    },
                ]
            ]
        )
        assert result == [
            {
                "generations": [" Aenean hendrerit aliquam velit. ..."],
                "statistics": {"input_tokens": [10], "output_tokens": [10]},
            }
        ]

    def test_serialization(self, _: MagicMock) -> None:
        llm = VertexAILLM(model="gemini-1.0-pro")

        _dump = {
            "model": "gemini-1.0-pro",
            "generation_kwargs": {},
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.llms.vertexai",
                "name": "VertexAILLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(VertexAILLM.from_dict(_dump), VertexAILLM)
