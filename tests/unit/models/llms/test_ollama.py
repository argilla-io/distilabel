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

from unittest.mock import AsyncMock, MagicMock, patch

import nest_asyncio
import pytest

from distilabel.models.llms.ollama import OllamaLLM


@patch("ollama.AsyncClient")
class TestOllamaLLM:
    def test_no_tokenizer_magpie_raise_value_error(self, _: MagicMock) -> None:
        with pytest.raises(
            ValueError,
            match="`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`",
        ):
            OllamaLLM(
                model="llama3.1",
                use_magpie_template=True,
                magpie_pre_query_template="llama3",
            )

    def test_ollama_llm(self, _: MagicMock) -> None:
        llm = OllamaLLM(model="notus")  # type: ignore
        assert isinstance(llm, OllamaLLM)
        assert llm.model_name == "notus"

    @pytest.mark.asyncio
    async def test_agenerate(self, mock_ollama: MagicMock) -> None:
        llm = OllamaLLM(model="notus")  # type: ignore
        llm._aclient = mock_ollama

        mocked_completion = {
            "message": {"content": "Aenean hendrerit aliquam velit..."},
            "prompt_eval_count": 10,
            "eval_count": 10,
        }
        llm._aclient.chat = AsyncMock(return_value=mocked_completion)

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
            "statistics": {"input_tokens": [10], "output_tokens": [10]},
        }

    @pytest.mark.asyncio
    async def test_generate(self, mock_ollama: MagicMock) -> None:
        llm = OllamaLLM(model="notus")  # type: ignore
        llm._aclient = mock_ollama

        mocked_completion = {
            "message": {"content": "Aenean hendrerit aliquam velit..."},
            "prompt_eval_count": 10,
            "eval_count": 10,
        }

        llm._aclient.chat = AsyncMock(return_value=mocked_completion)

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
                "statistics": {"input_tokens": [10], "output_tokens": [10]},
            }
        ]

    def test_serialization(self, _: MagicMock) -> None:
        llm = OllamaLLM(model="notus")  # type: ignore

        _dump = {
            "model": "notus",
            "host": None,
            "timeout": 120,
            "follow_redirects": True,
            "generation_kwargs": {},
            "structured_output": None,
            "jobs_ids": None,
            "magpie_pre_query_template": None,
            "tokenizer_id": None,
            "use_magpie_template": False,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.models.llms.ollama",
                "name": "OllamaLLM",
            },
        }

        assert llm.dump() == _dump
        assert isinstance(OllamaLLM.from_dict(_dump), OllamaLLM)
