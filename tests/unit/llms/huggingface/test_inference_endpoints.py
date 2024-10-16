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
import random
from typing import Generator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import nest_asyncio
import pytest
from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionOutputUsage,
)

from distilabel.llms.huggingface.inference_endpoints import InferenceEndpointsLLM


@pytest.fixture(autouse=True)
def mock_hf_token_env_variable() -> Generator[None, None, None]:
    with patch.dict(os.environ, {"HF_TOKEN": "hf_token"}):
        yield


@patch("huggingface_hub.AsyncInferenceClient")
class TestInferenceEndpointsLLM:
    def test_no_tokenizer_magpie_raise_value_error(
        self, mock_inference_client: MagicMock
    ) -> None:
        with pytest.raises(
            ValueError,
            match="`use_magpie_template` cannot be `True` if `tokenizer_id` is `None`",
        ):
            InferenceEndpointsLLM(
                base_url="http://localhost:8000",
                use_magpie_template=True,
                magpie_pre_query_template="llama3",
            )

    def test_tokenizer_id_set_if_model_id_and_structured_output(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            structured_output={"format": "regex", "schema": r"\b[A-Z][a-z]*\b"},
        )

        assert llm.tokenizer_id == llm.model_id

    def test_load_no_api_key(self, mock_inference_client: MagicMock) -> None:
        del os.environ["HF_TOKEN"]

        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            structured_output={"format": "regex", "schema": r"\b[A-Z][a-z]*\b"},
        )

        assert llm.tokenizer_id == llm.model_id

    def test_load_with_cached_token(self, mock_inference_client: MagicMock) -> None:
        llm = InferenceEndpointsLLM(base_url="http://localhost:8000")

        # Mock `huggingface_hub.constants.HF_TOKEN_PATH` to exist
        with (
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch(
                "builtins.open", new_callable=mock.mock_open, read_data="hf_token"
            ),
        ):
            # Should not raise any errors
            llm.load()

    def test_serverless_inference_endpoints_llm(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )

        assert isinstance(llm, InferenceEndpointsLLM)
        assert llm.model_name == "distilabel-internal-testing/tiny-random-mistral"

    def test_dedicated_inference_endpoints_llm(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            endpoint_name="tiny-random-mistral",
            endpoint_namespace="distilabel-internal-testing",
        )

        assert isinstance(llm, InferenceEndpointsLLM)
        assert llm.model_name == "tiny-random-mistral"

    def test_dedicated_inference_endpoints_llm_via_url(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            base_url="https://api-inference.huggingface.co/models/distilabel-internal-testing/tiny-random-mistral"
        )
        llm.load()

        assert isinstance(llm, InferenceEndpointsLLM)
        assert (
            llm.model_name
            == "https://api-inference.huggingface.co/models/distilabel-internal-testing/tiny-random-mistral"
        )

    @pytest.mark.asyncio
    async def test_agenerate_with_text_generation(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            tokenizer_id="distilabel-internal-testing/tiny-random-mistral",
        )
        llm.load()

        llm._aclient.text_generation = AsyncMock(
            return_value=MagicMock(
                generated_text="Aenean hendrerit aliquam velit...",
                details=MagicMock(
                    generated_tokens=66,
                ),
            )
        )

        result = await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": ["Aenean hendrerit aliquam velit..."],
            "statistics": {
                "input_tokens": [0],
                "output_tokens": [66],
            },
        }

    @pytest.mark.asyncio
    async def test_agenerate_with_chat_completion(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
        )
        llm.load()

        llm._aclient.chat_completion = AsyncMock(  # type: ignore
            return_value=ChatCompletionOutput(  # type: ignore
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="length",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=" Aenean hendrerit aliquam velit. ...",
                        ),
                    )
                ],
                created=1721045246,
                id="",
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                system_fingerprint="2.1.1-dev0-sha-4327210",
                usage=ChatCompletionOutputUsage(
                    completion_tokens=66, prompt_tokens=18, total_tokens=84
                ),
            )
        )

        result = await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": [" Aenean hendrerit aliquam velit. ..."],
            "statistics": {
                "input_tokens": [18],
                "output_tokens": [66],
            },
        }

    @pytest.mark.asyncio
    async def test_agenerate_with_chat_completion_fails(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
        )
        llm.load()

        llm._aclient.chat_completion = AsyncMock(  # type: ignore
            return_value=ChatCompletionOutput(  # type: ignore
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="eos_token",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                        ),
                    )
                ],
                created=1721045246,
                id="",
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                system_fingerprint="2.1.1-dev0-sha-4327210",
                usage=ChatCompletionOutputUsage(
                    completion_tokens=66, prompt_tokens=18, total_tokens=84
                ),
            )
        )

        result = await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": [None],
            "statistics": {
                "input_tokens": [18],
                "output_tokens": [66],
            },
        }

    @pytest.mark.asyncio
    async def test_generate(self, mock_inference_client: MagicMock) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            # tokenizer_id="distilabel-internal-testing/tiny-random-mistral",
        )
        llm.load()

        llm._aclient.chat_completion = AsyncMock(  # type: ignore
            return_value=ChatCompletionOutput(  # type: ignore
                choices=[
                    ChatCompletionOutputComplete(
                        finish_reason="eos_token",
                        index=0,
                        message=ChatCompletionOutputMessage(
                            role="assistant",
                            content=None,
                        ),
                    )
                ],
                created=1721045246,
                id="",
                model="meta-llama/Meta-Llama-3-70B-Instruct",
                system_fingerprint="2.1.1-dev0-sha-4327210",
                usage=ChatCompletionOutputUsage(
                    completion_tokens=66, prompt_tokens=18, total_tokens=84
                ),
            )
        )

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
                "generations": [None],
                "statistics": {
                    "input_tokens": [18],
                    "output_tokens": [66],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_agenerate_with_structured_output(
        self, mock_inference_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            tokenizer_id="distilabel-internal-testing/tiny-random-mistral",
            structured_output={"format": "regex", "schema": r"\b[A-Z][a-z]*\b"},
        )
        llm.load()

        llm._aclient.text_generation = AsyncMock(
            return_value=MagicMock(
                generated_text="Aenean hendrerit aliquam velit...",
                details=MagicMock(
                    generated_tokens=66,
                ),
            )
        )
        # Since there's a pseudo-random number within the generation kwargs, we set the seed
        # here first to ensure reproducibility within the tests
        random.seed(42)

        result = await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        )
        assert result == {
            "generations": ["Aenean hendrerit aliquam velit..."],
            "statistics": {
                "input_tokens": [0],
                "output_tokens": [66],
            },
        }

    def test_serialization(self, mock_inference_client: MagicMock) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            tokenizer_id="distilabel-internal-testing/tiny-random-mistral",
        )

        _dump = {
            "model_id": "distilabel-internal-testing/tiny-random-mistral",
            "endpoint_name": None,
            "endpoint_namespace": None,
            "base_url": None,
            "tokenizer_id": "distilabel-internal-testing/tiny-random-mistral",
            "generation_kwargs": {},
            "magpie_pre_query_template": None,
            "structured_output": None,
            "model_display_name": None,
            "use_magpie_template": False,
            "jobs_ids": None,
            "offline_batch_generation_block_until_done": None,
            "use_offline_batch_generation": False,
            "type_info": {
                "module": "distilabel.llms.huggingface.inference_endpoints",
                "name": "InferenceEndpointsLLM",
            },
        }
        assert llm.dump() == _dump
        assert isinstance(InferenceEndpointsLLM.from_dict(_dump), InferenceEndpointsLLM)
