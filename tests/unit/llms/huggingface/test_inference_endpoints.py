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

import random
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nest_asyncio
import pytest
from distilabel.llms.huggingface.inference_endpoints import InferenceEndpointsLLM

from ..utils import DummyUserDetail


@patch("huggingface_hub.AsyncInferenceClient")
@patch("openai.AsyncOpenAI")
class TestInferenceEndpointsLLM:
    def test_load_no_api_key(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )

        # Mock `huggingface_hub.constants.HF_TOKEN_PATH` to not exist
        with mock.patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with pytest.raises(
                ValueError,
                match="To use `InferenceEndpointsLLM` an API key must be provided",
            ):
                llm.load()

    def test_load_with_cached_token(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )

        # Mock `huggingface_hub.constants.HF_TOKEN_PATH` to exist
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "builtins.open", new_callable=mock.mock_open, read_data="hf_token"
        ):
            # Should not raise any errors
            llm.load()

    def test_serverless_inference_endpoints_llm(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )

        assert isinstance(llm, InferenceEndpointsLLM)
        assert llm.model_name == "distilabel-internal-testing/tiny-random-mistral"

    def test_dedicated_inference_endpoints_llm(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            endpoint_name="tiny-random-mistral",
            endpoint_namespace="distilabel-internal-testing",
        )

        assert isinstance(llm, InferenceEndpointsLLM)
        assert llm.model_name == "tiny-random-mistral"

    def test_dedicated_inference_endpoints_llm_via_url(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            base_url="https://api-inference.huggingface.co/models/distilabel-internal-testing/tiny-random-mistral"
        )

        assert isinstance(llm, InferenceEndpointsLLM)
        assert (
            llm.model_name
            == "https://api-inference.huggingface.co/models/distilabel-internal-testing/tiny-random-mistral"
        )

    @pytest.mark.asyncio
    async def test_agenerate_via_inference_client(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )
        llm._aclient = mock_inference_client

        llm._aclient.text_generation = AsyncMock(
            return_value=" Aenean hendrerit aliquam velit. ..."
        )

        assert await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        ) == [" Aenean hendrerit aliquam velit. ..."]

    @pytest.mark.asyncio
    async def test_agenerate_via_openai_client(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            use_openai_client=True,
        )
        llm._aclient = mock_openai_client

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
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
        ) == [" Aenean hendrerit aliquam velit. ..."]

    @pytest.mark.asyncio
    async def test_generate_via_inference_client(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral"
        )
        llm._aclient = mock_inference_client

        llm._aclient.text_generation = AsyncMock(
            return_value=" Aenean hendrerit aliquam velit. ..."
        )

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
        ) == [(" Aenean hendrerit aliquam velit. ...",)]

    @pytest.mark.asyncio
    async def test_generate_via_openai_client(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            use_openai_client=True,
        )
        llm._aclient = mock_openai_client

        mocked_completion = Mock(
            choices=[Mock(message=Mock(content=" Aenean hendrerit aliquam velit. ..."))]
        )
        llm._aclient.chat.completions.create = AsyncMock(return_value=mocked_completion)

        ...
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
        ) == [(" Aenean hendrerit aliquam velit. ...",)]

    @pytest.mark.asyncio
    async def test_agenerate_with_grammar(
        self, mock_inference_client: MagicMock, _: MagicMock
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
            grammar={"type": "regex", "value": r"\b[A-Z][a-z]*\b"},
        )
        llm._aclient = mock_inference_client

        llm._aclient.text_generation = AsyncMock(
            return_value=" Aenean hendrerit aliquam velit. ..."
        )

        # Since there's a pseudo-random number within the generation kwargs, we set the seed
        # here first to ensure reproducibility within the tests
        random.seed(42)

        assert await llm.agenerate(
            input=[
                {
                    "role": "user",
                    "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                },
            ]
        ) == [" Aenean hendrerit aliquam velit. ..."]

        kwargs = {
            "prompt": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "max_new_tokens": 128,
            "do_sample": False,
            "typical_p": None,
            "repetition_penalty": None,
            "temperature": 1.0,
            "top_p": None,
            "top_k": None,
            "stop_sequences": None,
            "return_full_text": False,
            "watermark": False,
            "grammar": {"type": "regex", "value": "\\b[A-Z][a-z]*\\b"},
            "seed": 478163327,  # pre-computed random value with `random.seed(42)`
        }
        mock_inference_client.text_generation.assert_called_with(**kwargs)

    def test_serialization(
        self,
        mock_inference_client: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        llm = InferenceEndpointsLLM(
            model_id="distilabel-internal-testing/tiny-random-mistral",
        )

        _dump = {
            "model_id": "distilabel-internal-testing/tiny-random-mistral",
            "endpoint_name": None,
            "endpoint_namespace": None,
            "base_url": None,
            "tokenizer_id": None,
            "generation_kwargs": {},
            "grammar": None,
            "model_display_name": None,
            "use_openai_client": False,
            "structured_output": None,
            "type_info": {
                "module": "distilabel.llms.huggingface.inference_endpoints",
                "name": "InferenceEndpointsLLM",
            },
        }
        assert llm.dump() == _dump
        assert isinstance(InferenceEndpointsLLM.from_dict(_dump), InferenceEndpointsLLM)
