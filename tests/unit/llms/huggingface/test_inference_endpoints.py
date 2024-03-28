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
from distilabel.llms.huggingface.inference_endpoints import InferenceEndpointsLLM


@patch("huggingface_hub.AsyncInferenceClient")
@patch("openai.AsyncOpenAI")
class TestInferenceEndpointsLLM:
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

    def test_serialization(
        self, mock_inference_client: MagicMock, mock_openai_client: MagicMock
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
            "model_display_name": None,
            "use_openai_client": False,
            "type_info": {
                "module": "distilabel.llms.huggingface.inference_endpoints",
                "name": "InferenceEndpointsLLM",
            },
        }
        assert llm.dump() == _dump
        assert isinstance(InferenceEndpointsLLM.from_dict(_dump), InferenceEndpointsLLM)
