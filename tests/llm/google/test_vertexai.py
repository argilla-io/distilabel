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

from typing import Generator
from unittest import mock

import pytest
from distilabel.llm.google.vertexai import VertexAIEndpointLLM, VertexAILLM
from distilabel.tasks.text_generation.base import TextGenerationTask
from vertexai.language_models._language_models import (
    MultiCandidateTextGenerationResponse,
    TextGenerationResponse,
)
from vertexai.preview.generative_models import GenerationResponse


@pytest.fixture
def generative_model_mock() -> Generator[None, None, None]:
    """Mocks the VertexAI `GenerativeModel` class, so no Google auth is needed."""
    with mock.patch("distilabel.llm.google.vertexai.GenerativeModel") as mock_obj:
        mock_obj.return_value = mock.MagicMock()
        yield


@pytest.fixture
def text_generation_model_mock() -> Generator[None, None, None]:
    """Mocks the VertexAI `TextGenerationModel.from_pretrained` method, so no Google auth
    is needed."""
    with mock.patch(
        "distilabel.llm.google.vertexai.TextGenerationModel.from_pretrained"
    ) as mock_obj:
        mock_obj.return_value = mock.MagicMock()
        yield


@pytest.fixture
def prediction_service_client_mock() -> Generator[None, None, None]:
    with mock.patch(
        "distilabel.llm.google.vertexai.PredictionServiceClient"
    ) as mock_obj:
        mock_obj.return_value = mock.MagicMock()
        yield


@pytest.mark.usefixtures("generative_model_mock")
class TestVertexAILLMGenerativeModel:
    def test_generate_contents(self) -> None:
        llm = VertexAILLM(task=TextGenerationTask())

        contents = llm._generate_contents(
            inputs=[
                {"input": "Gemini Pro is cool"},
                {"input": "but OSS LLMs are even cooler"},
            ]
        )

        assert contents == [
            [
                {"role": "user", "parts": [{"text": llm.task.system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood."}]},
                {"role": "user", "parts": [{"text": "Gemini Pro is cool"}]},
            ],
            [
                {"role": "user", "parts": [{"text": llm.task.system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood."}]},
                {"role": "user", "parts": [{"text": "but OSS LLMs are even cooler"}]},
            ],
        ]

    def test_generative_model_single_output(self) -> None:
        llm = VertexAILLM(task=TextGenerationTask())
        type(llm).model_name = mock.PropertyMock(return_value="gemini-pro")
        generation_response = mock.MagicMock(spec=GenerationResponse)
        generation_response.text = "Yes, I'm cool"
        llm._call_generative_model_with_backoff = mock.MagicMock(
            return_value=generation_response
        )

        contents = [
            {
                "role": "user",
                "parts": [{"text": "You're an educated AI assistant."}],
            },
            {"role": "model", "parts": [{"text": "Understood."}]},
            {"role": "user", "parts": [{"text": "Gemini Pro is cool"}]},
        ]

        llm_output = llm._generative_model_single_output(contents)

        assert llm_output == {
            "model_name": llm.model_name,
            "parsed_output": {"generations": "Yes, I'm cool"},
            "prompt_used": contents,
            "raw_output": "Yes, I'm cool",
        }

    def test_generative_model_single_output_raise_value_error(self) -> None:
        llm = VertexAILLM(task=TextGenerationTask())
        type(llm).model_name = mock.PropertyMock(return_value="gemini-pro")
        generation_response = mock.MagicMock(spec=GenerationResponse)
        type(generation_response).text = mock.PropertyMock(
            side_effect=ValueError("Content has no parts.")
        )
        llm._call_generative_model_with_backoff = mock.MagicMock(
            return_value=generation_response
        )
        contents = [
            {
                "role": "user",
                "parts": [{"text": "You're an educated AI assistant."}],
            },
            {"role": "model", "parts": [{"text": "Understood."}]},
            {"role": "user", "parts": [{"text": "Gemini Pro is cool"}]},
            {"role": "user", "parts": [{"text": "Gemini Pro is cool"}]},
        ]

        llm_output = llm._generative_model_single_output(contents)

        assert llm_output == {
            "model_name": llm.model_name,
            "prompt_used": contents,
            "raw_output": None,
            "parsed_output": None,
        }

    def test_generative_model_single_output_raise_exception(self) -> None:
        llm = VertexAILLM(task=TextGenerationTask())
        generation_response = mock.MagicMock(spec=GenerationResponse)
        generation_response.text = "Yes, I'm cool"
        llm.task.parse_output = mock.MagicMock(
            side_effect=Exception("Could not parse output")
        )
        llm._call_generative_model_with_backoff = mock.MagicMock(
            return_value=generation_response
        )
        contents = [
            {
                "role": "user",
                "parts": [{"text": "You're an educated AI assistant."}],
            },
            {"role": "model", "parts": [{"text": "Understood."}]},
            {"role": "user", "parts": [{"text": "Gemini Pro is cool"}]},
        ]

        llm_output = llm._generative_model_single_output(contents)

        assert llm_output == {
            "model_name": llm.model_name,
            "prompt_used": contents,
            "raw_output": "Yes, I'm cool",
            "parsed_output": None,
        }


@pytest.mark.usefixtures("text_generation_model_mock")
class TestVertexAILLMTextGenerationModel:
    def test_text_generation_model_single_output(self) -> None:
        llm = VertexAILLM(model="text-bison", task=TextGenerationTask())

        # mock calling the text generation model
        text_generation_response = mock.MagicMock(spec=TextGenerationResponse)
        text_generation_response.text = "Yes, it's cool"
        response = mock.MagicMock(spec=MultiCandidateTextGenerationResponse)
        type(response).candidates = mock.PropertyMock(
            return_value=[text_generation_response]
        )
        llm._call_text_generation_model = mock.MagicMock(return_value=response)

        llm_output = llm._text_generation_model_single_output(
            prompt="Gemini Pro is cool", num_generations=1
        )

        assert llm_output == [
            {
                "model_name": llm.model_name,
                "prompt_used": "Gemini Pro is cool",
                "raw_output": "Yes, it's cool",
                "parsed_output": {"generations": "Yes, it's cool"},
            }
        ]

    def test_text_generation_model_single_output_raise_exception(self) -> None:
        llm = VertexAILLM(model="text-bison", task=TextGenerationTask())

        # mock calling the text generation model
        text_generation_response = mock.MagicMock(spec=TextGenerationResponse)
        text_generation_response.text = "Yes, it's cool"
        response = mock.MagicMock(spec=MultiCandidateTextGenerationResponse)
        type(response).candidates = mock.PropertyMock(
            return_value=[text_generation_response]
        )

        llm._call_text_generation_model = mock.MagicMock(return_value=response)

        # simulate an exception when parsing the output
        llm.task.parse_output = mock.MagicMock(
            side_effect=Exception("Could not parse output")
        )

        llm_output = llm._text_generation_model_single_output(
            prompt="Gemini Pro is cool", num_generations=1
        )

        assert llm_output == [
            {
                "model_name": llm.model_name,
                "prompt_used": "Gemini Pro is cool",
                "raw_output": "Yes, it's cool",
                "parsed_output": None,
            }
        ]


@pytest.mark.usefixtures("prediction_service_client_mock")
class TestVertexAIEndpointLLM:
    def test_prepare_instances(self) -> None:
        llm = VertexAIEndpointLLM(
            task=TextGenerationTask(),
            endpoint_id="12345678",
            project="unit-test",
            location="us-central1",
            generation_kwargs={
                "temperature": 1.0,
                "max_tokens": 512,
                "top_p": 1.0,
                "top_k": 10,
            },
        )

        instances = llm._prepare_instances(
            ["Gemini Pro is cool", "but OSS LLMs are even cooler"], num_generations=3
        )

        assert len(instances) == 2

        assert instances[0].struct_value["prompt"] == "Gemini Pro is cool"
        assert instances[0].struct_value["n"] == 3
        assert instances[0].struct_value["temperature"] == 1.0
        assert instances[0].struct_value["max_tokens"] == 512
        assert instances[0].struct_value["top_p"] == 1.0
        assert instances[0].struct_value["top_k"] == 10

    def test_prepare_instances_with_custom_arguments(self) -> None:
        llm = VertexAIEndpointLLM(
            task=TextGenerationTask(),
            endpoint_id="12345678",
            project="unit-test",
            location="us-central1",
            generation_kwargs={
                "temperature": 1.0,
                "max_tokens": 512,
                "top_p": 1.0,
                "top_k": 10,
            },
            prompt_argument="PROMPT",
            num_generations_argument="N_GENERATIONS",
        )

        instances = llm._prepare_instances(["Gemini Pro is cool"], num_generations=3)

        assert instances[0].struct_value["PROMPT"] == "Gemini Pro is cool"
        assert instances[0].struct_value["N_GENERATIONS"] == 3

    def test_single_output(self) -> None:
        llm = VertexAIEndpointLLM(
            task=TextGenerationTask(),
            endpoint_id="12345678",
            project="unit-test",
            location="us-central1",
            generation_kwargs={
                "temperature": 1.0,
                "max_tokens": 512,
                "top_p": 1.0,
                "top_k": 10,
            },
        )
        type(llm).model_name = mock.PropertyMock(return_value="unit-test-model")

        # mock calling the prediction service
        response = mock.MagicMock()
        type(response).predictions = mock.PropertyMock(
            return_value=["Prompt:\nGemini Pro is cool\nOutput:\nYes, it's cool"]
        )
        llm._call_vertexai_endpoint = mock.MagicMock(return_value=response)

        llm_output = llm._single_output(
            llm._prepare_instances(["Gemini Pro is cool"], num_generations=1)[0]
        )

        assert llm_output == [
            {
                "model_name": llm.model_name,
                "prompt_used": "Gemini Pro is cool",
                "raw_output": "Yes, it's cool",
                "parsed_output": {"generations": "Yes, it's cool"},
            }
        ]
