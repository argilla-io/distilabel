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

import json
from unittest.mock import Mock

import pytest
from distilabel.llm.openai import JSONOpenAILLM
from distilabel.tasks import TextGenerationTask


@pytest.fixture(scope="module", autouse=True)
def mock_json_openai_llm():
    # Mocking available models and models that support JSON
    json_supporting_model = "gpt-3.5-turbo-1106"
    available_models = [json_supporting_model]
    JSONOpenAILLM.available_models = available_models
    JSONOpenAILLM._available_models = Mock(return_value=available_models)

    # Mocking the OpenAI client and chat method
    mocked_chat_return_value = Mock(
        choices=[Mock(message=Mock(content='{"answer": "Madrid"}'))]
    )
    mocked_openai_client = Mock(
        chat=Mock(completions=Mock(create=Mock(return_value=mocked_chat_return_value))),
        api_key="api_key",
        organization="organization",
        models=Mock(list=Mock(return_value=available_models)),
    )
    text_generation_task = TextGenerationTask()
    yield JSONOpenAILLM(
        model=json_supporting_model,
        task=text_generation_task,
        client=mocked_openai_client,
    )


def test_generate(mock_json_openai_llm):
    # Setup inputs
    _prompt = "write a json object with a key 'answer' and value 'Paris'"
    inputs = [{"input": _prompt}]
    outputs = mock_json_openai_llm.generate(inputs)
    # Asserts
    assert len(outputs) == 1
    assert isinstance(outputs[0], list)
    json_response = json.loads(outputs[0][0]["parsed_output"]["generations"])
    assert isinstance(json_response, dict)
    assert json_response["answer"] == "Madrid"
    mock_json_openai_llm.client.chat.completions.create.assert_called_once()


def test_available_models(mock_json_openai_llm):
    assert mock_json_openai_llm._available_models() == ["gpt-3.5-turbo-1106"]
    assert "gpt-3.5-turbo-1106" in mock_json_openai_llm.available_models
    mock_json_openai_llm._available_models.assert_called_once()
