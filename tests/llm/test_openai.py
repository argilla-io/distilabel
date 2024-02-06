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
from unittest.mock import Mock, patch

import pytest
from distilabel.llm.openai import JSONOpenAILLM
from distilabel.tasks import TextGenerationTask


@pytest.fixture(scope="module", autouse=True)
def mock_json_openai_llm():
    mocked_chat_return_value = Mock(
        choices=[Mock(message=Mock(content='{"answer": "Madrid"}'))]
    )
    with patch(
        "openai.resources.chat.Completions.create",
        return_value=mocked_chat_return_value,
    ):
        JSONOpenAILLM._available_models = Mock(return_value=["gpt-3.5-turbo-1106"])
        JSONOpenAILLM.available_models = ["gpt-3.5-turbo-1106"]
        yield JSONOpenAILLM(
            model="gpt-3.5-turbo-1106",
            task=TextGenerationTask(),
        )


def test_generate(mock_json_openai_llm):
    prompt = "write a json object with a key 'answer' and value 'Paris'"
    inputs = [{"input": prompt}]
    outputs = mock_json_openai_llm.generate(inputs)
    assert len(outputs) == 1
    assert isinstance(outputs[0], list)
    json_response = json.loads(outputs[0][0]["parsed_output"]["generations"])
    assert isinstance(json_response, dict)
    assert json_response["answer"] == "Madrid"


def test_available_models(mock_json_openai_llm):
    assert mock_json_openai_llm._available_models() == ["gpt-3.5-turbo-1106"]
    assert "gpt-3.5-turbo-1106" in mock_json_openai_llm.available_models
    mock_json_openai_llm._available_models.assert_called_once()
