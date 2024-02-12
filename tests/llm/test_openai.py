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


@pytest.fixture(scope="function", autouse=True)
def mock_json_openai_llm():
    # Mocking available models and models that support JSON
    json_supporting_model = "gpt-3.5-turbo-1106"
    non_json_supporting_model = "gpt-4"
    available_models = [
        Mock(id=json_supporting_model),
        Mock(id=non_json_supporting_model),
    ]
    # Mocking the OpenAI client and chat method
    mocked_chat = Mock(
        completions=Mock(
            create=Mock(
                return_value=Mock(
                    choices=[Mock(message=Mock(content='{"answer": "Madrid"}'))]
                )
            )
        )
    )
    mocked_models = Mock(list=Mock(return_value=Mock(data=available_models)))
    mocked_openai_client = Mock(
        chat=mocked_chat,
        models=mocked_models,
    )
    with pytest.raises(AssertionError):
        JSONOpenAILLM(
            model=non_json_supporting_model,
            task=TextGenerationTask(),
            client=mocked_openai_client,
        )
    yield JSONOpenAILLM(
        model=json_supporting_model,
        task=TextGenerationTask(),
        client=mocked_openai_client,
    )


def test_generate_json(mock_json_openai_llm):
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
