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
from unittest.mock import Mock, patch, TestCase

from distilabel.llm.openai import JSONOpenAILLM
from distilabel.tasks import TextGenerationTask


class TestJSONOpenAILLM(TestCase):
    @patch("distilabel.llm.openai.OpenAILLM.available_models")
    @patch("openai.resources.chat.Completions.create")
    def test_generate(self, mock_create):
        # Mock the available_models property
        mock_create.return_value = ["gpt-3.5-turbo-1106"]
        
        # Mock the response from the OpenAI API
        mock_create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"answer": "Madrid"}'))]
        )

        # Initialize the JSONOpenAILLM class
        llm = JSONOpenAILLM(task=TextGenerationTask())

        # Call the _generate method
        inputs = [
            {"input": "write a json object with a key 'answer' and value 'Paris'"}
        ]
        outputs = llm.generate(inputs)
        json_response = json.loads(outputs[0][0]["parsed_output"]["generations"])

        # Check the output
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(json_response, dict)
        self.assertEqual(json_response["answer"], "Madrid")
        self.assertEqual(llm.model, llm.json_supporting_models)

        # Check if the OpenAI API was called with the correct arguments
        prompts = llm._generate_prompts(inputs, default_format="openai")
        prompt = prompts[0]
        mock_create.assert_called_once_with(
            messages=prompt,
            model=llm.model,
            n=1,
            max_tokens=llm.max_tokens,
            frequency_penalty=llm.frequency_penalty,
            presence_penalty=llm.presence_penalty,
            temperature=llm.temperature,
            top_p=llm.top_p,
            timeout=50,
            response_format={"type": "json_object"},
        )
