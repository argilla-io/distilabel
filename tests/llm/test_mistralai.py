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

import re
from unittest import mock

import pytest
from distilabel.llm.mistralai import MistralAILLM
from distilabel.tasks.text_generation.base import TextGenerationTask
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from mistralai.models.common import UsageInfo

MODEL_NAME = "mistral-tiny"

system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
system_prompt = ""


@mock.patch("distilabel.llm.mistralai.MistralClient")
@mock.patch(
    "distilabel.llm.mistralai.MistralAILLM.available_models",
    [MODEL_NAME],
)
class TestAnyscaleLLM:
    def test_llm_anyscale(self, mock_mistralai):
        llm = MistralAILLM(
            model=MODEL_NAME,
            task=TextGenerationTask(),
            api_key="api.key",
        )
        assert isinstance(llm, MistralAILLM)
        assert llm.model_name == MODEL_NAME

    def test_unavailable_model(self, mock_mistralai):
        with pytest.raises(
            AssertionError,
            match=re.escape(
                f"Provided `model` is not available in MistralAI, available models are ['{MODEL_NAME}']"
            ),
        ):
            MistralAILLM(
                model="other_model",
                task=TextGenerationTask(),
                api_key="api.key",
            )

    def test_generate(self, mock_mistralai):
        llm = MistralAILLM(
            model=MODEL_NAME,
            task=TextGenerationTask(system_prompt=system_prompt),
            api_key="api.key",
        )

        input = "What is MistralAI?"
        model_response = "model response"
        contents = [
            [
                {
                    "model_name": llm.model_name,
                    "prompt_used": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": input},
                    ],
                    "raw_output": model_response,
                    "parsed_output": {"generations": model_response},
                }
            ]
        ]

        response_choice = ChatCompletionResponse(
            id="0",
            object="chat.completion",
            created=0,
            model=MODEL_NAME,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(**{"role": "user", "content": model_response}),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=1024),
        )

        llm.client.chat = mock.MagicMock(return_value=response_choice)
        result = llm.generate([{"input": input}], num_generations=1)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert len(result) == 1
        assert len(result[0]) == 1
        result = result[0][0]
        assert (
            result["parsed_output"]["generations"]
            == contents[0][0]["parsed_output"]["generations"]
        )

        result = llm.generate([{"input": input}, {"input": "other input"}])
        assert len(result) == 2
        assert (
            result[0][0]["parsed_output"]["generations"]
            == contents[0][0]["parsed_output"]["generations"]
        )
        assert (
            result[1][0]["parsed_output"]["generations"]
            == contents[0][0]["parsed_output"]["generations"]
        )

        result = llm.generate([{"input": input}], num_generations=2)
        assert len(result) == 1
        assert len(result[0]) == 2
        assert (
            result[0][0]["parsed_output"]["generations"]
            == contents[0][0]["parsed_output"]["generations"]
        )
        assert (
            result[0][1]["parsed_output"]["generations"]
            == contents[0][0]["parsed_output"]["generations"]
        )
