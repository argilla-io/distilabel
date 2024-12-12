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

from typing import Any, Dict, List
from unittest import mock

import pytest
from openai.pagination import SyncPage
from openai.types import Model
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel
from transformers import AutoTokenizer

from distilabel.models.llms import vLLM
from distilabel.models.llms.vllm import ClientvLLM


class Character(BaseModel):
    name: str
    description: str
    role: str
    weapon: str


class Animal(BaseModel):
    name: str
    species: str
    habitat: str
    diet: str


SAMPLE_DATA = [
    [
        {
            "instruction": [
                {"role": "user", "content": "Generate a character from a RPG game."}
            ],
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": [
                {
                    "role": "user",
                    "content": "Generate an animal from a zoo.",
                }
            ],
            "structured_output": {
                "format": "json",
                "schema": Animal.model_json_schema(),
            },
        },
        {
            "instruction": [{"role": "user", "content": "Repeated character"}],
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": [
                {
                    "role": "user",
                    "content": "What's the weather like today in Seattle in Celsius degrees?",
                }
            ],
            "structured_output": {
                "format": "regex",
                "schema": "(\\d{1,2})°C",
            },
        },
        {
            "instruction": [{"role": "user", "content": "Other character"}],
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": [{"role": "user", "content": "repeated regex"}],
            "structured_output": {
                "format": "regex",
                "schema": "(\\d{1,2})°C",
            },
        },
    ]
]


class TestvLLM:
    @pytest.mark.parametrize(
        "multi_structured_output",
        (True, False),
    )
    @pytest.mark.parametrize(
        "num_generations, expected_result",
        [
            (
                1,
                [
                    {
                        "generations": ["I'm fine thank you"],
                        "statistics": {"input_tokens": [21], "output_tokens": [6]},
                        "logprobs": [
                            [
                                [
                                    {"token": "I'm", "logprob": -1},
                                    {"token": "Hello", "logprob": -3},
                                ],
                                [
                                    {"token": "I'm", "logprob": -1},
                                    {"token": "Hello", "logprob": -3},
                                ],
                            ]
                        ],
                    }
                ],
            ),
            (
                2,
                [
                    {
                        "generations": ["I'm fine thank you"] * 2,
                        "statistics": {
                            "input_tokens": [21, 21],
                            "output_tokens": [6, 6],
                        },
                        "logprobs": [
                            [
                                [
                                    {"token": "I'm", "logprob": -1},
                                    {"token": "Hello", "logprob": -3},
                                ],
                                [
                                    {"token": "I'm", "logprob": -1},
                                    {"token": "Hello", "logprob": -3},
                                ],
                            ]
                        ]
                        * 2,
                    }
                ],
            ),
        ],
    )
    def test_generate(
        self,
        multi_structured_output: bool,
        num_generations: int,
        expected_result: List[Dict[str, Any]],
    ) -> None:
        llm = vLLM(model="dummy")
        tokenizer = AutoTokenizer.from_pretrained(
            "distilabel-internal-testing/tiny-random-mistral"
        )
        llm._tokenizer = tokenizer
        vllm_mock = mock.MagicMock()
        vllm_mock.get_tokenizer = mock.MagicMock(return_value=tokenizer)
        # mock the import by hacking sys.modules
        # https://stackoverflow.com/questions/60919705/how-to-mock-in-a-python-unittest-a-library-not-installed-locally
        import sys

        if "vllm" not in sys.modules:
            sys.modules["vllm"] = vllm_mock
        llm._model = vllm_mock

        mocked_requests_output = [
            mock.Mock(  # RequestOutput
                outputs=[
                    mock.Mock(  # CompletionOutput
                        text="I'm fine thank you",
                        token_ids=[1, 2, 3, 4, 5, 7],
                        logprobs=[
                            {
                                1: mock.Mock(decoded_token="I'm", logprob=-1),
                                2: mock.Mock(decoded_token="Hello", logprob=-3),
                            },
                            {
                                1: mock.Mock(decoded_token="I'm", logprob=-1),
                                2: mock.Mock(decoded_token="Hello", logprob=-3),
                            },
                        ],
                    )
                ]
                * num_generations,
            )
        ]

        llm._model.generate = mock.MagicMock(return_value=mocked_requests_output)
        if not multi_structured_output:
            formatted_inputs = [
                [
                    {"role": "system", "content": "sysprompt"},
                    {
                        "role": "user",
                        "content": "I'm fine thank you",
                    },
                ]
            ]
        else:
            formatted_inputs = [
                (
                    [
                        {"role": "system", "content": "sysprompt"},
                        {
                            "role": "user",
                            "content": "I'm fine thank you",
                        },
                    ],
                    {
                        # "format": "json",
                        "format": "regex",
                        "schema": r".*",
                        # "schema": Character.model_json_schema(),
                    },
                )
            ]
        result = llm.generate(inputs=formatted_inputs, num_generations=num_generations)
        assert result == expected_result


@mock.patch("openai.OpenAI")
@mock.patch("openai.AsyncOpenAI")
class TestClientvLLM:
    def test_clientvllm_model_name(
        self, _: mock.MagicMock, openai_mock: mock.MagicMock
    ) -> None:
        llm = ClientvLLM(
            base_url="http://localhost:8000/v1",
            tokenizer="google-bert/bert-base-uncased",
        )

        llm._client = mock.MagicMock()
        llm._client.models.list.return_value = SyncPage[Model](  # type: ignore
            data=[Model(id="llama", created=1234, object="model", owned_by="")],
            object="model",
        )

        assert llm.model_name == "llama"

    @pytest.mark.asyncio
    async def test_agenerate(
        self, _openai_mock: mock.MagicMock, _async_openai_mock: mock.MagicMock
    ) -> None:
        llm = ClientvLLM(
            base_url="http://localhost:8000/v1",
            tokenizer="distilabel-internal-testing/tiny-random-mistral",
        )

        llm.load()

        llm._aclient.completions.create = mock.AsyncMock(
            return_value=Completion(
                id="1234",
                created=1234,
                model="llama",
                object="text_completion",
                choices=[
                    CompletionChoice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        text="I'm fine thank you",
                    ),
                    CompletionChoice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        text="I'm fine thank you sir",
                    ),
                ],
                usage=CompletionUsage(
                    completion_tokens=10,
                    prompt_tokens=10,
                    total_tokens=20,
                ),
            )
        )

        generations = await llm.agenerate(
            input=[{"role": "user", "content": "Hi, how are you?"}]
        )

        assert generations == {
            "generations": ["I'm fine thank you", "I'm fine thank you sir"],
            "statistics": {
                "input_tokens": [10],
                "output_tokens": [10],
            },
        }
