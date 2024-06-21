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


from distilabel.pipeline import Pipeline
from distilabel.steps.formatting.sft import (
    FormatChatGenerationSFT,
    FormatTextGenerationSFT,
)


class TestFormatTextGenerationSFT:
    def test_process(self) -> None:
        step = FormatTextGenerationSFT(name="sft", pipeline=Pipeline(name="pipeline"))
        step.load()

        assert next(
            step.process(
                [
                    {
                        "instruction": "What's 2+2?",
                        "generation": "4",
                    }
                ]
            )
        ) == [
            {
                "instruction": "What's 2+2?",
                "generation": "4",
                "prompt": "What's 2+2?",
                "prompt_id": "7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29",
                "messages": [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            }
        ]

    def test_process_with_system_prompt(self) -> None:
        step = FormatTextGenerationSFT(name="sft", pipeline=Pipeline(name="pipeline"))
        step.load()

        assert next(
            step.process(
                [
                    {
                        "system_prompt": "You are a helpful assistant.",
                        "instruction": "What's 2+2?",
                        "generation": "4",
                    }
                ]
            )
        ) == [
            {
                "system_prompt": "You are a helpful assistant.",
                "instruction": "What's 2+2?",
                "generation": "4",
                "prompt": "What's 2+2?",
                "prompt_id": "7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
            }
        ]

    def test_process_with_function_call(self) -> None:
        step = FormatTextGenerationSFT(function_calling=True)
        # If is a json_schema, we don't need to transform it,
        # otherwise we assume it's an openai tool, and we need to transform it
        # to a json chema.
        step.load()

        from pydantic import BaseModel, Field

        class get_animal_name(BaseModel):
            name: str = Field(description="The name of the animal")
            species: str = Field(description="The species of the animal")

        import random

        random.seed(42)
        assert next(
            step.process(
                [
                    {
                        "instruction": "What's the most typical mascot of a sports team?",
                        "generation": "{'name': 'bear', 'species': 'mammal'}",
                        "structured_output": {
                            "format": "json",
                            "schema": get_animal_name.model_json_schema(),
                        },
                    }
                ]
            )
        ) == [
            {
                "instruction": "What's the most typical mascot of a sports team?",
                "generation": "{'name': 'bear', 'species': 'mammal'}",
                "prompt": "What's the most typical mascot of a sports team?",
                "prompt_id": "1415cfa117fecbf8f763bff1da19ee40e8739056d2f4b94276087e6692a3a380",
                "structured_output": {
                    "format": "json",
                    "schema": get_animal_name.model_json_schema(),
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the most typical mascot of a sports team?",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "NbrnTP3fA",
                                "type": "function",
                                "function": {
                                    "name": "get_animal_name",
                                    "arguments": "\"{'name': 'bear', 'species': 'mammal'}\"",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "\"{'name': 'bear', 'species': 'mammal'}\"",
                        "tool_call_id": "NbrnTP3fA",
                    },
                ],
            }
        ]


class TestFormatChatGenerationSFT:
    def test_process(self) -> None:
        step = FormatChatGenerationSFT(name="sft", pipeline=Pipeline(name="pipeline"))
        step.load()

        assert next(
            step.process(
                [
                    {
                        "messages": [{"role": "user", "content": "What's 2+2?"}],
                        "generation": "4",
                    }
                ]
            )
        ) == [
            {
                "messages": [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                "generation": "4",
                "prompt": "What's 2+2?",
                "prompt_id": "7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29",
            }
        ]
