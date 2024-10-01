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

    def test_process_with_tools(self) -> None:
        step = FormatTextGenerationSFT(tools=True)
        step.load()

        assert next(
            step.process(
                [
                    {
                        "instruction": "I'd like to convert the complex number 3 + 4j and 1 - 2j to polar coordinates.",
                        "generation": '[{"name": "complex_to_polar", "arguments": {"complex_number": "3 + 4j"}}, {"name": "complex_to_polar", "arguments": {"complex_number": "1 - 2j"}}]',
                        "tools": '[{"type":"function","function":{"name":"complex_to_polar","description":"Converts a complex number to its polar coordinate representation.","parameters":{"type":"object","properties":{"complex_number":{"type":"object","description":"A complex number in the form of `real + imaginary * 1j`."}},"required":["complex_number"]}}}]',
                    }
                ]
            )
        ) == [
            {
                "instruction": "I'd like to convert the complex number 3 + 4j and 1 - 2j to polar coordinates.",
                "generation": '[{"name": "complex_to_polar", "arguments": {"complex_number": "3 + 4j"}}, {"name": "complex_to_polar", "arguments": {"complex_number": "1 - 2j"}}]',
                "tools": '[{"type":"function","function":{"name":"complex_to_polar","description":"Converts a complex number to its polar coordinate representation.","parameters":{"type":"object","properties":{"complex_number":{"type":"object","description":"A complex number in the form of `real + imaginary * 1j`."}},"required":["complex_number"]}}}]',
                "prompt": "I'd like to convert the complex number 3 + 4j and 1 - 2j to polar coordinates.",
                "prompt_id": "f772a2cd00f4cc268358c4f74a618c512fbb0991cdb53cfa219197cb2adf7f42",
                "messages": [
                    {
                        "role": "user",
                        "content": "I'd like to convert the complex number 3 + 4j and 1 - 2j to polar coordinates.",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": '[{"name": "complex_to_polar", "arguments": {"complex_number": "3 + 4j"}}, {"name": "complex_to_polar", "arguments": {"complex_number": "1 - 2j"}}]',
                    },
                    {
                        "role": "tool",
                        "content": '[{"type":"function","function":{"name":"complex_to_polar","description":"Converts a complex number to its polar coordinate representation.","parameters":{"type":"object","properties":{"complex_number":{"type":"object","description":"A complex number in the form of `real + imaginary * 1j`."}},"required":["complex_number"]}}}]',
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
