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

from typing import Any, Dict, List, Union

import pytest
from distilabel.llms import LLM
from distilabel.steps.tasks.structured_generation import StructuredGeneration
from distilabel.steps.tasks.typing import ChatType
from pydantic import BaseModel


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


class DummyLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    def generate(
        self, inputs: List["ChatType"], num_generations: int = 1, **kwargs: Any
    ) -> List[List[Union[str, None]]]:
        return [["output" for _ in range(num_generations)] for _ in inputs]


class TestStructuredGeneration:
    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                {
                    "instruction": "What's the weather like today in Seattle in Celsius degrees?",
                    "grammar": {  # Unexpected name
                        "type": "regex",
                        "value": "(\\d{1,2})°C",
                    },
                },
                (
                    [
                        {
                            "role": "user",
                            "content": "What's the weather like today in Seattle in Celsius degrees?",
                        }
                    ],
                    None,
                ),
            ),
            (
                {
                    "instruction": "What's the weather like today in Seattle in Celsius degrees?",
                    "structured_output": {
                        "format": "regex",
                        "schema": "(\\d{1,2})°C",
                    },
                },
                (
                    [
                        {
                            "role": "user",
                            "content": "What's the weather like today in Seattle in Celsius degrees?",
                        }
                    ],
                    {
                        "format": "regex",
                        "schema": "(\\d{1,2})°C",
                    },
                ),
            ),
        ],
    )
    def test_format_input_with_data(
        self, data: Dict[str, Any], expected: List[Dict[str, Any]]
    ) -> None:
        task = StructuredGeneration(
            name="task",
            llm=DummyLLM(),
        )
        task.load()
        assert task.format_input(data) == expected
