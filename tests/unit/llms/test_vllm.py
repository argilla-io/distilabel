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

from typing import List

import numpy as np
import pytest
from distilabel.llms import vLLM
from distilabel.llms.vllm import _sort_batches
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


SAMPLE_DATA = [
    [
        {
            "instruction": "Generate a character from a RPG game.",
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": "Generate an animal from a zoo.",
            "structured_output": {
                "format": "json",
                "schema": Animal.model_json_schema(),
            },
        },
        {
            "instruction": "Repeated character",
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": "What's the weather like today in Seattle in Celsius degrees?",
            "structured_output": {
                "format": "regex",
                "schema": "(\\d{1,2})°C",
            },
        },
        {
            "instruction": "Other character",
            "structured_output": {
                "format": "json",
                "schema": Character.model_json_schema(),
            },
        },
        {
            "instruction": "repeated regex",
            "structured_output": {
                "format": "regex",
                "schema": "(\\d{1,2})°C",
            },
        },
    ]
]


# Just a mock to avoid loading the model
class DummyTokenizer:
    def __init__(self) -> None:
        pass

    def apply_chat_template(self, input, **kwargs):
        return input


class TestvLLM:
    @pytest.mark.parametrize(
        "num_generations, expected_sorted_batches",
        [
            (
                1,
                [
                    "Generate a character from a RPG game.",
                    "Generate an animal from a zoo.",
                    "Repeated character",
                    "What's the weather like today in Seattle in Celsius degrees?",
                    "Other character",
                    "repeated regex",
                ],
            ),
            (
                3,
                np.repeat(
                    [
                        "Generate a character from a RPG game.",
                        "Generate an animal from a zoo.",
                        "Repeated character",
                        "What's the weather like today in Seattle in Celsius degrees?",
                        "Other character",
                        "repeated regex",
                    ],
                    3,
                ).tolist(),
            ),
        ],
    )
    def test_prepare_batches_and_sort_back(
        self, num_generations: int, expected_sorted_batches: List[str]
    ):
        formatted_inputs = [
            (item["instruction"], item["structured_output"])
            for row in SAMPLE_DATA
            for item in row
        ]
        llm = vLLM(model="dummy")
        llm._tokenizer = DummyTokenizer()
        batches, indices = llm._prepare_batches(formatted_inputs)
        # NOTE: We have to simulate calling self._model.generate(n=num_generations) and then sorting the results
        num_generations_batches = []
        for batch in batches:
            num_generations_batches.append(
                (np.repeat(batch[0], num_generations).tolist(), batch[1])
            )
        batches = num_generations_batches
        # Recreate as the output from batched_outputs += [[output.text for output in outputs.outputs] for outputs in batch_outputs]
        batches = [batch for batch, _ in batches]
        sorted_batches = _sort_batches(
            batches, indices, num_generations=num_generations
        )

        assert sorted_batches == [
            np.repeat(
                [
                    "Generate a character from a RPG game.",
                    "Generate an animal from a zoo.",
                    "Repeated character",
                ],
                num_generations,
            ).tolist(),
            np.repeat(
                ["What's the weather like today in Seattle in Celsius degrees?"],
                num_generations,
            ).tolist(),
            np.repeat(
                [
                    "Other character",
                    "repeated regex",
                ],
                num_generations,
            ).tolist(),
        ]
