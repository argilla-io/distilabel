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

import random
import time
from typing import TYPE_CHECKING, List

import pytest

from distilabel.pipeline import Pipeline, routing_batch_function
from distilabel.steps import LoadDataFromDicts, StepInput, step

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


@routing_batch_function()
def random_routing_batch(steps: List[str]) -> List[str]:
    return random.sample(steps, 2)


@step(outputs=["generation"])
def Generate(inputs: StepInput) -> "StepOutput":
    # random sleep to simulate processing time
    sleep_time = random.uniform(1.0, 2.0)
    time.sleep(sleep_time)
    for input in inputs:
        input["generation"] = "I slept for {} seconds".format(sleep_time)
    yield inputs


@step(outputs=["generations"])
def Generate2(inputs: StepInput) -> "StepOutput":
    sleep_time = random.uniform(1.0, 2.0)
    time.sleep(sleep_time)
    for input in inputs:
        input["2generation"] = "I slept for {} seconds".format(sleep_time)
    yield inputs


@step(outputs=["generations"])
def CombineGenerations(*inputs: StepInput) -> "StepOutput":
    generation_key = (
        "2generation" if "2generation" in inputs[0][0].keys() else "generation"
    )

    combined_list = []
    for rows in zip(*inputs):
        combined_dict = {
            "index": rows[0]["index"],
            "instruction": [row["instruction"] for row in rows],
            f"{generation_key}s": [row[generation_key] for row in rows],
        }

        # Check consistency in "index" and "instruction"
        if any(row["index"] != combined_dict["index"] for row in rows):
            raise ValueError("Inconsistent 'index' or 'instruction'")

        combined_list.append(combined_dict)

    yield combined_list


@pytest.mark.xfail
@pytest.mark.timeout(240)
def test_routing_batch_function() -> None:
    with Pipeline(name="test") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[{"index": i, "instruction": f"Instruction {i}"} for i in range(1000)]
        )

        generates = [Generate() for _ in range(4)]

        combine_generations = CombineGenerations()

        load_dataset >> random_routing_batch >> generates >> combine_generations

    distiset = pipeline.run(use_cache=False)

    for i, row in enumerate(distiset["default"]["train"]):
        assert row["index"] == i
        assert row["instruction"] == [f"Instruction {i}", f"Instruction {i}"]
        assert len(row["generations"]) == 2


@pytest.mark.xfail
@pytest.mark.timeout(240)
def test_routing_batch_function_irregular_batch_sizes() -> None:
    with Pipeline(name="test") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[{"index": i, "instruction": f"Instruction {i}"} for i in range(1000)],
            batch_size=200,
        )

        generates = [
            Generate(input_batch_size=input_batch_size)
            for input_batch_size in [25, 50, 100, 200]
        ]

        combine_generations = CombineGenerations(input_batch_size=25)

        load_dataset >> random_routing_batch >> generates >> combine_generations

    distiset = pipeline.run(use_cache=False)

    for i, row in enumerate(distiset["default"]["train"]):
        assert row["index"] == i
        assert row["instruction"] == [f"Instruction {i}", f"Instruction {i}"]
        assert len(row["generations"]) == 2


@pytest.mark.xfail
@pytest.mark.timeout(240)
def test_multiple_routing_batch_function() -> None:
    batch_size = 200

    with Pipeline(name="test") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[
                {
                    "index": i,
                    "instruction": f"Instruction {i}",
                    "batch": i // batch_size,
                }
                for i in range(1000)
            ],
            batch_size=batch_size,
        )

        generates = [
            Generate(input_batch_size=input_batch_size)
            for input_batch_size in [25, 50, 100, 200]
        ]

        combine_generations = CombineGenerations(input_batch_size=25)

        generates2 = [Generate2(input_batch_size=25) for _ in range(4)]

        combine_generations_2 = CombineGenerations(input_batch_size=25)

        (
            load_dataset
            >> random_routing_batch
            >> generates
            >> combine_generations
            >> random_routing_batch
            >> generates2
            >> combine_generations_2
        )

    distiset = pipeline.run(use_cache=False)

    for i, row in enumerate(distiset["default"]["train"]):
        assert row["index"] == i
        assert len(row["2generations"]) == 2
