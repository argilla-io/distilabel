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
from typing import TYPE_CHECKING

import pytest

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, StepInput, StepResources, step

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


@step(outputs=["generation"])
def Generate(inputs: StepInput) -> "StepOutput":
    # random sleep to simulate processing time
    sleep_time = random.uniform(1.0, 2.0)
    time.sleep(sleep_time)
    for input in inputs:
        input["generation"] = "I slept for {} seconds".format(sleep_time)
    yield inputs


@step(outputs=["generations"])
def CombineGenerations(*inputs: StepInput) -> "StepOutput":
    combined_list = []
    for rows in zip(*inputs):
        combined_dict = {
            "index": rows[0]["index"],
            "instruction": [row["instruction"] for row in rows],
            "generations": [row["generation"] for row in rows],
        }

        # Check consistency in "index" and "instruction"
        if any(row["index"] != combined_dict["index"] for row in rows):
            raise ValueError("Inconsistent 'index' or 'instruction'")

        combined_list.append(combined_dict)

    yield combined_list


@pytest.mark.xfail
def test_multiple_replicas() -> None:
    with Pipeline(name="test") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[{"index": i, "instruction": f"Instruction {i}"} for i in range(1000)]
        )

        generates = [Generate(resources=StepResources(replicas=2)) for _ in range(4)]

        combine_generations = CombineGenerations(input_batch_size=500)

        load_dataset >> generates >> combine_generations

    distiset = pipeline.run(use_cache=False)

    for i, row in enumerate(distiset["default"]["train"]):
        assert row["index"] == i
        assert len(row["generations"]) == 4
