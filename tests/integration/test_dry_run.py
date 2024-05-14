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
from distilabel.steps import LoadDataFromDicts, StepInput, StepOutput, step


@step(inputs=["instruction"], outputs=["response"])
def SucceedAlways(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["response"] = "This step always succeeds"
    yield inputs


def test_dry_run():
    with Pipeline(name="other-pipe") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[
                {"instruction": "Tell me a joke."},
            ]
            * 50,
            batch_size=20,
        )
        text_generation = SucceedAlways()

        load_dataset >> text_generation

    distiset = pipeline.dry_run(parameters={load_dataset.name: {"batch_size": 8}})
    assert len(distiset["default"]["train"]) == 1
    assert pipeline._dry_run is False

    with Pipeline(name="other-pipe") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[
                {"instruction": "Tell me a joke."},
            ]
            * 50,
            batch_size=20,
        )
        text_generation = SucceedAlways()

        load_dataset >> text_generation

    distiset = pipeline.run(
        parameters={load_dataset.name: {"batch_size": 10}}, use_cache=False
    )
    assert len(distiset["default"]["train"]) == 50
