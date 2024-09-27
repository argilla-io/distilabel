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

from typing import TYPE_CHECKING

from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts, StepInput, step

if TYPE_CHECKING:
    from distilabel.steps import StepOutput


@step(inputs=["instruction"], outputs=["response"])
def FailAlways(_: StepInput) -> "StepOutput":
    raise Exception("This step always fails")


@step(inputs=["instruction"], outputs=["response"])
def SucceedAlways(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["response"] = "This step always succeeds"
    yield inputs


def test_branching_missalignment_because_step_fails_processing_batch() -> None:
    with Pipeline(name="") as pipeline:
        load_data = LoadDataFromDicts(data=[{"instruction": i} for i in range(20)])

        fail = FailAlways()
        succeed = SucceedAlways()
        combine = GroupColumns(columns=["response"])

        load_data >> [fail, succeed] >> combine

    distiset = pipeline.run(use_cache=False)

    assert (
        distiset["default"]["train"]["grouped_response"]
        == [[None, "This step always succeeds"]] * 20
    )
