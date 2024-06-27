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
from unittest import mock

from distilabel.pipeline import Pipeline
from distilabel.steps import (
    GroupColumns,
    LoadDataFromDicts,
    StepInput,
    StepResources,
    step,
)

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


@step(inputs=["instruction"], outputs=["generation"])
def Generate(input: StepInput) -> "StepOutput":
    for row in input:
        row["generation"] = "I'VE GENERATED SOMETHING YAY"
    yield input


@step(step_type="global")
def Global(inputs: StepInput) -> "StepOutput":
    yield inputs


with Pipeline(name="pipeline") as pipeline:
    load_data = LoadDataFromDicts(
        data=[{"instruction": f"{i} instruction"} for i in range(1000)]
    )

    generates_0 = [Generate(resources=StepResources(replicas=i)) for i in range(1, 4)]

    group_0 = GroupColumns(columns=["generation"], output_columns=["generations"])

    global_0 = Global()

    generates_1 = [Generate(resources=StepResources(replicas=i)) for i in range(1, 3)]

    group_1 = GroupColumns(columns=["generation"], output_columns=["generations"])

    global_1 = Global()

    (
        load_data
        >> generates_0
        >> group_0
        >> global_0
        >> generates_1
        >> group_1
        >> global_1
    )


def test_load_stages() -> None:
    with mock.patch.object(
        pipeline, "_all_steps_loaded", wraps=pipeline._all_steps_loaded
    ) as all_steps_loaded_mock:
        pipeline.run(use_cache=False)

    all_steps_loaded_mock.assert_has_calls(
        [
            mock.call(stage=0),
            mock.call(stage=1),
            mock.call(stage=2),
            mock.call(stage=3),
        ]
    )
