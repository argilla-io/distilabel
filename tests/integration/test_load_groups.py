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
from distilabel.steps import LoadDataFromDicts, StepInput, step

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


@step(inputs=["instruction"], outputs=["instruction2"])
def DummyStep(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction2"] = "miau"
    yield inputs


@step(inputs=["instruction"], outputs=["instruction2"])
def DummyStep2(*inputs: StepInput) -> "StepOutput":
    outputs = []
    for rows in zip(*inputs):
        combined = {}
        for row in rows:
            combined.update(row)
        outputs.append(combined)
    yield outputs


@step(inputs=["instruction"], outputs=["instruction2"], step_type="global")
def GlobalDummyStep(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction2"] = "miau"
    yield inputs


def test_load_groups() -> None:
    with Pipeline() as pipeline:
        generator = LoadDataFromDicts(data=[{"instruction": "Hi"}] * 50)
        dummy_step_0 = DummyStep()
        dummy_step_1 = DummyStep()
        dummy_step_2 = DummyStep2()
        global_dummy_step = GlobalDummyStep()
        dummy_step_3 = DummyStep()
        dummy_step_4 = DummyStep()
        dummy_step_5 = DummyStep()

        (
            generator
            >> [dummy_step_0, dummy_step_1]
            >> dummy_step_2
            >> global_dummy_step
            >> dummy_step_3
            >> [dummy_step_4, dummy_step_5]
        )

    with mock.patch.object(
        pipeline, "_run_stage_steps_and_wait", wraps=pipeline._run_stage_steps_and_wait
    ) as run_stage_mock:
        # `dummy_step_0` should be executed in isolation
        pipeline.run(load_groups=[[dummy_step_0.name], [dummy_step_3.name]])

    assert run_stage_mock.call_count == 6
