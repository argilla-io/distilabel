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

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest import mock

from distilabel.pipeline import Pipeline, sample_n_steps
from distilabel.steps import (
    GroupColumns,
    LoadDataFromDicts,
    StepInput,
    StepResources,
    step,
)

if TYPE_CHECKING:
    from distilabel.pipeline.batch import _Batch
    from distilabel.typing import StepOutput


routing_batch_function = sample_n_steps(2)


@step(inputs=["instruction"], outputs=["generation"])
def Generate(input: StepInput) -> "StepOutput":
    for row in input:
        row["generation"] = "I'VE GENERATED SOMETHING YAY"
    yield input


@step(step_type="global")
def Global(inputs: StepInput) -> "StepOutput":
    yield inputs


def test_load_stages() -> None:
    with Pipeline(name="pipeline") as pipeline:
        load_data = LoadDataFromDicts(
            data=[{"instruction": f"{i} instruction"} for i in range(1000)]
        )

        generates_0 = [
            Generate(resources=StepResources(replicas=i)) for i in range(1, 4)
        ]

        group_0 = GroupColumns(columns=["generation"], output_columns=["generations"])

        global_0 = Global()

        generates_1 = [
            Generate(resources=StepResources(replicas=i)) for i in range(1, 3)
        ]

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

    with mock.patch.object(
        pipeline, "_run_stage_steps_and_wait", wraps=pipeline._run_stage_steps_and_wait
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


def test_load_stages_with_routing_batch_function() -> None:
    with Pipeline(name="pipeline") as pipeline:
        load_data = LoadDataFromDicts(
            data=[{"instruction": f"{i} instruction"} for i in range(1000)]
        )

        generates_0 = [
            Generate(resources=StepResources(replicas=i)) for i in range(1, 4)
        ]

        group_0 = GroupColumns(columns=["generation"], output_columns=["generations"])

        global_0 = Global()

        load_data >> routing_batch_function >> generates_0 >> group_0 >> global_0

    with mock.patch.object(
        pipeline, "_run_stage_steps_and_wait", wraps=pipeline._run_stage_steps_and_wait
    ) as all_steps_loaded_mock:
        pipeline.run(use_cache=False)

    all_steps_loaded_mock.assert_has_calls([mock.call(stage=0), mock.call(stage=1)])


def test_load_stages_status_load_from_cache() -> None:
    with TemporaryDirectory() as tmp_dir:
        with Pipeline(name="pipeline", cache_dir=tmp_dir) as pipeline:
            load_data = LoadDataFromDicts(
                data=[{"instruction": f"{i} instruction"} for i in range(1000)]
            )

            generates_0 = [
                Generate(resources=StepResources(replicas=i)) for i in range(1, 4)
            ]

            group_0 = GroupColumns(
                columns=["generation"], output_columns=["generations"]
            )

            global_0 = Global()

            generates_1 = [
                Generate(resources=StepResources(replicas=i)) for i in range(1, 3)
            ]

            group_1 = GroupColumns(
                columns=["generation"], output_columns=["generations"]
            )

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

            original_process_batch = pipeline._process_batch

        def _process_batch_wrapper(
            batch: "_Batch", send_last_batch_flag: bool = True
        ) -> None:
            if batch.step_name == group_1.name and batch.seq_no == 10:
                pipeline._stop_called = True
            original_process_batch(batch, send_last_batch_flag)

        # Run first time and stop the pipeline when specific batch received (simulate CTRL + C)
        with mock.patch.object(pipeline, "_process_batch", _process_batch_wrapper):
            pipeline.run(use_cache=True)

        distiset = pipeline.run(use_cache=True)

        assert len(distiset["default"]["train"]) == 1000
