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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import StepResources

from .utils import DummyGeneratorStep, DummyStep1, DummyStep2

if TYPE_CHECKING:
    pass


class TestPipeline:
    @mock.patch("distilabel.pipeline.local._StepWrapper")
    def test_run_steps(self, step_wrapper_mock: mock.MagicMock) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(
                name="dummy_step_1", resources=StepResources(replicas=2)
            )
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> dummy_step_1 >> dummy_step_2

        pipeline._pool = mock.MagicMock()
        pipeline._manager = mock.MagicMock()
        pipeline._output_queue = mock.MagicMock()
        pipeline._load_queue = mock.MagicMock()
        pipeline._create_steps_input_queues()
        pipeline._run_steps(
            steps=[dummy_generator.name, dummy_step_1.name, dummy_step_2.name]  # type: ignore
        )

        assert pipeline._manager.Queue.call_count == 3

        step_wrapper_mock.assert_has_calls(
            [
                mock.call(
                    step=dummy_generator,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                    ray_pipeline=False,
                ),
                mock.call(
                    step=dummy_step_1,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                    ray_pipeline=False,
                ),
                mock.call(
                    step=dummy_step_1,
                    replica=1,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                    ray_pipeline=False,
                ),
                mock.call(
                    step=dummy_step_2,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                    ray_pipeline=False,
                ),
            ],
        )

        pipeline._pool.apply_async.assert_has_calls(
            [
                mock.call(
                    step_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
                mock.call(
                    step_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
                mock.call(
                    step_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
            ]
        )

    def test_ray(self) -> None:
        with Pipeline(
            name="dummy",
            description="dummy",
            cache_dir="/tmp",
            enable_metadata=True,
            requirements=["dummy"],
        ) as pipeline:
            generator = DummyGeneratorStep()
            dummy = DummyStep1()

            generator >> dummy

        ray_pipeline = pipeline.ray()

        assert ray_pipeline.name == pipeline.name
        assert ray_pipeline.description == pipeline.description
        assert ray_pipeline._cache_dir == pipeline._cache_dir
        assert ray_pipeline._enable_metadata == pipeline._enable_metadata
        assert ray_pipeline.requirements == pipeline.requirements
        assert ray_pipeline.dag == pipeline.dag

    def test_run_detected_ray(self) -> None:
        with Pipeline(
            name="dummy",
            description="dummy",
            cache_dir="/tmp",
            enable_metadata=True,
            requirements=["dummy"],
        ) as pipeline:
            generator = DummyGeneratorStep()
            dummy = DummyStep1()

            generator >> dummy

        run_pipeline_mock = mock.MagicMock()

        with (
            mock.patch(
                "distilabel.pipeline.local.script_executed_in_ray_cluster",
                return_value=True,
            ),
            mock.patch(
                "distilabel.pipeline.local.Pipeline.ray", return_value=run_pipeline_mock
            ) as ray_mock,
        ):
            pipeline.run()

        ray_mock.assert_called_once()
        run_pipeline_mock.run.assert_called_once()
