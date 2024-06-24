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

from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.batch_manager import _BatchManager
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import StepResources

from .utils import DummyGeneratorStep, DummyStep1, DummyStep2

if TYPE_CHECKING:
    from distilabel.steps.base import GeneratorStep


class TestLocalPipeline:
    def test_request_initial_batches(
        self, dummy_generator_step: "GeneratorStep"
    ) -> None:
        send_batch_to_step_mock = mock.MagicMock()
        pipeline = dummy_generator_step.pipeline
        pipeline._send_batch_to_step = send_batch_to_step_mock  # type: ignore
        pipeline._batch_manager = _BatchManager.from_dag(pipeline.dag)  # type: ignore
        pipeline._request_initial_batches()  # type: ignore

        send_batch_to_step_mock.assert_has_calls(
            [
                mock.call(_Batch(seq_no=0, step_name=step_name, last_batch=False))
                for step_name in pipeline.dag.root_steps
            ]
        )

    def test_send_batch_to_step(self, dummy_generator_step: "GeneratorStep") -> None:
        pipeline = dummy_generator_step.pipeline

        input_queue = mock.MagicMock()
        step = mock.MagicMock()
        step.__getitem__.return_value = input_queue
        get_step_mock = mock.MagicMock(return_value=step)
        pipeline.dag.get_step = get_step_mock  # type: ignore

        batch_manager_mock = mock.MagicMock()
        pipeline._batch_manager = batch_manager_mock  # type: ignore

        batch = _Batch(
            seq_no=0, step_name=dummy_generator_step.name, last_batch=False, data=[[]]
        )
        pipeline._send_batch_to_step(batch=batch)  # type: ignore

        get_step_mock.assert_has_calls([mock.call(dummy_generator_step.name)])
        input_queue.put.assert_called_once_with(batch)

    @mock.patch("distilabel.pipeline.local._ProcessWrapper")
    def test_create_processes(self, process_wrapper_mock: mock.MagicMock) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(
                name="dummy_step_1", resources=StepResources(replicas=2)
            )
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_step_1.connect(dummy_step_2)

        pipeline._pool = mock.MagicMock()
        pipeline._manager = mock.MagicMock()
        pipeline._output_queue = mock.MagicMock()
        pipeline._load_queue = mock.MagicMock()
        pipeline._run_steps()

        assert pipeline._manager.Queue.call_count == 3

        process_wrapper_mock.assert_has_calls(
            [
                mock.call(
                    step=dummy_generator,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                ),
                mock.call(
                    step=dummy_step_1,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                ),
                mock.call(
                    step=dummy_step_1,
                    replica=1,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                ),
                mock.call(
                    step=dummy_step_2,
                    replica=0,
                    input_queue=mock.ANY,
                    output_queue=pipeline._output_queue,
                    load_queue=pipeline._load_queue,
                    dry_run=False,
                ),
            ],
        )

        pipeline._pool.apply_async.assert_has_calls(
            [
                mock.call(
                    process_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
                mock.call(
                    process_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
                mock.call(
                    process_wrapper_mock.return_value.run,
                    error_callback=pipeline._error_callback,
                ),
            ]
        )
