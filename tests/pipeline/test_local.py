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

from distilabel.pipeline.base import _Batch
from distilabel.pipeline.local import Pipeline

from tests.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

if TYPE_CHECKING:
    from distilabel.pipeline.step.base import GeneratorStep


class TestPipeline:
    def test_request_initial_batches(
        self, dummy_generator_step: "GeneratorStep"
    ) -> None:
        request_batch_to_generator_mock = mock.MagicMock()
        pipeline = dummy_generator_step.pipeline
        pipeline._request_batch_to_generator = request_batch_to_generator_mock  # type: ignore

        pipeline._request_initial_batches()  # type: ignore

        request_batch_to_generator_mock.assert_has_calls(
            [mock.call(step_name) for step_name in pipeline.dag.root_steps]
        )

    def test_send_batch_to_step(self, dummy_generator_step: "GeneratorStep") -> None:
        pipeline = dummy_generator_step.pipeline

        input_queue = mock.MagicMock()
        step = mock.MagicMock()
        step.__getitem__.return_value = input_queue
        get_step_mock = mock.MagicMock(return_value=step)
        pipeline.dag.get_step = get_step_mock  # type: ignore

        batch = _Batch(step_name="invented", last_batch=False, data=[[]])
        pipeline._send_batch_to_step(step_name=dummy_generator_step.name, batch=batch)  # type: ignore

        get_step_mock.assert_called_once_with(dummy_generator_step.name)
        input_queue.put.assert_called_once_with(
            _Batch(step_name=dummy_generator_step.name, last_batch=False, data=[[]])
        )

    def test_request_batch_to_generator(
        self, dummy_generator_step: "GeneratorStep"
    ) -> None:
        pipeline = dummy_generator_step.pipeline
        send_batch_to_step_mock = mock.MagicMock()
        pipeline._send_batch_to_step = send_batch_to_step_mock  # type: ignore

        pipeline._request_batch_to_generator(step_name=dummy_generator_step.name)  # type: ignore

        send_batch_to_step_mock.assert_called_once_with(
            dummy_generator_step.name,
            _Batch(step_name=dummy_generator_step.name, last_batch=False),
        )

    @mock.patch("distilabel.pipeline.local._ProcessWrapper")
    def test_create_processes(self, process_wrapper_mock: mock.MagicMock) -> None:
        pool = mock.MagicMock()
        manager = mock.MagicMock()
        queue = mock.MagicMock()

        with Pipeline() as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_step_1.connect(dummy_step_2)

        pipeline._run_steps_in_loop(pool, manager, queue)

        assert manager.Queue.call_count == 3

        process_wrapper_mock.assert_has_calls(
            [
                mock.call(
                    step=dummy_generator,
                    input_queue=mock.ANY,
                    output_queue=queue,
                ),
                mock.call(
                    step=dummy_step_1,
                    input_queue=mock.ANY,
                    output_queue=queue,
                ),
                mock.call(
                    step=dummy_step_2,
                    input_queue=mock.ANY,
                    output_queue=queue,
                ),
            ],
        )

        pool.apply_async.assert_has_calls(
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
