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

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from distilabel.pipeline.base import _Batch, _BatchManager
from distilabel.pipeline.local import Pipeline, _WriteBuffer
from distilabel.utils.data import Distiset, _create_dataset

from .utils import DummyGeneratorStep, DummyStep1, DummyStep2, batch_gen

if TYPE_CHECKING:
    from distilabel.steps.base import GeneratorStep


class TestPipeline:
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

        batch = _Batch(
            seq_no=0, step_name=dummy_generator_step.name, last_batch=False, data=[[]]
        )
        pipeline._send_batch_to_step(batch=batch)  # type: ignore

        get_step_mock.assert_called_once_with(dummy_generator_step.name)
        input_queue.put.assert_called_once_with(batch)

    @mock.patch("distilabel.pipeline.local._ProcessWrapper")
    def test_create_processes(self, process_wrapper_mock: mock.MagicMock) -> None:
        pool = mock.MagicMock()
        manager = mock.MagicMock()
        queue = mock.MagicMock()
        shared_info = mock.MagicMock()

        with Pipeline() as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_step_1.connect(dummy_step_2)

        pipeline._run_steps_in_loop(pool, manager, queue, shared_info)

        assert manager.Queue.call_count == 3

        process_wrapper_mock.assert_has_calls(
            [
                mock.call(
                    step=dummy_generator,
                    input_queue=mock.ANY,
                    output_queue=queue,
                    shared_info=shared_info,
                ),
                mock.call(
                    step=dummy_step_1,
                    input_queue=mock.ANY,
                    output_queue=queue,
                    shared_info=shared_info,
                ),
                mock.call(
                    step=dummy_step_2,
                    input_queue=mock.ANY,
                    output_queue=queue,
                    shared_info=shared_info,
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


class TestWriteBuffer:
    def test_write_buffer_one_leaf_step_and_create_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "data"
            with Pipeline() as pipeline:
                dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
                dummy_step_1 = DummyStep1(name="dummy_step_1")
                dummy_step_2 = DummyStep2(name="dummy_step_2")

                dummy_generator.connect(dummy_step_1)
                dummy_step_1.connect(dummy_step_2)

            write_buffer = _WriteBuffer(path=folder, leaf_steps=pipeline.dag.leaf_steps)
            batch = batch_gen(dummy_step_2.name)
            assert len(write_buffer._buffers) == 1

            assert all(values is None for _, values in write_buffer._buffers.items())

            write_buffer.add_batch(batch.step_name, batch)
            assert write_buffer._get_filename(batch.step_name).exists()
            write_buffer.close()

            ds = _create_dataset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 1
            assert len(ds["dummy_step_2"]) == 3

    def test_write_buffer_multiple_leaf_steps_and_create_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "data"
            with Pipeline() as pipeline:
                dummy_generator_1 = DummyGeneratorStep(name="dummy_generator_step_1")
                dummy_generator_2 = DummyGeneratorStep(name="dummy_generator_step_2")
                dummy_step_1 = DummyStep1(name="dummy_step_1")
                dummy_step_2 = DummyStep2(name="dummy_step_2")
                dummy_step_3 = DummyStep2(name="dummy_step_3")

                dummy_generator_1.connect(dummy_step_1)
                dummy_generator_2.connect(dummy_step_2)
                dummy_step_1.connect(dummy_step_2)
                dummy_step_1.connect(dummy_step_3)

            write_buffer = _WriteBuffer(path=folder, leaf_steps=pipeline.dag.leaf_steps)

            # Now we write here only in case we are working with leaf steps
            batch_step_2 = batch_gen(dummy_step_2.name, col_name="a")
            batch_step_3 = batch_gen(dummy_step_3.name, col_name="b")
            assert all(values is None for _, values in write_buffer._buffers.items())
            assert len(write_buffer._buffers) == 2

            write_buffer.add_batch(batch_step_2.step_name, batch_step_2)
            assert write_buffer._get_filename(batch_step_2.step_name).exists()
            assert not write_buffer._get_filename(batch_step_3.step_name).exists()
            write_buffer.add_batch(batch_step_3.step_name, batch_step_3)
            assert write_buffer._get_filename(batch_step_3.step_name).exists()
            write_buffer.close()

            ds = _create_dataset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 2
            assert len(ds["dummy_step_2"]) == 3
            assert len(ds["dummy_step_3"]) == 3
