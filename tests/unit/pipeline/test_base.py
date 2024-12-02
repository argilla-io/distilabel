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

import logging
import os
import tempfile
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

import pytest
from datasets import Dataset
from fsspec.implementations.local import LocalFileSystem
from pydantic import Field
from upath import UPath

from distilabel import constants
from distilabel.constants import (
    INPUT_QUEUE_ATTR_NAME,
    LAST_BATCH_SENT_FLAG,
    STEPS_ARTIFACTS_PATH,
)
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.base import (
    _STEP_LOAD_FAILED_CODE,
    _STEP_NOT_LOADED_CODE,
    BasePipeline,
    _GlobalPipelineManager,
)
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.batch_manager import _BatchManager
from distilabel.pipeline.routing_batch_function import (
    routing_batch_function,
    sample_n_steps,
)
from distilabel.pipeline.write_buffer import _WriteBuffer
from distilabel.steps.base import Step, StepInput, StepResources, _Step
from distilabel.steps.typing import StepOutput
from distilabel.utils.requirements import requirements
from distilabel.utils.serialization import TYPE_INFO_KEY

from .utils import (
    DummyGeneratorStep,
    DummyGlobalStep,
    DummyStep1,
    DummyStep2,
)


class DummyPipeline(BasePipeline):
    @property
    def QueueClass(self) -> Callable:
        return Queue

    def _run_step(self, step: "_Step", input_queue: "Queue[Any]", replica: int) -> None:
        pass

    def _teardown(self) -> None:
        pass

    def _set_steps_not_loaded_exception(self) -> None:
        pass

    def _stop(self) -> None:
        pass


class TestGlobalPipelineManager:
    def teardown_method(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)

    def test_set_pipeline(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline

    def test_set_pipeline_none(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)
        assert _GlobalPipelineManager.get_pipeline() is None

    def test_get_pipeline(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline


class TestBasePipeline:
    def test_aggregated_steps_signature(self) -> None:
        with DummyPipeline(name="dummy") as pipeline_0:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        with DummyPipeline(name="dummy") as pipeline_1:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        assert (
            pipeline_0.aggregated_steps_signature
            == pipeline_1.aggregated_steps_signature
        )

    def test_context_manager(self) -> None:
        assert _GlobalPipelineManager.get_pipeline() is None

        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            assert pipeline is not None
            assert _GlobalPipelineManager.get_pipeline() == pipeline

        assert _GlobalPipelineManager.get_pipeline() is None

    def test_add_dataset_generator_step(self) -> None:
        with DummyPipeline() as pipeline:
            step_1 = DummyStep1()

        dataset = Dataset.from_list(
            [{"instruction": "Hello"}, {"instruction": "Hello again"}]
        )
        pipeline._add_dataset_generator_step(dataset, 123)
        step = pipeline.dag.get_step("load_data_from_hub_0")[constants.STEP_ATTR_NAME]

        assert step.name in pipeline.dag.get_step_predecessors(step_1.name)  # type: ignore
        assert step.batch_size == 123  # type: ignore

    @pytest.mark.parametrize("use_cache", [False, True])
    def test_load_batch_manager(self, use_cache: bool) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DummyPipeline(name="unit-test-pipeline", cache_dir=temp_dir)
            pipeline._load_batch_manager(use_cache=True)
            pipeline._cache()

            with (
                mock.patch(
                    "distilabel.pipeline.base._BatchManager.load_from_cache"
                ) as mock_load_from_cache,
                mock.patch(
                    "distilabel.pipeline.base._BatchManager.from_dag"
                ) as mock_from_dag,
            ):
                pipeline._load_batch_manager(use_cache=use_cache)

            if use_cache:
                mock_load_from_cache.assert_called_once_with(
                    dag=pipeline.dag,
                    batch_manager_path=pipeline._cache_location["batch_manager"],
                    steps_data_path=pipeline._cache_location["steps_data"],
                )
                mock_from_dag.assert_not_called()
            else:
                mock_load_from_cache.assert_not_called()
                mock_from_dag.assert_called_once_with(
                    dag=pipeline.dag,
                    use_cache=use_cache,
                    steps_data_path=pipeline._cache_location["steps_data"],
                )

    def test_setup_write_buffer(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        pipeline._setup_write_buffer()
        assert isinstance(pipeline._write_buffer, _WriteBuffer)

    def test_setup_fsspec(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        with mock.patch("fsspec.filesystem") as mock_filesystem:
            pipeline._setup_fsspec({"path": "gcs://my-bucket", "extra": "stuff"})

        mock_filesystem.assert_called_once_with("gcs", **{"extra": "stuff"})

    def test_setup_fsspec_default(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")
        pipeline._setup_fsspec()

        assert isinstance(pipeline._fs, LocalFileSystem)
        assert (
            pipeline._storage_base_path
            == f"file://{pipeline._cache_location['batch_input_data']}"
        )

    def test_setup_fsspec_raises_value_error(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        with pytest.raises(ValueError, match="The 'path' key must be present"):
            pipeline._setup_fsspec({"key": "random"})

    def test_set_pipeline_artifacts_path_in_steps(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        pipeline._set_pipeline_artifacts_path_in_steps()

        artifacts_directory = pipeline._cache_location["data"] / STEPS_ARTIFACTS_PATH
        assert generator.artifacts_directory == artifacts_directory / generator.name  # type: ignore
        assert step.artifacts_directory == artifacts_directory / step.name  # type: ignore
        assert step2.artifacts_directory == artifacts_directory / step2.name  # type: ignore
        assert step3.artifacts_directory == artifacts_directory / step3.name  # type: ignore

    def test_init_steps_load_status(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        pipeline._init_steps_load_status()
        assert pipeline._steps_load_status == {
            generator.name: _STEP_NOT_LOADED_CODE,
            step.name: _STEP_NOT_LOADED_CODE,
            step2.name: _STEP_NOT_LOADED_CODE,
            step3.name: _STEP_NOT_LOADED_CODE,
        }

    def test_initialize_pipeline_execution(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        pipeline._current_stage = 0
        pipeline._run_stage_steps_and_wait = mock.MagicMock(return_value=True)
        pipeline._set_steps_not_loaded_exception = mock.MagicMock()
        pipeline._request_initial_batches = mock.MagicMock()

        pipeline._initialize_pipeline_execution()

        pipeline._run_stage_steps_and_wait.assert_called_once_with(
            stage=pipeline._current_stage
        )
        pipeline._set_steps_not_loaded_exception.assert_not_called()
        pipeline._request_initial_batches.assert_called_once()

    def test_should_continue_processing(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            pass

        pipeline._batch_manager = mock.MagicMock()
        pipeline._stop_called = False
        pipeline._batch_manager.can_generate.return_value = True

        assert pipeline._should_continue_processing()

        pipeline._batch_manager.can_generate.return_value = False

        assert not pipeline._should_continue_processing()

    def test_set_step_for_recovering_offline_batch_generation(self) -> None:
        with DummyPipeline() as pipeline:
            step = DummyStep1()

        data = [[{"a": 0}, {"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]]
        pipeline._set_step_for_recovering_offline_batch_generation(step=step, data=data)

        assert pipeline._recover_offline_batch_generate_for_step == (step.name, data)

    def test_should_load_next_stage(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()
            global_step = DummyGlobalStep()

            generator >> [step, step2] >> step3 >> global_step

        pipeline._current_stage = 0
        pipeline._stages_last_batch = [[], []]

        assert not pipeline._should_load_next_stage()

        pipeline._stages_last_batch = [[step3.name], []]  # type: ignore

        assert pipeline._should_load_next_stage()

        pipeline._current_stage = 1
        pipeline._stages_last_batch = [  # type: ignore
            [step3.name],
            [global_step.name],
        ]

        assert not pipeline._should_load_next_stage()

    def test_update_stage(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            pass

        pipeline._current_stage = 0
        pipeline._should_load_next_stage = mock.MagicMock(return_value=True)
        pipeline._run_stage_steps_and_wait = mock.MagicMock(return_value=True)

        pipeline._update_stage()

        assert pipeline._current_stage == 1
        pipeline._run_stage_steps_and_wait.assert_called_once_with(
            stage=pipeline._current_stage
        )

    def test_run_load_queue_loop(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        pipeline._load_queue = Queue()
        pipeline._steps_load_status = {"dummy": 0}
        pipeline._load_queue.put({"name": "dummy", "status": "loaded"})

        thread = pipeline._run_load_queue_loop_in_thread()
        pipeline._load_queue.put(None)
        thread.join()

        assert pipeline._steps_load_status["dummy"] == 1

    def test_run_load_queue_loop_receiving_none(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        pipeline._load_queue = Queue()
        pipeline._load_queue.put(None)

        thread = pipeline._run_load_queue_loop_in_thread()
        thread.join()

        assert not thread.is_alive()

    def test_is_step_running(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

        pipeline._steps_load_status = {  # type: ignore
            generator.name: 1,
            step.name: 0,
        }

        assert pipeline._is_step_running(generator.name)  # type: ignore
        assert not pipeline._is_step_running(step.name)  # type: ignore

    def test_run_stage_steps_and_wait(self, caplog) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()
            step4 = DummyGlobalStep()

            generator >> [step, step2] >> step3 >> step4

        pipeline._load_batch_manager()
        pipeline._create_steps_input_queues()
        pipeline._steps_load_status = {  # type: ignore
            generator.name: 1,
            step.name: 1,
            step2.name: 1,
            step3.name: 1,
            step4.name: -999,
        }
        caplog.set_level(logging.INFO)

        assert pipeline._run_stage_steps_and_wait(stage=0) is True
        assert "All the steps from stage 0 have been loaded!" in caplog.text

    def test_run_stage_steps_and_wait_with_failing_step(self, caplog) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()
            step4 = DummyGlobalStep()

            generator >> [step, step2] >> step3 >> step4

        pipeline._init_steps_load_status()
        pipeline._load_batch_manager()
        pipeline._create_steps_input_queues()
        pipeline._steps_load_status[generator.name] = _STEP_LOAD_FAILED_CODE  # type: ignore
        caplog.set_level(logging.INFO)

        assert pipeline._run_stage_steps_and_wait(stage=0) is False
        assert "Failed to load all the steps of stage 0" in caplog.text

    def test_run_stage_steps_and_wait_stop_called(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()
            step4 = DummyGlobalStep()

            generator >> [step, step2] >> step3 >> step4

        pipeline._init_steps_load_status()
        pipeline._load_batch_manager()
        pipeline._create_steps_input_queues()
        pipeline._stop_called = True

        assert pipeline._run_stage_steps_and_wait(stage=0) is False

    def test_handle_stop(self) -> None:
        with DummyPipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        pipeline._add_batches_back_to_batch_manager = mock.MagicMock()
        pipeline._wait_step_input_queue_empty = mock.MagicMock()
        pipeline._consume_output_queue = mock.MagicMock()
        pipeline._stages_last_batch = [[]]

        pipeline._handle_stop()

        pipeline._add_batches_back_to_batch_manager.assert_called_once()
        pipeline._wait_step_input_queue_empty.assert_has_calls(
            [
                mock.call(generator.name),
                mock.call(step.name),
                mock.call(step2.name),
                mock.call(step3.name),
            ],
            any_order=True,
        )
        pipeline._consume_output_queue.assert_called_once()

    @pytest.mark.parametrize(
        "num_workers,expected", [(0, True), (_STEP_LOAD_FAILED_CODE, True), (1, False)]
    )
    def test_check_step_not_loaded_or_finished(
        self, num_workers: int, expected: bool
    ) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")
        pipeline._steps_load_status = {"dummy": num_workers}

        assert pipeline._check_step_not_loaded_or_finished("dummy") is expected

    def test_is_convergence_step(self) -> None:
        sample_two_steps = sample_n_steps(2)

        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> sample_two_steps >> [step, step2] >> step3

        pipeline.dag.validate()

        assert not pipeline._is_convergence_step(generator.name)  # type: ignore
        assert not pipeline._is_convergence_step(step.name)  # type: ignore
        assert not pipeline._is_convergence_step(step2.name)  # type: ignore
        assert pipeline._is_convergence_step(step3.name)  # type: ignore

    def test_create_step_input_queue(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        generator_name: str = generator.name  # type: ignore
        input_queue = pipeline._create_step_input_queue(generator_name)
        assert isinstance(input_queue, Queue)
        assert isinstance(
            pipeline.dag.get_step(generator_name)[INPUT_QUEUE_ATTR_NAME], Queue
        )

    def test_create_steps_input_queues(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            steps = [DummyStep1() for _ in range(5)]

            generator >> steps

        pipeline._create_steps_input_queues()
        assert len(pipeline._steps_input_queues) == 6

    def test_run_steps(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1(resources=StepResources(replicas=2))
            global_step = DummyGlobalStep()

            generator >> step >> global_step

        pipeline._run_step = mock.MagicMock()
        pipeline._create_steps_input_queues()
        pipeline._run_steps(steps=[generator.name, step.name])  # type: ignore

        pipeline._run_step.assert_has_calls(
            [
                mock.call(step=mock.ANY, input_queue=mock.ANY, replica=0),
                mock.call(step=mock.ANY, input_queue=mock.ANY, replica=0),
                mock.call(step=mock.ANY, input_queue=mock.ANY, replica=1),
            ]
        )

    def test_add_batches_back_to_batch_manager(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        generator_name: str = generator.name  # type: ignore
        step_name: str = step.name  # type: ignore

        pipeline._batch_manager = _BatchManager.from_dag(pipeline.dag)
        generator_queue = Queue()
        pipeline.dag.set_step_attr(
            generator_name, INPUT_QUEUE_ATTR_NAME, generator_queue
        )
        step_queue = Queue()
        pipeline.dag.set_step_attr(step_name, INPUT_QUEUE_ATTR_NAME, step_queue)

        generator_queue.put(
            _Batch(seq_no=0, step_name=generator_name, last_batch=False)
        )
        generator_queue.put(
            _Batch(seq_no=1, step_name=generator_name, last_batch=False)
        )

        step_batch_0 = _Batch(seq_no=0, step_name=step_name, last_batch=False)
        step_batch_1 = _Batch(seq_no=0, step_name=step_name, last_batch=False)
        step_queue.put(step_batch_0)
        step_queue.put(step_batch_1)

        pipeline._add_batches_back_to_batch_manager()

        assert pipeline._batch_manager._steps[step_name].built_batches == [
            step_batch_0,
            step_batch_1,
        ]

    def test_consume_output_queue(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        pipeline._output_queue = Queue()
        pipeline._write_buffer = mock.MagicMock()
        pipeline._handle_batch_on_stop = mock.MagicMock()

        generator_name: str = generator.name  # type: ignore
        step_name: str = step.name  # type: ignore

        generator_batch = _Batch(seq_no=0, step_name=generator_name, last_batch=False)
        step_batch = _Batch(seq_no=0, step_name=step_name, last_batch=False)

        pipeline._output_queue.put(generator_batch)
        pipeline._output_queue.put(step_batch)

        pipeline._consume_output_queue()

        pipeline._write_buffer.add_batch.assert_called_once_with(step_batch)
        pipeline._handle_batch_on_stop.assert_has_calls(
            [
                mock.call(generator_batch),
                mock.call(step_batch),
            ]
        )

    def test_send_batch_to_step(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            global_step = DummyGlobalStep()

            generator >> [step, global_step]

        pipeline._batch_manager = mock.MagicMock()
        pipeline._send_to_step = mock.MagicMock()
        pipeline._setup_fsspec()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            batch = _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
            pipeline._send_batch_to_step(batch)
            pipeline._batch_manager.set_last_batch_sent.assert_called_once_with(batch)

            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
            )

        # `write_batch_data_to_fs` shouldn't have been called because last batch sent with
        # `_send_batch_to_step` is from a non-global step.
        mock_write.assert_not_called()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=global_step.name, last_batch=False)  # type: ignore
            )

        # `write_batch_data_to_fs` should have been called because last batch sent with
        # `_send_batch_to_step` is from a global step.
        mock_write.assert_called_once_with(
            pipeline._fs,
            UPath(pipeline._storage_base_path) / global_step.name,
        )

        pipeline._use_fs_to_pass_data = True

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
            )

        # `write_batch_data_to_fs` shouldn't have been called because generator receives
        # empty batches, so there's no data to write.
        mock_write.assert_not_called()

        with mock.patch(
            "distilabel.pipeline.base._Batch.write_batch_data_to_fs"
        ) as mock_write:
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
            )
            pipeline._send_batch_to_step(
                _Batch(seq_no=0, step_name=global_step.name, last_batch=False)  # type: ignore
            )

        mock_write.assert_has_calls(
            [
                mock.call(
                    pipeline._fs,
                    UPath(pipeline._storage_base_path) / step.name,
                ),
                mock.call(
                    pipeline._fs,
                    UPath(pipeline._storage_base_path) / global_step.name,
                ),
            ]
        )

    def test_register_batch(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        pipeline._batch_manager = mock.MagicMock()
        batch = _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
        pipeline._register_batch(batch)

        pipeline._batch_manager.register_batch.assert_called_once_with(
            batch, steps_data_path=pipeline._cache_location["steps_data"]
        )

    def test_send_last_batch_flag_to_step(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        step_name: str = step.name  # type: ignore

        pipeline._batch_manager = _BatchManager(
            steps={},
            last_batch_received={step_name: None},
            last_batch_sent={step_name: None},
            last_batch_flag_sent_to=[],
        )

        with mock.patch.object(pipeline, "_send_to_step") as mock_sent_to_step:
            pipeline._send_last_batch_flag_to_step(step_name)

        mock_sent_to_step.assert_called_once_with(step_name, LAST_BATCH_SENT_FLAG)

    def test_request_initial_batches(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1(input_batch_size=5)

            generator >> step

            generator2 = DummyGeneratorStep()
            step2 = DummyStep1(input_batch_size=5)

            generator2 >> step2

        pipeline._batch_manager = _BatchManager.from_dag(pipeline.dag)
        pipeline._steps_load_status = {  # type: ignore
            generator.name: 1,
            step.name: 1,
            generator2.name: 1,
            step2.name: 1,
        }

        # Simulate there were batches from the cache for the steps
        batch_0 = _Batch(
            seq_no=0,
            step_name=generator.name,  # type: ignore
            last_batch=False,
            data=[[{"a": i} for i in range(5)]],
        )
        pipeline._batch_manager._steps[step.name].data[generator.name] = [  # type: ignore
            batch_0
        ]

        batch_1 = _Batch(
            seq_no=0,
            step_name=generator2.name,  # type: ignore
            last_batch=False,
            data=[[{"b": i} for i in range(5)]],
        )  # type: ignore
        pipeline._batch_manager._steps[step2.name].data[generator2.name] = [  # type: ignore
            batch_1
        ]

        with mock.patch.object(
            pipeline, "_send_batch_to_step"
        ) as mock_send_batch_to_step:
            pipeline._request_initial_batches()

        mock_send_batch_to_step.assert_has_calls(
            [
                mock.call(mock.ANY),
                mock.call(mock.ANY),
                mock.call(_Batch(seq_no=0, step_name=generator.name, last_batch=False)),  # type: ignore
                mock.call(
                    _Batch(seq_no=0, step_name=generator2.name, last_batch=False)  # type: ignore
                ),
            ],
            any_order=True,
        )

    def test_request_more_batches_if_needed(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        generator_name: str = generator.name  # type: ignore

        pipeline._batch_manager = _BatchManager.from_dag(pipeline.dag)

        batch = _Batch(seq_no=0, step_name=generator_name, last_batch=False)
        pipeline._batch_manager._last_batch_sent[generator_name] = batch

        with mock.patch.object(
            pipeline, "_send_batch_to_step"
        ) as mock_send_batch_to_step:
            pipeline._request_more_batches_if_needed(step)

        mock_send_batch_to_step.assert_called_once_with(batch.next_batch())

    def test_handle_batch_on_stop(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1(input_batch_size=5)
            step2 = DummyStep1(input_batch_size=5)
            step3 = DummyStep1(input_batch_size=5)

            generator >> [step, step2, step3]

        batch_manager_mock = mock.MagicMock()
        pipeline._batch_manager = batch_manager_mock

        batch = _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
        pipeline._handle_batch_on_stop(batch)

        batch_manager_mock.register_batch.assert_called_once_with(
            batch, steps_data_path=pipeline._cache_location["steps_data"]
        )
        batch_manager_mock.add_batch.assert_has_calls(
            [
                mock.call(step.name, batch),
                mock.call(step2.name, batch),
                mock.call(step3.name, batch),
            ]
        )

    def test_get_step_from_batch(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()

            generator >> step

        batch = _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
        assert pipeline._get_step_from_batch(batch) == generator

        batch = _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
        assert pipeline._get_step_from_batch(batch) == step

    def test_notify_steps_to_stop(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1(input_batch_size=5)
            global_step = DummyGlobalStep()

            generator >> step >> global_step

        pipeline._steps_load_status = {  # type: ignore
            generator.name: 1,
            step.name: 1,
            global_step.name: -999,
        }

        with mock.patch.object(pipeline, "_send_to_step") as mock_send_to_step:
            pipeline._notify_steps_to_stop()

        mock_send_to_step.assert_has_calls(
            [
                mock.call(generator.name, None),
                mock.call(step.name, None),
            ]
        )

    def test_get_successors(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1()
            step2 = DummyStep1()
            step3 = DummyStep2()

            generator >> [step, step2] >> step3

        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
        ) == ([step.name, step2.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
        ) == ([step3.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step2.name, last_batch=False)  # type: ignore
        ) == ([step3.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step3.name, last_batch=False)  # type: ignore
        ) == ([], [], False)

    def test_get_successors_with_routing_batch_function(self) -> None:
        @routing_batch_function()
        def fixed_routing_batch_function(steps: List[str]) -> List[str]:
            return ["step_2", "step_3"]

        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep()
            step = DummyStep1(name="step_1")
            step2 = DummyStep1(name="step_2")
            step3 = DummyStep1(name="step_3")
            step4 = DummyStep2(name="step_4")

            generator >> fixed_routing_batch_function >> [step, step2, step3] >> step4

        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=generator.name, last_batch=False)  # type: ignore
        ) == (["step_2", "step_3"], ["step_1"], True)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step.name, last_batch=False)  # type: ignore
        ) == ([step4.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step2.name, last_batch=False)  # type: ignore
        ) == ([step4.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step3.name, last_batch=False)  # type: ignore
        ) == ([step4.name], [], False)
        assert pipeline._get_successors(
            _Batch(seq_no=0, step_name=step4.name, last_batch=False)  # type: ignore
        ) == ([], [], False)

    def test_set_next_expected_seq_no(self) -> None:
        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            generator = DummyGeneratorStep(name="generator")
            step = DummyStep1(name="step_1")
            step2 = DummyStep1(name="step_2")
            step3 = DummyStep1(name="step_3")
            step4 = DummyStep2(name="step_4")

            generator >> [step, step2, step3] >> step4

        pipeline._batch_manager = mock.MagicMock()

        pipeline._set_next_expected_seq_no(
            steps=["step_1", "step_2"], from_step="generator", next_expected_seq_no=666
        )

        pipeline._batch_manager.set_next_expected_seq_no.assert_has_calls(
            [
                mock.call(
                    step_name="step_1", from_step="generator", next_expected_seq_no=666
                ),
                mock.call(
                    step_name="step_2", from_step="generator", next_expected_seq_no=666
                ),
            ]
        )

    def test_get_runtime_parameters_info(self) -> None:
        class DummyStep1(Step):
            runtime_param1: RuntimeParameter[str] = Field(
                default=None, description="runtime_param1 description"
            )
            runtime_param2: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param2 description"
            )

            def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
                pass

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
                pass

        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            DummyStep1(name="dummy_step_1")
            DummyStep2(name="dummy_step_2")

        assert pipeline.get_runtime_parameters_info() == {
            "dummy_step_1": [
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "description": "The number of replicas for the step.",
                            "name": "replicas",
                            "optional": True,
                        },
                        {
                            "description": "The number of CPUs assigned to each step replica.",
                            "name": "cpus",
                            "optional": True,
                        },
                        {
                            "description": "The number of GPUs assigned to each step replica.",
                            "name": "gpus",
                            "optional": True,
                        },
                        {
                            "description": "The memory in bytes required for each step replica.",
                            "name": "memory",
                            "optional": True,
                        },
                        {
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                            "name": "resources",
                            "optional": True,
                        },
                    ],
                },
                {
                    "description": "The number of rows that will contain the batches processed by the "
                    "step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "runtime_param1",
                    "description": "runtime_param1 description",
                    "optional": False,
                },
                {
                    "name": "runtime_param2",
                    "description": "runtime_param2 description",
                    "optional": True,
                },
            ],
            "dummy_step_2": [
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "description": "The number of replicas for the step.",
                            "name": "replicas",
                            "optional": True,
                        },
                        {
                            "description": "The number of CPUs assigned to each step replica.",
                            "name": "cpus",
                            "optional": True,
                        },
                        {
                            "description": "The number of GPUs assigned to each step replica.",
                            "name": "gpus",
                            "optional": True,
                        },
                        {
                            "description": "The memory in bytes required for each step replica.",
                            "name": "memory",
                            "optional": True,
                        },
                        {
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                            "name": "resources",
                            "optional": True,
                        },
                    ],
                },
                {
                    "description": "The number of rows that will contain the batches processed by the "
                    "step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "name": "runtime_param3",
                    "description": "runtime_param3 description",
                    "optional": False,
                },
                {
                    "name": "runtime_param4",
                    "description": "runtime_param4 description",
                    "optional": True,
                },
            ],
        }

    # Test no log, Test log, test log without close match
    @pytest.mark.parametrize(
        "parameters, expected",
        (
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {"runtime_param3": "value1"},
                },
                "",
            ),
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {
                        "runtime_param3": "value1",
                        "runtime_param_unknown": "value1",
                    },
                },
                "Did you mean any of:",
            ),
            (
                {
                    "dummy_step_1": {"runtime_param1": "value1"},
                    "dummy_step_2": {
                        "runtime_param3": "value1",
                        "weird_name": "value1",
                    },
                },
                "Available runtime parameters for the step",
            ),
        ),
    )
    def test_check_runtime_parameters(
        self, caplog, parameters: Dict[str, Any], expected: str
    ) -> None:
        class DummyStep1(Step):
            runtime_param1: RuntimeParameter[str] = Field(
                default=None, description="runtime_param1 description"
            )
            runtime_param2: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param2 description"
            )

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                yield [{}]

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                yield [{}]

        with DummyPipeline(name="unit-test-pipeline") as pipeline:
            gen_step = DummyGeneratorStep(name="dummy_generator_step")
            step1 = DummyStep1(name="dummy_step_1")
            step2 = DummyStep2(name="dummy_step_2")

            gen_step >> step1 >> step2

        pipeline.run(parameters=parameters)
        if expected:
            assert expected in caplog.text
        else:
            assert "Did you mean any of:" not in expected
            assert "Available runtime parameters for the step" not in expected

    def test_cache_dir_env_variable(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["DISTILABEL_CACHE_DIR"] = "/tmp/unit-test"
            pipeline = DummyPipeline(name="unit-test-pipeline")
            assert pipeline._cache_dir == Path("/tmp/unit-test")

    @pytest.mark.parametrize(
        "in_pipeline, names",
        (
            (
                True,
                [
                    "dummy_generator_step_0",
                    "dummy_step1_0",
                    "dummy_step2_0",
                    "dummy_step1_1",
                ],
            ),
            # TODO: Activate this test once we merge the option of not passing a Pipeline
            # (
            #     False, ["dummy_generator_step", "dummy_step1", "dummy_step2"]
            # )
        ),
    )
    def test_step_names_inferred(self, in_pipeline: bool, names: List[str]) -> None:
        if in_pipeline:
            with DummyPipeline(name="unit-test-pipeline"):
                gen_step = DummyGeneratorStep()
                step1_0 = DummyStep1()
                step2 = DummyStep2()
                step1_1 = DummyStep1()

                gen_step >> step1_0 >> step2 >> step1_1
        else:
            gen_step = DummyGeneratorStep()
            step1_0 = DummyStep1()
            step2 = DummyStep2()
            step1_1 = DummyStep1()

        assert gen_step.name == names[0]
        assert step1_0.name == names[1]
        assert step2.name == names[2]
        assert step1_1.name == names[3]

    def test_infer_step_names_big_pipeline(self) -> None:
        # Tests that the name of the steps are inferred correctly when the pipeline is big (say 50 steps).
        with DummyPipeline(name="unit-test-pipeline") as pipe:
            gen_step = DummyGeneratorStep()
            for _ in range(50):
                gen_step.connect(DummyStep1())
        assert list(pipe.dag.G)[-1] == "dummy_step1_49"

    @pytest.mark.parametrize(
        "requirements, expected",
        [
            (None, []),
            (["pandas", "numpy"], ["numpy", "pandas"]),
            (["*`wrong", "pandas>1.0"], ["pandas>1.0"]),
            (["pandas", "pandas", "pandas"], ["pandas"]),
        ],
    )
    def test_requirements(self, requirements: List[str], expected: List[str]) -> None:
        with DummyPipeline(
            name="unit-test-pipeline", requirements=requirements
        ) as pipeline:
            assert pipeline.requirements == expected

    @pytest.mark.parametrize(
        "requirements, expected",
        [
            (None, []),
            (["distilabel"], []),
            (["distilabel", "yfinance"], ["yfinance"]),
            (
                ["distilabel>=3000", "yfinance==1.0.0"],
                ["distilabel>=3000", "yfinance==1.0.0"],
            ),
        ],
    )
    def test_requirements_to_install(
        self, requirements: List[str], expected: List[str]
    ) -> None:
        with DummyPipeline(
            name="unit-test-pipeline", requirements=requirements
        ) as pipeline:
            assert pipeline.requirements_to_install() == expected

    def test_pipeline_error_from_requirements(self):
        @requirements(["distilabel>=0.0.1"])
        class CustomStep(Step):
            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
                for input in inputs:
                    input["response"] = "unit test"
                yield inputs

        with pytest.raises(
            ModuleNotFoundError,
            match=r"Please install the following requirements to run the pipeline: \ndistilabel>=0.0.1\nrandom_requirement",
        ):
            with DummyPipeline(
                name="unit-test-pipeline", requirements=["random_requirement"]
            ) as pipeline:
                gen_step = DummyGeneratorStep()
                step1_0 = DummyStep1()
                step2 = CustomStep()

                gen_step >> step1_0 >> step2
            pipeline.run()

    def test_pipeline_with_dataset_and_generator_step(self):
        with pytest.raises(ValueError) as exc_info:
            with DummyPipeline(name="unit-test-pipeline") as pipeline:
                gen_step = DummyGeneratorStep()
                step1_0 = DummyStep1()
                gen_step >> step1_0

            pipeline.run(
                use_cache=False, dataset=[{"instruction": "Tell me a joke."}] * 10
            )
            exc_info.value.args[0].startswith(
                "There is already a `GeneratorStep` in the pipeline"
            )

    def test_optional_name(self):
        from distilabel.pipeline.base import _PIPELINE_DEFAULT_NAME

        assert DummyPipeline().name == _PIPELINE_DEFAULT_NAME

        with DummyPipeline() as pipeline:
            gen_step = DummyGeneratorStep()
            step1_0 = DummyStep1()
            gen_step >> step1_0

        assert pipeline.name == "pipeline_dummy_generator_step_0_dummy_step1_0"

    def test_validate_load_groups_step_not_in_pipeline(self) -> None:
        pipeline = DummyPipeline()

        with pytest.raises(
            ValueError,
            match="Step with name 'random' included in group 0 of the `load_groups` is not an step included in the pipeline.",
        ):
            pipeline._validate_load_groups(load_groups=[["random"]])

    def test_validate_load_groups_including_global_step(self) -> None:
        pipeline = DummyPipeline()
        step = DummyGlobalStep(pipeline=pipeline)
        with pytest.raises(
            ValueError,
            match=f"Global step '{step.name}' has been included in a load group.",
        ):
            pipeline._validate_load_groups(load_groups=[[step.name]])

    def test_validate_load_groups_duplicate_step(self) -> None:
        pipeline = DummyPipeline()
        dummy_step_1 = DummyStep1(pipeline=pipeline)

        with pytest.raises(
            ValueError,
            match=f"Step with name '{dummy_step_1.name}' in load group 1 has been already included in a previous load group.",
        ):
            pipeline._validate_load_groups(
                load_groups=[[dummy_step_1.name], [dummy_step_1.name]]
            )

    def test_validate_load_groups_non_immediate_predecessor(self) -> None:
        pipeline = DummyPipeline()
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)
        dummy_step_2 = DummyStep1(name="demon", pipeline=pipeline, input_batch_size=7)

        generator_step_1 >> dummy_step_1 >> dummy_step_2

        with pytest.raises(
            ValueError,
            match=f"Step with name '{dummy_step_2.name}' cannot be in the same load group as the step with name '{generator_step_1.name}'.",
        ):
            pipeline._validate_load_groups(
                load_groups=[[generator_step_1.name, dummy_step_2.name]]
            )


class TestPipelineSerialization:
    @pytest.mark.parametrize(
        "requirements, expected",
        [
            (None, []),
            (["distilabel>=0.0.1"], ["distilabel>=0.0.1"]),
        ],
    )
    def test_base_pipeline_dump(
        self, requirements: Optional[List[str]], expected: List[str]
    ):
        pipeline = DummyPipeline(name="unit-test-pipeline", requirements=requirements)
        dump = pipeline.dump()
        assert len(dump.keys()) == 3
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert "requirements" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert (
            dump["pipeline"][TYPE_INFO_KEY]["module"] == "tests.unit.pipeline.test_base"
        )
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "DummyPipeline"
        assert dump["requirements"] == expected

    @pytest.mark.parametrize(
        "requirements",
        [
            None,
            ["distilabel>=0.0.1"],
        ],
    )
    def test_base_pipeline_from_dict(self, requirements: Optional[List[str]]):
        pipeline = DummyPipeline(name="unit-test-pipeline", requirements=requirements)
        pipe = DummyPipeline.from_dict(pipeline.dump())
        assert isinstance(pipe, DummyPipeline)

    @pytest.mark.parametrize(
        "requirements, expected",
        [
            (None, []),
            (["distilabel>=0.0.1"], ["distilabel>=0.0.1"]),
        ],
    )
    def test_pipeline_dump(
        self, requirements: Optional[List[str]], expected: List[str]
    ):
        from distilabel.pipeline.local import Pipeline

        pipeline = Pipeline(name="unit-test-pipeline", requirements=requirements)
        dump = pipeline.dump()
        assert len(dump.keys()) == 3
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert "requirements" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert dump["pipeline"][TYPE_INFO_KEY]["module"] == "distilabel.pipeline.local"
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "Pipeline"
        assert dump["requirements"] == expected

    @pytest.mark.parametrize(
        "format, name, loader",
        [
            ("yaml", "pipe.yaml", DummyPipeline.from_yaml),
            ("json", "pipe.json", DummyPipeline.from_json),
            ("invalid", "pipe.invalid", None),
        ],
    )
    def test_pipeline_to_from_file_format(
        self,
        format: str,
        name: str,
        loader: Callable,
    ) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / name
            if format == "invalid":
                with pytest.raises(ValueError):
                    pipeline.save(filename, format=format)
            else:
                pipeline.save(filename, format=format)
                assert filename.exists()
                pipe_from_file = loader(filename)
                assert isinstance(pipe_from_file, DummyPipeline)

    def test_base_pipeline_signature(self) -> None:
        pipeline = DummyPipeline(name="unit-test-pipeline")
        # Doesn't matter if it's exactly this or not, the test should fail if we change the
        # way this is created.
        assert pipeline.signature == "da39a3ee5e6b4b0d3255bfef95601890afd80709"

        # Maybe not the best place for this test, but does the work for now
        from distilabel.pipeline.local import Pipeline
        from distilabel.pipeline.routing_batch_function import sample_n_steps
        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        sample_two_steps = sample_n_steps(2)

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator")
            dummy_step_1_0 = DummyStep1(name="dummy_step_1_0")
            dummy_step_1_1 = DummyStep1(name="dummy_step_1_1")
            dummy_step_1_2 = DummyStep1(name="dummy_step_1_2")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            (
                dummy_generator
                >> sample_two_steps
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        assert pipeline.signature == "edff8f5bb8b51da406ff274e640f87264f014e3b"

        # attributes shouldn't affect in pipeline signature
        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator")
            dummy_step_1_0 = DummyStep1(name="dummy_step_1_0", attr1=17238497128934)
            dummy_step_1_1 = DummyStep1(name="dummy_step_1_1")
            dummy_step_1_2 = DummyStep1(name="dummy_step_1_2")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            (
                dummy_generator
                >> sample_two_steps
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        assert pipeline.signature == "edff8f5bb8b51da406ff274e640f87264f014e3b"

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator")
            dummy_step_1_0 = DummyStep1(name="dummy_step_1_0")
            dummy_step_1_1 = DummyStep1(name="dummy_step_1_1")
            dummy_step_1_2 = DummyStep1(name="dummy_step_1_2")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            (
                dummy_generator
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        assert pipeline.signature == "5634172be496319d50848b1679b2a8781cc5581f"

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_second_time")
            dummy_step_1_0 = DummyStep1(
                name="dummy_step_1_0_second_time", attr1=17238497128934
            )
            dummy_step_1_1 = DummyStep1(name="dummy_step_1_1_second_time")
            dummy_step_1_2 = DummyStep1(name="dummy_step_1_2_second_time")
            dummy_step_2 = DummyStep2(name="dummy_step_2_second_time")

            (
                dummy_generator
                >> sample_two_steps
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        assert pipeline.signature == "806dad3fca0f8274af0f374660d4e3eb25d62d12"

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_second_time")
            dummy_step_1_0 = DummyStep1(
                name="dummy_step_1_0_second_time", attr1=17238497128934
            )
            dummy_step_1_1 = DummyStep1(name="dummy_step_1_1_second_time")

            (dummy_generator >> sample_two_steps >> [dummy_step_1_0, dummy_step_1_1])

        assert pipeline.signature == "7222ce34c677bea3720ef3d08c2673b29b61ff9b"

    def test_binary_rshift_operator(self) -> None:
        # Tests the steps can be connected using the >> operator.
        from distilabel.pipeline.local import Pipeline
        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_step_1.connect(dummy_step_2)

            signature_1 = pipeline_1.signature

        with Pipeline(name="unit-test-pipeline-3") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> dummy_step_1 >> dummy_step_2

            signature_2 = pipeline_2.signature

        assert signature_1 == signature_2

    def test_binary_rshift_operator_with_list(self) -> None:
        # Tests the steps can be connected using the >> operator when using a list.
        from distilabel.pipeline.local import Pipeline
        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator.connect(dummy_step_1)
            dummy_generator.connect(dummy_step_2)

            signature_1 = pipeline_1.signature

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> [dummy_step_1, dummy_step_2]

            signature_2 = pipeline_2.signature

        assert signature_1 == signature_2

    def test_binary_rrshift_operator(self) -> None:
        # Tests that a list of steps can be connected to a single step using the >> operator.
        # It usses the __rrshift__ method instead of the __rshift__ as it applies to the list
        # instead of the Step.

        from distilabel.pipeline.local import Pipeline
        from tests.unit.pipeline.utils import DummyGlobalStep, DummyStep1, DummyStep2

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_step_1.connect(dummy_global)
            dummy_step_2.connect(dummy_global)

            signature_1 = pipeline_1.signature

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2.signature

        assert signature_1 == signature_2

    def test_binary_operators(self) -> None:
        # Tests the steps can be connected with the binary operators,
        # the general case of step1 >> [step2, step3] >> step4
        from distilabel.pipeline.local import Pipeline
        from tests.unit.pipeline.utils import (
            DummyGeneratorStep,
            DummyGlobalStep,
            DummyStep1,
            DummyStep2,
        )

        with Pipeline(name="unit-test-pipeline-1") as pipeline_1:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_generator.connect(dummy_step_1)
            dummy_generator.connect(dummy_step_2)
            dummy_step_1.connect(dummy_global)
            dummy_step_2.connect(dummy_global)

            signature_1 = pipeline_1.signature

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_generator >> [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2.signature

        assert signature_1 == signature_2
