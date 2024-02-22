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
from typing import TYPE_CHECKING, Callable

import pytest
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.base import (
    BasePipeline,
    _Batch,
    _BatchManager,
    _GlobalPipelineManager,
)

if TYPE_CHECKING:
    from distilabel.pipeline.step.base import GeneratorStep, Step


class TestGlobalPipelineManager:
    def teardown_method(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)

    def test_set_pipeline(self) -> None:
        pipeline = BasePipeline()
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline

    def test_set_pipeline_none(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)
        assert _GlobalPipelineManager.get_pipeline() is None

    def test_get_pipeline(self) -> None:
        pipeline = BasePipeline()
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline


class TestBasePipeline:
    def test_context_manager(self) -> None:
        assert _GlobalPipelineManager.get_pipeline() is None

        with BasePipeline() as pipeline:
            assert pipeline is not None
            assert _GlobalPipelineManager.get_pipeline() == pipeline

        assert _GlobalPipelineManager.get_pipeline() is None


class TestBatch:
    def test_next_batch(self) -> None:
        batch = _Batch(seq_no=0, step_name="step1", last_batch=False)
        next_batch = batch.next_batch()

        assert next_batch == _Batch(seq_no=1, step_name="step1", last_batch=False)

    def test_from_batches(self) -> None:
        batches = [
            _Batch(seq_no=0, step_name="step1", last_batch=False, data=[[]]),
            _Batch(seq_no=0, step_name="step2", last_batch=False, data=[[]]),
        ]
        batch = _Batch.from_batches(step_name="step3", batches=batches)
        assert batch == _Batch(
            seq_no=0, step_name="step3", last_batch=False, data=[[], []]
        )


class TestBatchManager:
    def test_add_batch(self) -> None:
        batch_manager = _BatchManager(
            batches={
                "step3": {
                    "step1": [],
                    "step2": [],
                }
            }
        )

        batch_from_step_1 = _Batch(
            seq_no=0, step_name="step1", last_batch=False, data=[[]]
        )
        batches = batch_manager.add_batch(to_step="step3", batch=batch_from_step_1)

        assert batches is None
        assert batch_manager._batches["step3"]["step1"] == [batch_from_step_1]

    def test_add_batch_all_batches_received(self) -> None:
        first_batch_from_step_2 = _Batch(
            seq_no=0, step_name="step2", last_batch=False, data=[[]]
        )
        batch_manager = _BatchManager(
            batches={
                "step3": {
                    "step1": [],
                    "step2": [first_batch_from_step_2],
                }
            }
        )

        batch_from_step_1 = _Batch(
            seq_no=0, step_name="step1", last_batch=False, data=[[]]
        )
        batches = batch_manager.add_batch(to_step="step3", batch=batch_from_step_1)

        assert batches == [batch_from_step_1, first_batch_from_step_2]
        assert batch_manager._batches["step3"]["step1"] == []
        assert batch_manager._batches["step3"]["step2"] == []

    def test_add_batch_for_step_with_batch_waiting(self) -> None:
        batch_manager = _BatchManager(
            batches={
                "step2": {
                    "step1": [
                        _Batch(seq_no=0, step_name="step1", last_batch=False, data=[[]])
                    ]
                }
            }
        )

        with pytest.raises(
            ValueError,
            match="A batch from 'step1' to 'step2' with sequence number 0 was already received",
        ):
            batch_manager.add_batch(
                to_step="step2",
                batch=_Batch(seq_no=0, step_name="step1", last_batch=False, data=[[]]),
            )

    def test_from_dag(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_step_1", "dummy_step_2")

        batch_manager = _BatchManager.from_dag(dag)

        assert batch_manager._batches == {
            "dummy_step_1": {"dummy_generator_step": []},
            "dummy_step_2": {"dummy_step_1": []},
        }

    def test_step_input_batches_received(self) -> None:
        batch_from_step_1 = _Batch(
            seq_no=0, step_name="step1", last_batch=False, data=[[]]
        )
        batch_from_step_2 = _Batch(
            seq_no=0, step_name="step2", last_batch=True, data=[[]]
        )
        batch_manager = _BatchManager(
            batches={"step3": {"step1": [batch_from_step_1], "step2": []}}
        )

        assert batch_manager._step_input_batches_received("step3") is False

        batch_manager._batches["step3"]["step2"] = [batch_from_step_2]

        assert batch_manager._step_input_batches_received("step3") is True


class TestPipelineSerialization:
    def test_base_pipeline_dump(self):
        pipeline = BasePipeline()
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "dag" in dump
        assert "_type_info_" in dump
        assert dump["_type_info_"]["module"] == "distilabel.pipeline.base"
        assert dump["_type_info_"]["name"] == "BasePipeline"

    def test_base_pipeline_from_dict(self):
        pipeline = BasePipeline()
        pipe = BasePipeline.from_dict(pipeline.dump())
        assert isinstance(pipe, BasePipeline)

    def test_pipeline_dump(self):
        from distilabel.pipeline.local import Pipeline

        pipeline = Pipeline()
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "dag" in dump
        assert "_type_info_" in dump
        assert dump["_type_info_"]["module"] == "distilabel.pipeline.local"
        assert dump["_type_info_"]["name"] == "Pipeline"

    @pytest.mark.parametrize(
        "format, name, loader",
        [
            ("yaml", "pipe.yaml", BasePipeline.from_yaml),
            ("json", "pipe.json", BasePipeline.from_json),
            ("invalid", "pipe.invalid", None),
        ],
    )
    def test_pipeline_to_from_file_format(
        self,
        format: str,
        name: str,
        loader: Callable,
    ) -> None:
        pipeline = BasePipeline()

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / name
            if format == "invalid":
                with pytest.raises(ValueError):
                    pipeline.save(filename, format=format)
            else:
                pipeline.save(filename, format=format)
                assert filename.exists()
                pipe_from_file = loader(filename)
                assert isinstance(pipe_from_file, BasePipeline)
