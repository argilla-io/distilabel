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

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from unittest import mock

import pytest
from distilabel.distiset import Distiset, create_distiset
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.base import (
    BasePipeline,
    _Batch,
    _BatchManager,
    _BatchManagerStep,
    _GlobalPipelineManager,
    _WriteBuffer,
)
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import GlobalStep, Step, StepInput
from distilabel.utils.serialization import TYPE_INFO_KEY
from pydantic import Field

from .utils import DummyGeneratorStep, DummyStep1, DummyStep2, batch_gen

if TYPE_CHECKING:
    from distilabel.steps.base import GeneratorStep


class TestGlobalPipelineManager:
    def teardown_method(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)

    def test_set_pipeline(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline

    def test_set_pipeline_none(self) -> None:
        _GlobalPipelineManager.set_pipeline(None)
        assert _GlobalPipelineManager.get_pipeline() is None

    def test_get_pipeline(self) -> None:
        pipeline = BasePipeline(name="unit-test-pipeline")
        _GlobalPipelineManager.set_pipeline(pipeline)
        assert _GlobalPipelineManager.get_pipeline() == pipeline


class TestBasePipeline:
    def test_context_manager(self) -> None:
        assert _GlobalPipelineManager.get_pipeline() is None

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            assert pipeline is not None
            assert _GlobalPipelineManager.get_pipeline() == pipeline

        assert _GlobalPipelineManager.get_pipeline() is None

    def test_get_runtime_parameters_info(self) -> None:
        class DummyStep1(Step):
            runtime_param1: RuntimeParameter[str] = Field(
                default=None, description="runtime_param1 description"
            )
            runtime_param2: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param2 description"
            )

            def process(self, inputs: StepInput) -> None:
                pass

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> None:
                pass

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            DummyStep1(name="dummy_step_1")
            DummyStep2(name="dummy_step_2")

        assert pipeline.get_runtime_parameters_info() == {
            "dummy_step_1": [
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

            def process(self, inputs: StepInput) -> None:
                pass

        class DummyStep2(Step):
            runtime_param3: RuntimeParameter[str] = Field(
                default=None, description="runtime_param3 description"
            )
            runtime_param4: Optional[RuntimeParameter[str]] = Field(
                default=None, description="runtime_param4 description"
            )

            def process(self, inputs: StepInput) -> None:
                pass

        with BasePipeline(name="unit-test-pipeline") as pipeline:
            gen_step = DummyGeneratorStep(name="dummy_generator_step")
            step1 = DummyStep1(name="dummy_step_1")
            step2 = DummyStep2(name="dummy_step_2")

            gen_step >> step1 >> step2

        pipeline.run(parameters=parameters)
        if expected:
            assert expected in caplog.text
        else:
            assert caplog.text == expected

    def test_cache_dir_env_variable(self) -> None:
        with mock.patch.dict(os.environ, clear=True):
            os.environ["DISTILABEL_CACHE_DIR"] = "/tmp/unit-test"
            pipeline = BasePipeline(name="unit-test-pipeline")
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
            with BasePipeline(name="unit-test-pipeline"):
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
        with BasePipeline(name="unit-test-pipeline") as pipe:
            gen_step = DummyGeneratorStep()
            for _ in range(50):
                gen_step.connect(DummyStep1())
        assert list(pipe.dag.G)[-1] == "dummy_step1_49"


class TestBatch:
    def test_get_data(self) -> None:
        batch = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[
                [
                    {"a": 0},
                    {"a": 1},
                    {"a": 2},
                    {"a": 3},
                    {"a": 4},
                    {"a": 5},
                    {"a": 6},
                ]
            ],
        )

        batch.set_data(
            [
                [
                    {"a": 0},
                    {"a": 1},
                    {"a": 2},
                    {"a": 3},
                    {"a": 4},
                    {"a": 5},
                    {"a": 6},
                ]
            ]
        )

        old_hash = batch.data_hash

        data = batch.get_data(5)
        assert data == [{"a": 0}, {"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        assert batch.data == [[{"a": 5}, {"a": 6}]]
        assert batch.data_hash != old_hash

    def test_set_data(self) -> None:
        batch = _Batch(seq_no=0, step_name="step1", last_batch=False)
        data = [[{"i": i} for i in range(5000)]]
        batch.set_data(data)

        assert batch.data == data
        assert batch.size == 5000

    def test_next_batch(self) -> None:
        batch = _Batch(seq_no=0, step_name="step1", last_batch=False)
        next_batch = batch.next_batch()

        assert next_batch == _Batch(seq_no=1, step_name="step1", last_batch=False)

    def test_accumulate(self) -> None:
        batches = [
            [
                _Batch(
                    seq_no=0,
                    step_name="step1",
                    last_batch=False,
                    data=[[{"a": 1}, {"a": 2}, {"a": 3}]],
                ),
                _Batch(
                    seq_no=1,
                    step_name="step1",
                    last_batch=True,
                    data=[[{"a": 4}, {"a": 5}, {"a": 6}]],
                ),
            ],
            [
                _Batch(
                    seq_no=0,
                    step_name="step2",
                    last_batch=False,
                    data=[[{"b": 1}, {"b": 2}, {"b": 3}]],
                ),
                _Batch(
                    seq_no=1,
                    step_name="step2",
                    last_batch=True,
                    data=[[{"b": 4}, {"b": 5}, {"b": 6}]],
                ),
            ],
        ]

        batch = _Batch.accumulate("step3", batches)

        assert batch.seq_no == 0
        assert batch.step_name == "step3"
        assert batch.last_batch is True
        assert batch.data == [
            [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}, {"a": 6}],
            [{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}, {"b": 6}],
        ]

    def test_dump(self) -> None:
        batch = _Batch(seq_no=0, step_name="step1", last_batch=False)
        assert batch.dump() == {
            "seq_no": 0,
            "size": 0,
            "step_name": "step1",
            "last_batch": False,
            "data": [],
            "data_hash": None,
            "accumulated": False,
            "created_from": {},
            "batch_routed_to": [],
            "type_info": {"module": "distilabel.pipeline.base", "name": "_Batch"},
        }

        batch = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}]],
            data_hash="hash",
            accumulated=False,
            created_from={"step0": [(0, 5), (1, 5)]},
            batch_routed_to=["step2", "step3"],
        )
        assert batch.dump() == {
            "seq_no": 0,
            "size": 0,
            "step_name": "step1",
            "last_batch": False,
            "data": [[{"a": 1}, {"a": 2}, {"a": 3}]],
            "data_hash": "hash",
            "accumulated": False,
            "created_from": {"step0": [(0, 5), (1, 5)]},
            "batch_routed_to": ["step2", "step3"],
            "type_info": {"module": "distilabel.pipeline.base", "name": "_Batch"},
        }

    def test_from_dict(self) -> None:
        assert isinstance(
            _Batch.from_dict(
                {
                    "seq_no": 0,
                    "step_name": "step1",
                    "last_batch": False,
                    "data": [[{"a": 1}, {"a": 2}, {"a": 3}]],
                    "accumulated": False,
                    "type_info": {
                        "module": "distilabel.pipeline.base",
                        "name": "_Batch",
                    },
                }
            ),
            _Batch,
        )


class TestBatchManagerStep:
    def test_add_batch(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2", accumulate=False, input_batch_size=10, data={"step1": []}
        )

        batch = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}]],
        )

        batch_manager_step.add_batch(batch)

        assert batch_manager_step.data["step1"] == [batch]
        assert batch_manager_step.last_batch_received == []

    def test_add_batch_with_prepend(self) -> None:
        batch_1 = _Batch(
            seq_no=1,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 6}, {"a": 7}, {"a": 8}, {"a": 9}, {"a": 10}]],
        )
        batch_manager_step = _BatchManagerStep(
            step_name="step2",
            accumulate=False,
            input_batch_size=10,
            data={"step1": [batch_1]},
        )

        batch_0 = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
        )
        batch_manager_step.add_batch(batch_0, prepend=True)

        assert batch_manager_step.data["step1"] == [batch_0, batch_1]
        assert batch_manager_step.last_batch_received == []

    def test_add_batch_last_batch(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2", accumulate=False, input_batch_size=10, data={"step1": []}
        )

        batch = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=True,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}]],
        )

        batch_manager_step.add_batch(batch)

        assert batch_manager_step.data["step1"] == [batch]
        assert batch_manager_step.last_batch_received == ["step1"]

    def test_get_batch(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=2,
            data={
                "step1": [
                    _Batch(
                        seq_no=0,
                        step_name="step1",
                        last_batch=False,
                        data=[
                            [
                                {"a": 1},
                                {"a": 2},
                                {"a": 3},
                                {"a": 4},
                                {"a": 5},
                            ]
                        ],
                        size=5,
                    )
                ],
                "step2": [
                    _Batch(
                        seq_no=0,
                        step_name="step2",
                        last_batch=False,
                        data=[
                            [
                                {"b": 1},
                                {"b": 2},
                                {"b": 3},
                                {"b": 4},
                                {"b": 5},
                                {"b": 6},
                            ]
                        ],
                        size=5,
                    )
                ],
            },
        )

        batch = batch_manager_step.get_batch()

        assert batch == _Batch(
            step_name="step3",
            seq_no=0,
            last_batch=False,
            data=[
                [
                    {"a": 1},
                    {"a": 2},
                ],
                [
                    {"b": 1},
                    {"b": 2},
                ],
            ],
            created_from={"step1": [(0, 5)], "step2": [(0, 5)]},
        )

        batch = batch_manager_step.get_batch()

        assert batch == _Batch(
            step_name="step3",
            seq_no=1,
            last_batch=False,
            data=[
                [
                    {"a": 3},
                    {"a": 4},
                ],
                [
                    {"b": 3},
                    {"b": 4},
                ],
            ],
            created_from={"step1": [(0, 5)], "step2": [(0, 5)]},
        )

    def test_get_batches_accumulate(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data={
                "step1": [
                    _Batch(
                        seq_no=0,
                        step_name="step1",
                        last_batch=True,
                        data=[
                            [
                                {"a": 1},
                                {"a": 2},
                                {"a": 3},
                                {"a": 4},
                                {"a": 5},
                            ]
                        ],
                        size=5,
                    )
                ],
                "step2": [
                    _Batch(
                        seq_no=0,
                        step_name="step2",
                        last_batch=True,
                        data=[
                            [
                                {"b": 1},
                                {"b": 2},
                                {"b": 3},
                                {"b": 4},
                                {"b": 5},
                                {"b": 6},
                            ]
                        ],
                        size=6,
                    )
                ],
            },
            last_batch_received=["step1", "step2"],
        )

        batch = batch_manager_step.get_batch()

        assert batch == _Batch(
            step_name="step3",
            seq_no=0,
            last_batch=True,
            accumulated=True,
            data=[
                [
                    {"a": 1},
                    {"a": 2},
                    {"a": 3},
                    {"a": 4},
                    {"a": 5},
                ],
                [
                    {"b": 1},
                    {"b": 2},
                    {"b": 3},
                    {"b": 4},
                    {"b": 5},
                    {"b": 6},
                ],
            ],
            created_from={"step1": [(0, 5)], "step2": [(0, 6)]},
        )

    def test_get_batches_not_enough_data(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=2,
            data={
                "step1": [
                    _Batch(
                        seq_no=0,
                        step_name="step1",
                        last_batch=False,
                        data=[
                            [
                                {"a": 1},
                            ]
                        ],
                    )
                ],
                "step2": [
                    _Batch(
                        seq_no=0,
                        step_name="step2",
                        last_batch=False,
                        data=[
                            [
                                {"b": 1},
                                {"b": 2},
                            ]
                        ],
                    )
                ],
            },
        )

        assert batch_manager_step.get_batch() is None

    def test_from_step(self, dummy_step_1: "Step") -> None:
        batch_manager_step = _BatchManagerStep.from_step(
            step=dummy_step_1, predecessors=["step1", "step2"]
        )

        assert batch_manager_step.step_name == "dummy_step_1"
        assert batch_manager_step.accumulate is False
        assert batch_manager_step.input_batch_size == 50
        assert batch_manager_step.data == {"step1": [], "step2": []}
        assert batch_manager_step.seq_no == 0
        assert batch_manager_step.last_batch_received == []

    def test_from_step_with_global_step(self, dummy_global_step: "GlobalStep") -> None:
        batch_manager_step = _BatchManagerStep.from_step(
            step=dummy_global_step, predecessors=["step1", "step2"]
        )

        assert batch_manager_step.step_name == "dummy_global_step"
        assert batch_manager_step.accumulate is True
        assert batch_manager_step.input_batch_size == 50
        assert batch_manager_step.data == {"step1": [], "step2": []}
        assert batch_manager_step.seq_no == 0
        assert batch_manager_step.last_batch_received == []

    def test_get_seq_no(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2", accumulate=False, input_batch_size=5, data={"step1": []}
        )

        seq_no = batch_manager_step._get_seq_no()

        assert seq_no == 0
        assert batch_manager_step.seq_no == 1

    def test_get_data(self) -> None:
        batch_step_1 = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}, {"a": 6}]],
            size=6,
            batch_routed_to=["step1", "step2"],
        )
        batch_step_2 = _Batch(
            seq_no=0,
            step_name="step2",
            last_batch=False,
            data=[
                [
                    {"b": 1},
                    {"b": 2},
                    {"b": 3},
                    {"b": 4},
                    {"b": 5},
                    {"b": 6},
                    {"b": 7},
                ]
            ],
            size=7,
            batch_routed_to=["step1", "step2"],
        )
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=5,
            data={
                "step1": [batch_step_1],
                "step2": [batch_step_2],
            },
        )

        data, created_from, routed_to = batch_manager_step._get_data()
        assert data == [
            [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}],
            [{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}],
        ]
        assert created_from == {"step1": [(0, 6)], "step2": [(0, 7)]}
        assert routed_to == ["step1", "step2"]

        assert batch_manager_step.data == {
            "step1": [
                _Batch(
                    seq_no=0,
                    step_name="step1",
                    last_batch=False,
                    data=[[{"a": 6}]],
                    data_hash=batch_step_1.data_hash,
                    size=6,
                    batch_routed_to=["step1", "step2"],
                )
            ],
            "step2": [
                _Batch(
                    seq_no=0,
                    step_name="step2",
                    last_batch=False,
                    data=[[{"b": 6}, {"b": 7}]],
                    data_hash=batch_step_2.data_hash,
                    size=7,
                    batch_routed_to=["step1", "step2"],
                )
            ],
        }

    def test_get_data_accumulate(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data={
                "step1": [
                    _Batch(
                        seq_no=0,
                        step_name="step1",
                        last_batch=False,
                        data=[
                            [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}, {"a": 6}]
                        ],
                        size=6,
                    )
                ],
                "step2": [
                    _Batch(
                        seq_no=0,
                        step_name="step2",
                        last_batch=False,
                        data=[
                            [
                                {"b": 1},
                                {"b": 2},
                                {"b": 3},
                                {"b": 4},
                                {"b": 5},
                                {"b": 6},
                                {"b": 7},
                            ]
                        ],
                        size=7,
                    )
                ],
            },
        )

        data, created_from, routed_to = batch_manager_step._get_data()

        assert data == [
            [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}, {"a": 6}],
            [{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}, {"b": 6}, {"b": 7}],
        ]
        assert created_from == {"step1": [(0, 6)], "step2": [(0, 7)]}
        assert routed_to == []

        assert batch_manager_step.data == {"step1": [], "step2": []}

    def test_get_data_convergence_step(self) -> None:
        batch_a_0 = _Batch(
            seq_no=0,
            step_name="A",
            last_batch=False,
            data=[
                [
                    {"generation": "Hello, I'm A 0"},
                    {"generation": "Hello, I'm A 0"},
                    {"generation": "Hello, I'm A 0"},
                ]
            ],
            size=3,
            created_from={"Z": [(0, 3)]},
        )

        batch_a_1 = _Batch(
            seq_no=1,
            step_name="A",
            last_batch=False,
            data=[
                [
                    {"generation": "Hello, I'm A 1"},
                    {"generation": "Hello, I'm A 1"},
                    {"generation": "Hello, I'm A 1"},
                ]
            ],
            size=3,
            created_from={"Z": [(1, 3)]},
        )

        batch_b_0 = _Batch(
            seq_no=0,
            step_name="B",
            last_batch=False,
            data=[
                [
                    {"generation": "Hello, I'm B 0"},
                    {"generation": "Hello, I'm B 0"},
                    {"generation": "Hello, I'm B 0"},
                ]
            ],
            size=3,
            created_from={"Z": [(0, 3)]},
        )

        batch_c_0 = _Batch(
            seq_no=0,
            step_name="C",
            last_batch=False,
            data=[
                [
                    {"generation": "Hello, I'm C 0"},
                    {"generation": "Hello, I'm C 0"},
                    {"generation": "Hello, I'm C 0"},
                ]
            ],
            size=3,
            created_from={"Z": [(1, 3)]},
        )

        batch_manager_step = _BatchManagerStep(
            step_name="D",
            input_batch_size=3,
            convergence_step=True,
            accumulate=False,
            data={"A": [batch_a_0, batch_a_1], "B": [batch_b_0], "C": [batch_c_0]},
        )

        data, created_from, routed_to = batch_manager_step._get_data()

        assert data == [
            [
                {"generation": "Hello, I'm A 0"},
                {"generation": "Hello, I'm A 0"},
                {"generation": "Hello, I'm A 0"},
            ],
            [
                {"generation": "Hello, I'm B 0"},
                {"generation": "Hello, I'm B 0"},
                {"generation": "Hello, I'm B 0"},
            ],
        ]
        assert created_from == {"A": [(0, 3)], "B": [(0, 3)]}
        assert routed_to == []
        assert batch_manager_step.next_expected_created_from_batch_seq_no == 1

        data, created_from, routed_to = batch_manager_step._get_data()

        assert data == [
            [
                {"generation": "Hello, I'm A 1"},
                {"generation": "Hello, I'm A 1"},
                {"generation": "Hello, I'm A 1"},
            ],
            [
                {"generation": "Hello, I'm C 0"},
                {"generation": "Hello, I'm C 0"},
                {"generation": "Hello, I'm C 0"},
            ],
        ]
        assert created_from == {"A": [(1, 3)], "C": [(0, 3)]}
        assert routed_to == []
        assert batch_manager_step.next_expected_created_from_batch_seq_no == 2

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ]
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}]],
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[
                                [
                                    {"a": 1},
                                    {"a": 2},
                                    {"a": 3},
                                    {"a": 4},
                                    {"a": 5},
                                    {"a": 6},
                                ]
                            ],
                        )
                    ]
                },
                ["step1"],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ]
                },
                ["step1"],
                True,
            ),
        ],
    )
    def test_last_batch(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2",
            accumulate=False,
            input_batch_size=5,
            data=data,
            last_batch_received=last_batch_received,
        )

        assert batch_manager_step._last_batch() is expected

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                ["step1"],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                ["step1", "step2"],
                True,
            ),
        ],
    )
    def test_last_batch_accumulate(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data=data,
            last_batch_received=last_batch_received,
        )

        assert batch_manager_step._last_batch() is expected

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}]],
                            created_from={"step0": [(0, 3)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}]],
                            created_from={"step0": [(0, 3)]},
                        )
                    ],
                },
                [],
                True,
            ),
        ],
    )
    def test_last_batch_convergence_step(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            data=data,
            last_batch_received=last_batch_received,
            input_batch_size=3,
            convergence_step=True,
        )

        assert batch_manager_step._last_batch() is expected

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                ["step1", "step2"],
                True,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}]],
                        )
                    ],
                },
                ["step1", "step2"],
                True,
            ),
        ],
    )
    def test_ready_to_create_batch(
        self,
        data: Dict[str, List[Dict[str, Any]]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2",
            accumulate=False,
            input_batch_size=5,
            data=data,
            last_batch_received=last_batch_received,
        )

        assert batch_manager_step._ready_to_create_batch() is expected

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                ["step1", "step2"],
                True,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                        )
                    ],
                },
                ["step1"],
                False,
            ),
        ],
    )
    def test_ready_to_create_batch_accumulate(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data=data,
            last_batch_received=last_batch_received,
        )

        assert batch_manager_step._ready_to_create_batch() is expected

    def test_dump(self) -> None:
        batch_step_1 = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=True,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}, {"a": 6}]],
            data_hash="hash0",
            size=6,
        )
        batch_step_2 = _Batch(
            seq_no=0,
            step_name="step2",
            last_batch=True,
            data=[
                [{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}, {"b": 6}, {"b": 7}]
            ],
            data_hash="hash1",
            size=7,
        )
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data={
                "step1": [batch_step_1],
                "step2": [batch_step_2],
            },
        )
        assert batch_manager_step.dump() == {
            "step_name": "step3",
            "accumulate": True,
            "convergence_step": False,
            "convergence_step_batches_consumed": {},
            "input_batch_size": None,
            "data": {
                "step1": [
                    {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": True,
                        "data": [
                            [
                                {"a": 1},
                                {"a": 2},
                                {"a": 3},
                                {"a": 4},
                                {"a": 5},
                                {"a": 6},
                            ]
                        ],
                        "data_hash": "hash0",
                        "size": 6,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                    }
                ],
                "step2": [
                    {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": True,
                        "data": [
                            [
                                {"b": 1},
                                {"b": 2},
                                {"b": 3},
                                {"b": 4},
                                {"b": 5},
                                {"b": 6},
                                {"b": 7},
                            ]
                        ],
                        "data_hash": "hash1",
                        "size": 7,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                    }
                ],
            },
            "seq_no": 0,
            "last_batch_received": [],
            "next_expected_created_from_batch_seq_no": 0,
            "type_info": {
                "module": "distilabel.pipeline.base",
                "name": "_BatchManagerStep",
            },
        }

    @pytest.mark.parametrize(
        "data, last_batch_received, expected",
        [
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                    "step2": [],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                },
                [],
                True,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 4)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                },
                [],
                False,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 4)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 4)]},
                        )
                    ],
                },
                ["step1", "step2"],
                True,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=False,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 4)]},
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=0,
                            step_name="step2",
                            last_batch=False,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}]],
                            batch_routed_to=["step1", "step2"],
                            created_from={"step0": [(0, 5)]},
                        )
                    ],
                },
                [],
                False,
            ),
        ],
    )
    def test_ready_to_create_batch_convergence_step(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=5,
            data=data,
            last_batch_received=last_batch_received,
            convergence_step=True,
        )

        assert batch_manager_step._ready_to_create_batch() is expected

    def test_from_dict(self) -> None:
        batch_manager_step = _BatchManagerStep.from_dict(
            {
                "step_name": "step3",
                "accumulate": True,
                "convergence_step": False,
                "convergence_step_batches_consumed": {0: {"Z": 1234}},
                "input_batch_size": None,
                "data": {
                    "step1": [
                        {
                            "seq_no": 0,
                            "step_name": "step1",
                            "last_batch": True,
                            "data": [
                                [
                                    {"a": 1},
                                    {"a": 2},
                                    {"a": 3},
                                    {"a": 4},
                                    {"a": 5},
                                    {"a": 6},
                                ]
                            ],
                            "size": 6,
                            "accumulated": False,
                            "created_from": {},
                            "batch_routed_to": [],
                        }
                    ],
                    "step2": [
                        {
                            "seq_no": 0,
                            "step_name": "step2",
                            "last_batch": True,
                            "data": [
                                [
                                    {"b": 1},
                                    {"b": 2},
                                    {"b": 3},
                                    {"b": 4},
                                    {"b": 5},
                                    {"b": 6},
                                    {"b": 7},
                                ]
                            ],
                            "size": 7,
                            "accumulated": False,
                            "created_from": {},
                            "batch_routed_to": [],
                        }
                    ],
                },
                "seq_no": 0,
                "last_batch_received": [],
                "type_info": {
                    "module": "distilabel.pipeline.base",
                    "name": "_BatchManagerStep",
                },
            }
        )

        assert isinstance(batch_manager_step, _BatchManagerStep)
        assert batch_manager_step.step_name == "step3"
        assert batch_manager_step.accumulate is True
        assert batch_manager_step.convergence_step is False
        assert batch_manager_step.convergence_step_batches_consumed == {0: {"Z": 1234}}
        assert batch_manager_step.input_batch_size is None
        assert batch_manager_step.seq_no == 0
        assert batch_manager_step.last_batch_received == []


class TestBatchManager:
    def test_add_batch(self) -> None:
        batch_manager = _BatchManager(
            steps={
                "step3": _BatchManagerStep(
                    step_name="step3",
                    accumulate=False,
                    input_batch_size=5,
                    data={"step1": [], "step2": []},
                )
            },
            last_batch_received={"step3": None},
            last_batch_sent={"step3": None},
            last_batch_flag_sent_to=[],
        )

        batch_from_step_1 = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
        )
        batch_manager.add_batch(to_step="step3", batch=batch_from_step_1)

        assert batch_manager._steps["step3"].data == {
            "step1": [batch_from_step_1],
            "step2": [],
        }

    def test_add_batch_with_prepend(self) -> None:
        batch_1 = _Batch(
            seq_no=1,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 6}, {"a": 7}, {"a": 8}, {"a": 9}, {"a": 10}]],
        )
        batch_manager = _BatchManager(
            steps={
                "step3": _BatchManagerStep(
                    step_name="step3",
                    accumulate=False,
                    input_batch_size=5,
                    data={
                        "step1": [batch_1],
                        "step2": [],
                    },
                )
            },
            last_batch_received={"step3": None},
            last_batch_sent={"step3": None},
            last_batch_flag_sent_to=[],
        )
        batch_0 = _Batch(
            seq_no=0,
            step_name="step1",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
        )
        batch_manager.add_batch(to_step="step3", batch=batch_0, prepend=True)
        assert batch_manager._steps["step3"].data == {
            "step1": [batch_0, batch_1],
            "step2": [],
        }

    def test_from_dag(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
        dummy_global_step: "GlobalStep",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_step(dummy_global_step)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_generator_step", "dummy_global_step")
        dag.add_edge("dummy_step_1", "dummy_step_2")

        batch_manager = _BatchManager.from_dag(dag)

        assert batch_manager._steps == {
            "dummy_step_1": _BatchManagerStep(
                step_name="dummy_step_1",
                accumulate=False,
                input_batch_size=50,
                data={"dummy_generator_step": []},
            ),
            "dummy_global_step": _BatchManagerStep(
                step_name="dummy_global_step",
                accumulate=True,
                input_batch_size=50,
                data={"dummy_generator_step": []},
            ),
            "dummy_step_2": _BatchManagerStep(
                step_name="dummy_step_2",
                accumulate=False,
                input_batch_size=50,
                data={"dummy_step_1": []},
            ),
        }

    def test_can_generate(self) -> None:
        batch_manager = _BatchManager(
            steps={},
            last_batch_received={
                "step_1": _Batch(seq_no=0, step_name="step_1", last_batch=False),
                "step_2": _Batch(seq_no=0, step_name="step_2", last_batch=False),
                "step_3": _Batch(seq_no=0, step_name="step_3", last_batch=False),
            },
            last_batch_sent={"step_1": None, "step_2": None, "step_3": None},
            last_batch_flag_sent_to=[],
        )

        assert batch_manager.can_generate()

        batch_1 = _Batch(seq_no=0, step_name="step_1", last_batch=True)
        batch_2 = _Batch(seq_no=0, step_name="step_2", last_batch=True)
        batch_3 = _Batch(seq_no=0, step_name="step_3", last_batch=True)

        batch_manager = _BatchManager(
            steps={},
            last_batch_received={
                "step_1": batch_1,
                "step_2": batch_2,
                "step_3": batch_3,
            },
            last_batch_sent={"step_1": batch_1, "step_2": batch_2, "step_3": batch_3},
            last_batch_flag_sent_to=[],
        )

        assert not batch_manager.can_generate()

    def test_dump(self) -> None:
        batch_manager = _BatchManager(
            steps={
                "step3": _BatchManagerStep(
                    step_name="step3",
                    accumulate=False,
                    input_batch_size=5,
                    data={"step1": [], "step2": []},
                    seq_no=1,
                )
            },
            last_batch_received={
                "step3": _Batch(
                    seq_no=0,
                    step_name="step3",
                    last_batch=False,
                )
            },
            last_batch_sent={
                "step3": _Batch(
                    seq_no=1,
                    step_name="step3",
                    last_batch=False,
                )
            },
            last_batch_flag_sent_to=["step99"],
        )
        assert batch_manager.dump() == {
            "steps": {
                "step3": {
                    "step_name": "step3",
                    "accumulate": False,
                    "convergence_step": False,
                    "convergence_step_batches_consumed": {},
                    "input_batch_size": 5,
                    "data": {"step1": [], "step2": []},
                    "seq_no": 1,
                    "last_batch_received": [],
                    "next_expected_created_from_batch_seq_no": 0,
                    "type_info": {
                        "module": "distilabel.pipeline.base",
                        "name": "_BatchManagerStep",
                    },
                },
            },
            "last_batch_received": {
                "step3": {
                    "seq_no": 0,
                    "step_name": "step3",
                    "batch_routed_to": [],
                    "created_from": {},
                    "last_batch": False,
                    "data": [],
                    "size": 0,
                    "accumulated": False,
                    "type_info": {
                        "module": "distilabel.pipeline.base",
                        "name": "_Batch",
                    },
                }
            },
            "last_batch_sent": {
                "step3": {
                    "seq_no": 1,
                    "step_name": "step3",
                    "batch_routed_to": [],
                    "created_from": {},
                    "last_batch": False,
                    "data": [],
                    "size": 0,
                    "accumulated": False,
                    "type_info": {
                        "module": "distilabel.pipeline.base",
                        "name": "_Batch",
                    },
                }
            },
            "last_batch_flag_sent_to": ["step99"],
            "type_info": {
                "module": "distilabel.pipeline.base",
                "name": "_BatchManager",
            },
        }

    def test_from_dict(self) -> None:
        batch_manager_step = _BatchManagerStep.from_dict(
            {
                "step_name": "step3",
                "accumulate": True,
                "convergence_step": False,
                "input_batch_size": None,
                "data": {
                    "step1": [
                        {
                            "seq_no": 0,
                            "step_name": "step1",
                            "last_batch": True,
                            "data": [
                                [
                                    {"a": 1},
                                    {"a": 2},
                                    {"a": 3},
                                    {"a": 4},
                                    {"a": 5},
                                    {"a": 6},
                                ]
                            ],
                            "accumulated": False,
                            "created_from": {},
                            "batch_routed_to": [],
                        }
                    ],
                    "step2": [
                        {
                            "seq_no": 0,
                            "step_name": "step2",
                            "last_batch": True,
                            "data": [
                                [
                                    {"b": 1},
                                    {"b": 2},
                                    {"b": 3},
                                    {"b": 4},
                                    {"b": 5},
                                    {"b": 6},
                                    {"b": 7},
                                ]
                            ],
                            "accumulated": False,
                            "created_from": {},
                            "batch_routed_to": [],
                        }
                    ],
                },
                "seq_no": 0,
                "last_batch_received": [],
                "type_info": {
                    "module": "distilabel.pipeline.base",
                    "name": "_BatchManagerStep",
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            batch_manager_step.save(Path(tmpdirname) / "batch_manager_step3.json")

            batch_manager = _BatchManager.from_dict(
                {
                    "steps": {
                        "step3": str(Path(tmpdirname) / "batch_manager_step3.json")
                    },
                    "last_batch_received": {
                        "step3": {
                            "seq_no": 0,
                            "step_name": "step3",
                            "last_batch": False,
                            "data": [],
                            "accumulated": False,
                            "type_info": {
                                "module": "distilabel.pipeline.base",
                                "name": "_Batch",
                            },
                        }
                    },
                    "last_batch_sent": {
                        "step3": {
                            "seq_no": 0,
                            "step_name": "step3",
                            "last_batch": False,
                            "data": [],
                            "accumulated": False,
                            "type_info": {
                                "module": "distilabel.pipeline.base",
                                "name": "_Batch",
                            },
                        },
                    },
                    "last_batch_flag_sent_to": [],
                    "type_info": {
                        "module": "distilabel.pipeline.base",
                        "name": "_BatchManager",
                    },
                }
            )
        assert isinstance(batch_manager, _BatchManager)
        assert all(
            isinstance(step, _BatchManagerStep)
            for _, step in batch_manager._steps.items()
        )
        assert all(
            isinstance(batch, _Batch)
            for _, batch in batch_manager._last_batch_received.items()
        )


class TestPipelineSerialization:
    def test_base_pipeline_dump(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert dump["pipeline"][TYPE_INFO_KEY]["module"] == "distilabel.pipeline.base"
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "BasePipeline"

    def test_base_pipeline_from_dict(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        pipe = BasePipeline.from_dict(pipeline.dump())
        assert isinstance(pipe, BasePipeline)

    def test_pipeline_dump(self):
        from distilabel.pipeline.local import Pipeline

        pipeline = Pipeline(name="unit-test-pipeline")
        dump = pipeline.dump()
        assert len(dump.keys()) == 2
        assert "pipeline" in dump
        assert "distilabel" in dump
        assert TYPE_INFO_KEY in dump["pipeline"]
        assert dump["pipeline"][TYPE_INFO_KEY]["module"] == "distilabel.pipeline.local"
        assert dump["pipeline"][TYPE_INFO_KEY]["name"] == "Pipeline"

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
        pipeline = BasePipeline(name="unit-test-pipeline")

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

    def test_base_pipeline_signature(self):
        pipeline = BasePipeline(name="unit-test-pipeline")
        # Doesn't matter if it's exactly this or not, the test should fail if we change the
        # way this is created.
        signature = pipeline._create_signature()
        assert signature == "da39a3ee5e6b4b0d3255bfef95601890afd80709"

        # Maybe not the best place for this test, but does the work for now
        from distilabel.pipeline.local import Pipeline
        from distilabel.pipeline.routing_batch_function import sample_n_steps

        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        sample_two_steps = sample_n_steps(2)

        with Pipeline(name="unit-test-pipeline") as pipeline:
            dummy_generator = DummyGeneratorStep()
            dummy_step_1_0 = DummyStep1()
            dummy_step_1_1 = DummyStep1()
            dummy_step_1_2 = DummyStep1()
            dummy_step_2 = DummyStep2()

            (
                dummy_generator
                >> sample_two_steps
                >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                >> dummy_step_2
            )

        signature = pipeline._create_signature()
        assert signature == "a11ac46253598e6fe126420b23b9ad31c6422c92"

    @pytest.mark.parametrize("use_cache", [True, False])
    def test_run_pipe_and_load_from_cache(self, use_cache: bool):
        # Maybe not the best place for this test, but does the work for now
        from distilabel.pipeline.base import BasePipeline
        from distilabel.pipeline.routing_batch_function import sample_n_steps

        from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2

        sample_two_steps = sample_n_steps(2)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with BasePipeline(
                name="unit-test-pipeline", cache_dir=tmpdirname
            ) as pipeline:
                dummy_generator = DummyGeneratorStep()
                dummy_step_1_0 = DummyStep1()
                dummy_step_1_1 = DummyStep1()
                dummy_step_1_2 = DummyStep1()
                dummy_step_2 = DummyStep2()

                (
                    dummy_generator
                    >> sample_two_steps
                    >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                    >> dummy_step_2
                )

                pipeline.run({}, use_cache=use_cache)

                assert not pipeline._cache_location["pipeline"].exists()
                # Set the _BatchManager to the pipeline to check it exists afterwards
                pipeline._batch_manager = _BatchManager.from_dag(pipeline.dag)
                pipeline._cache()

                assert pipeline._cache_location["pipeline"].exists()

            with BasePipeline(name="unit-test-pipeline", cache_dir=tmpdirname) as pipe:
                dummy_generator = DummyGeneratorStep()
                dummy_step_1_0 = DummyStep1()
                dummy_step_1_1 = DummyStep1()
                dummy_step_1_2 = DummyStep1()
                dummy_step_2 = DummyStep2()

                (
                    dummy_generator
                    >> sample_two_steps
                    >> [dummy_step_1_0, dummy_step_1_1, dummy_step_1_2]
                    >> dummy_step_2
                )

                pipe.run({}, use_cache=use_cache)
                if use_cache:
                    assert pipe._batch_manager
                else:
                    assert not pipe._batch_manager

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

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-3") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> dummy_step_1 >> dummy_step_2

            signature_2 = pipeline_2._create_signature()

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

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")

            dummy_generator >> [dummy_step_1, dummy_step_2]

            signature_2 = pipeline_2._create_signature()

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

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2._create_signature()

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

            signature_1 = pipeline_1._create_signature()

        with Pipeline(name="unit-test-pipeline-2") as pipeline_2:
            dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
            dummy_step_1 = DummyStep1(name="dummy_step_1")
            dummy_step_2 = DummyStep2(name="dummy_step_2")
            dummy_global = DummyGlobalStep(name="dummy_global_step")

            dummy_generator >> [dummy_step_1, dummy_step_2] >> dummy_global
            signature_2 = pipeline_2._create_signature()

        assert signature_1 == signature_2


class TestWriteBuffer:
    def test_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "data"
            with Pipeline(name="unit-test-pipeline") as pipeline:
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

            assert write_buffer._buffers == {"dummy_step_2": [], "dummy_step_3": []}
            assert write_buffer._buffers_dump_batch_size == {
                "dummy_step_2": 50,
                "dummy_step_3": 50,
            }
            assert write_buffer._buffer_last_schema == {}
            assert write_buffer._buffers_last_file == {
                "dummy_step_2": 1,
                "dummy_step_3": 1,
            }

    def test_write_buffer_one_leaf_step_and_create_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "data"
            with Pipeline(name="unit-test-pipeline") as pipeline:
                dummy_generator = DummyGeneratorStep(name="dummy_generator_step")
                dummy_step_1 = DummyStep1(name="dummy_step_1")
                dummy_step_2 = DummyStep2(name="dummy_step_2")

                dummy_generator.connect(dummy_step_1)
                dummy_step_1.connect(dummy_step_2)

            write_buffer = _WriteBuffer(path=folder, leaf_steps=pipeline.dag.leaf_steps)

            # Add one batch with 5 rows, shouldn't write anything 5 < 50
            batch = batch_gen(dummy_step_2.name)
            write_buffer.add_batch(batch)

            # Add 45 more rows, should write now
            for _ in range(9):
                batch = batch_gen(dummy_step_2.name)
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00001.parquet").exists()

            # Add 50 more rows, we should have a new file
            for _ in range(10):
                batch = batch_gen(dummy_step_2.name)
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00002.parquet").exists()

            # Add more rows and close the write buffer, we should have a new file
            for _ in range(5):
                batch = batch_gen(dummy_step_2.name)
                write_buffer.add_batch(batch)

            write_buffer.close()

            assert Path(folder, "dummy_step_2", "00003.parquet").exists()

            ds = create_distiset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 1
            assert len(ds["default"]["train"]) == 125

    def test_write_buffer_multiple_leaf_steps_and_create_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            folder = Path(tmpdirname) / "data"
            with Pipeline(name="unit-test-pipeline") as pipeline:
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

            for _ in range(10):
                batch = batch_gen(dummy_step_2.name)
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00001.parquet").exists()

            for _ in range(10):
                batch = batch_gen(dummy_step_3.name)
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_3", "00001.parquet").exists()

            for _ in range(5):
                batch = batch_gen(dummy_step_2.name)
                write_buffer.add_batch(batch)

            for _ in range(5):
                batch = batch_gen(dummy_step_3.name)
                write_buffer.add_batch(batch)

            write_buffer.close()

            assert Path(folder, "dummy_step_2", "00002.parquet").exists()
            assert Path(folder, "dummy_step_3", "00002.parquet").exists()

            ds = create_distiset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 2
            assert len(ds["dummy_step_2"]["train"]) == 75
            assert len(ds["dummy_step_3"]["train"]) == 75
