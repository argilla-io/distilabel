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
from typing import Dict, List

import pytest

from distilabel.pipeline._dag import DAG
from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.batch_manager import _BatchManager, _BatchManagerStep
from distilabel.steps.base import GeneratorStep, GlobalStep, Step


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
            step_name="step2",
            last_batch=False,
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
        )
        batch_manager_step.add_batch(batch_0, prepend=True)

        assert batch_manager_step.built_batches == [batch_0]
        assert batch_manager_step.data["step1"] == [batch_1]
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
        previously_built_batch = _Batch(
            seq_no=0,
            step_name="step3",
            last_batch=False,
            data=[
                [
                    {"a": -1},
                    {"a": 0},
                ],
                [
                    {"b": -1},
                    {"b": 0},
                ],
            ],
        )

        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=2,
            seq_no=1,
            data={
                "step1": [
                    _Batch(
                        seq_no=1,
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
                        seq_no=1,
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
            built_batches=[previously_built_batch],
            next_expected_seq_no={"step1": (1, 1), "step2": (1, 1)},
        )

        batch = batch_manager_step.get_batch()

        assert batch == previously_built_batch

        batch = batch_manager_step.get_batch()

        assert batch == _Batch(
            step_name="step3",
            seq_no=1,
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
            created_from={"step1": [(1, 5)], "step2": [(1, 5)]},
        )

        batch = batch_manager_step.get_batch()

        assert batch == _Batch(
            step_name="step3",
            seq_no=2,
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
            created_from={"step1": [(1, 5)], "step2": [(1, 5)]},
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
            next_expected_seq_no={"step1": (0, 0), "step2": (0, 0)},
        )

        assert batch_manager_step.get_batch() is None

    def test_set_next_expected_seq_no(self) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=False,
            input_batch_size=2,
            data={
                "step1": [],
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
                "step3": [
                    _Batch(
                        seq_no=2,
                        step_name="step3",
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
            next_expected_seq_no={"step1": (0, 0), "step2": (0, 0), "step3": (0, 0)},
        )

        batch_manager_step.set_next_expected_seq_no(
            from_step="step1", next_expected_seq_no=1
        )

        assert batch_manager_step.next_expected_seq_no["step1"] == (1, 1)

        batch_manager_step.set_next_expected_seq_no(
            from_step="step2", next_expected_seq_no=1
        )

        assert batch_manager_step.next_expected_seq_no["step2"] == (0, 1)

        batch_manager_step.set_next_expected_seq_no(
            from_step="step3", next_expected_seq_no=1
        )

        assert batch_manager_step.next_expected_seq_no["step3"] == (1, 1)

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
            next_expected_seq_no={"step1": (0, 0), "step2": (0, 0)},
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
        "data, last_batch_received, next_expected_seq_no, expected",
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
                {"step1": (0, 0), "step2": (0, 0)},
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
                {"step1": (0, 0), "step2": (0, 0)},
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
                {"step1": (0, 0), "step2": (0, 0)},
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
                {"step1": (0, 0), "step2": (0, 0)},
                True,
            ),
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=1,
                            step_name="step1",
                            last_batch=True,
                            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]],
                        )
                    ],
                    "step2": [
                        _Batch(
                            seq_no=1,
                            step_name="step2",
                            last_batch=True,
                            data=[[{"b": 1}, {"b": 2}, {"b": 3}, {"b": 4}]],
                        )
                    ],
                },
                ["step1", "step2"],
                {"step1": (0, 0), "step2": (0, 0)},
                False,
            ),
        ],
    )
    def test_ready_to_create_batch(
        self,
        data: Dict[str, List[_Batch]],
        last_batch_received: List[str],
        next_expected_seq_no: Dict[str, int],
        expected: bool,
    ) -> None:
        batch_manager_step = _BatchManagerStep(
            step_name="step2",
            accumulate=False,
            input_batch_size=5,
            data=data,
            last_batch_received=last_batch_received,
            next_expected_seq_no=next_expected_seq_no,
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
            (
                {
                    "step1": [
                        _Batch(
                            seq_no=0,
                            step_name="step1",
                            last_batch=True,
                            data=[[]],
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
        batch_step_3 = _Batch(
            seq_no=0,
            step_name="step3",
            last_batch=True,
            data=[[{"c": 1}, {"c": 2}, {"c": 3}, {"c": 4}, {"c": 5}]],
            data_hash="hash2",
            size=5,
        )
        batch_manager_step = _BatchManagerStep(
            step_name="step3",
            accumulate=True,
            data={
                "step1": [batch_step_1],
                "step2": [batch_step_2],
            },
            built_batches=[batch_step_3],
            next_expected_seq_no={"step1": (0, 0), "step2": (0, 0)},
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
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
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
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
                    }
                ],
            },
            "built_batches": [
                {
                    "seq_no": 0,
                    "step_name": "step3",
                    "last_batch": True,
                    "data": [[{"c": 1}, {"c": 2}, {"c": 3}, {"c": 4}, {"c": 5}]],
                    "data_hash": "hash2",
                    "size": 5,
                    "accumulated": False,
                    "batch_routed_to": [],
                    "created_from": {},
                    "type_info": {
                        "module": "distilabel.pipeline.batch",
                        "name": "_Batch",
                    },
                }
            ],
            "seq_no": 0,
            "last_batch_received": [],
            "next_expected_created_from_batch_seq_no": 0,
            "next_expected_seq_no": {
                "step1": (0, 0),
                "step2": (0, 0),
            },
            "type_info": {
                "module": "distilabel.pipeline.batch_manager",
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
                    "module": "distilabel.pipeline.batch_manager",
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
        assert batch_manager._steps["step3"].built_batches == [batch_0]
        assert batch_manager._steps["step3"].data == {
            "step1": [batch_1],
            "step2": [],
        }

    def test_add_batch_to_recover_offline_batch_generation(self) -> None:
        batch_manager = _BatchManager(
            steps={
                "step1": _BatchManagerStep(
                    step_name="step0",
                    accumulate=True,
                    input_batch_size=5,
                    data={},
                )
            },
            last_batch_received={
                "step1": _Batch(seq_no=0, step_name="step1", last_batch=True)
            },
            last_batch_sent={"step1": None},
            last_batch_flag_sent_to=[],
        )

        batch_manager.add_batch_to_recover_offline_batch_generation(
            to_step="step1",
            data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
        )

        assert batch_manager._steps["step1"].built_batches == [
            _Batch(
                seq_no=0,
                step_name="step1",
                last_batch=True,
                data=[[{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]],
            )
        ]
        assert batch_manager._last_batch_received["step1"] is None

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
                next_expected_seq_no={"dummy_generator_step": (0, 0)},
            ),
            "dummy_global_step": _BatchManagerStep(
                step_name="dummy_global_step",
                accumulate=True,
                input_batch_size=50,
                data={"dummy_generator_step": []},
                next_expected_seq_no={"dummy_generator_step": (0, 0)},
            ),
            "dummy_step_2": _BatchManagerStep(
                step_name="dummy_step_2",
                accumulate=False,
                input_batch_size=50,
                data={"dummy_step_1": []},
                next_expected_seq_no={"dummy_step_1": (0, 0)},
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
        built_batch = _Batch(
            seq_no=0,
            last_batch=False,
            step_name="step3",
            data=[[]],
            data_hash="hash",
        )

        batch_manager = _BatchManager(
            steps={
                "step3": _BatchManagerStep(
                    step_name="step3",
                    accumulate=False,
                    input_batch_size=5,
                    data={"step1": [], "step2": []},
                    built_batches=[built_batch],
                    next_expected_seq_no={
                        "step1": (1, 1),
                        "step2": (1, 1),
                    },
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
                    "built_batches": [
                        {
                            "seq_no": 0,
                            "step_name": "step3",
                            "last_batch": False,
                            "data": [[]],
                            "data_hash": "hash",
                            "size": 0,
                            "accumulated": False,
                            "batch_routed_to": [],
                            "created_from": {},
                            "type_info": {
                                "module": "distilabel.pipeline.batch",
                                "name": "_Batch",
                            },
                        }
                    ],
                    "seq_no": 1,
                    "last_batch_received": [],
                    "next_expected_created_from_batch_seq_no": 0,
                    "next_expected_seq_no": {
                        "step1": (1, 1),
                        "step2": (1, 1),
                    },
                    "type_info": {
                        "module": "distilabel.pipeline.batch_manager",
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
                    "data_hash": None,
                    "size": 0,
                    "accumulated": False,
                    "type_info": {
                        "module": "distilabel.pipeline.batch",
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
                    "data_hash": None,
                    "size": 0,
                    "accumulated": False,
                    "type_info": {
                        "module": "distilabel.pipeline.batch",
                        "name": "_Batch",
                    },
                }
            },
            "last_batch_flag_sent_to": ["step99"],
            "type_info": {
                "module": "distilabel.pipeline.batch_manager",
                "name": "_BatchManager",
            },
        }

    def test_from_dict(self) -> None:
        batch_manager = _BatchManager.from_dict(
            {
                "steps": {
                    "step1": {
                        "step_name": "step1",
                        "accumulate": True,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {0: {"Z": 1234}},
                        "input_batch_size": None,
                        "data": {
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
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch_manager",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                    "step2": {
                        "step_name": "step2",
                        "accumulate": False,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {0: {"Z": 1234}},
                        "input_batch_size": 50,
                        "data": {
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
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch_manager",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                },
                "last_batch_received": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_sent": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_flag_sent_to": ["step3"],
                "type_info": {
                    "module": "distilabel.pipeline.batch_manager",
                    "name": "_BatchManager",
                },
            }
        )

        assert isinstance(batch_manager, _BatchManager)

        assert len(batch_manager._steps) == 2
        for step in batch_manager._steps.values():
            assert isinstance(step, _BatchManagerStep)

        assert len(batch_manager._last_batch_received) == 2
        for step in batch_manager._last_batch_received.values():
            assert isinstance(step, _Batch)

        assert len(batch_manager._last_batch_sent) == 2
        for step in batch_manager._last_batch_sent.values():
            assert isinstance(step, _Batch)

        assert batch_manager._last_batch_flag_sent_to == ["step3"]

    def test_cache(self) -> None:
        batch_manager = _BatchManager.from_dict(
            {
                "steps": {
                    "step1": {
                        "step_name": "step1",
                        "accumulate": True,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {"0": {"Z": 1234}},
                        "input_batch_size": None,
                        "data": {
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
                                    "data_hash": "1234",
                                    "size": 7,
                                    "accumulated": False,
                                    "created_from": {},
                                    "batch_routed_to": [],
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch_manager",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "built_batches": [
                            {
                                "seq_no": 0,
                                "step_name": "step1",
                                "last_batch": False,
                                "data": [
                                    [
                                        {"a": 1},
                                        {"a": 2},
                                        {"a": 3},
                                        {"a": 4},
                                        {"a": 5},
                                    ]
                                ],
                                "data_hash": "1234",
                                "size": 5,
                                "accumulated": False,
                                "batch_routed_to": [],
                                "created_from": {},
                                "type_info": {
                                    "module": "distilabel.pipeline.batch",
                                    "name": "_Batch",
                                },
                            }
                        ],
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                    "step2": {
                        "step_name": "step2",
                        "accumulate": False,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {"0": {"Z": 1234}},
                        "input_batch_size": 50,
                        "data": {
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
                                    "data_hash": "1234",
                                    "size": 7,
                                    "accumulated": False,
                                    "created_from": {},
                                    "batch_routed_to": [],
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch_manager",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "built_batches": [
                            {
                                "seq_no": 0,
                                "step_name": "step1",
                                "last_batch": False,
                                "data": [
                                    [
                                        {"a": 1},
                                        {"a": 2},
                                        {"a": 3},
                                        {"a": 4},
                                        {"a": 5},
                                    ]
                                ],
                                "data_hash": "1234",
                                "size": 5,
                                "accumulated": False,
                                "batch_routed_to": [],
                                "created_from": {},
                                "type_info": {
                                    "module": "distilabel.pipeline.batch",
                                    "name": "_Batch",
                                },
                            }
                        ],
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                },
                "last_batch_received": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_sent": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_flag_sent_to": ["step3"],
                "type_info": {
                    "module": "distilabel.pipeline.batch_manager",
                    "name": "_BatchManager",
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_manager_path = Path(tmp_dir) / "batch_manager.json"
            batch_manager.cache(batch_manager_path)

            assert batch_manager_path.exists() and batch_manager_path.is_file()

            for step_name, step in batch_manager._steps.items():
                batch_manager_step_dir = (
                    Path(tmp_dir) / "batch_manager_steps" / step_name
                )
                assert (
                    batch_manager_step_dir.exists() and batch_manager_step_dir.is_dir()
                )

                batch_manager_step_path = (
                    batch_manager_step_dir / "batch_manager_step.json"
                )
                assert (
                    batch_manager_step_path.exists()
                    and batch_manager_step_path.is_file()
                )

                built_batches_dir = batch_manager_step_dir / "built_batches"
                assert built_batches_dir.exists()

                for batch in step.built_batches:
                    batch_path = (
                        built_batches_dir
                        / f"batch_{batch.seq_no}_{batch.data_hash}.json"
                    )
                    assert batch_path.exists() and batch_path.is_file()

                for buffered_step_name in step.data:
                    buffered_step_dir = batch_manager_step_dir / buffered_step_name
                    assert buffered_step_dir.exists() and buffered_step_dir.is_dir()

                    for batch in step.data[buffered_step_name]:
                        batch_path = (
                            buffered_step_dir
                            / f"batch_{batch.seq_no}_{batch.data_hash}.json"
                        )
                        assert batch_path.exists() and batch_path.is_file()

    def test_load_from_cache(self) -> None:
        batch_manager = _BatchManager.from_dict(
            {
                "steps": {
                    "step1": {
                        "step_name": "step1",
                        "accumulate": True,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {"0": {"Z": 1234}},
                        "input_batch_size": None,
                        "data": {
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
                                    "data_hash": "1234",
                                    "size": 7,
                                    "accumulated": False,
                                    "created_from": {},
                                    "batch_routed_to": [],
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "built_batches": [
                            {
                                "seq_no": 0,
                                "step_name": "step1",
                                "last_batch": False,
                                "data": [
                                    [
                                        {"a": 1},
                                        {"a": 2},
                                        {"a": 3},
                                        {"a": 4},
                                        {"a": 5},
                                    ]
                                ],
                                "data_hash": "1234",
                                "size": 5,
                                "accumulated": False,
                                "batch_routed_to": [],
                                "created_from": {},
                                "type_info": {
                                    "module": "distilabel.pipeline.batch",
                                    "name": "_Batch",
                                },
                            }
                        ],
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                    "step2": {
                        "step_name": "step2",
                        "accumulate": False,
                        "convergence_step": False,
                        "convergence_step_batches_consumed": {"0": {"Z": 1234}},
                        "input_batch_size": 50,
                        "data": {
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
                                    "data_hash": "1234",
                                    "size": 7,
                                    "accumulated": False,
                                    "created_from": {},
                                    "batch_routed_to": [],
                                    "type_info": {
                                        "module": "distilabel.pipeline.batch",
                                        "name": "_Batch",
                                    },
                                }
                            ],
                        },
                        "built_batches": [
                            {
                                "seq_no": 0,
                                "step_name": "step1",
                                "last_batch": False,
                                "data": [
                                    [
                                        {"a": 1},
                                        {"a": 2},
                                        {"a": 3},
                                        {"a": 4},
                                        {"a": 5},
                                    ]
                                ],
                                "data_hash": "1234",
                                "size": 5,
                                "accumulated": False,
                                "batch_routed_to": [],
                                "created_from": {},
                                "type_info": {
                                    "module": "distilabel.pipeline.batch",
                                    "name": "_Batch",
                                },
                            }
                        ],
                        "seq_no": 0,
                        "last_batch_received": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch_manager",
                            "name": "_BatchManagerStep",
                        },
                    },
                },
                "last_batch_received": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_sent": {
                    "step1": {
                        "seq_no": 0,
                        "step_name": "step1",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
                    },
                    "step2": {
                        "seq_no": 0,
                        "step_name": "step2",
                        "last_batch": False,
                        "data": [],
                        "size": 0,
                        "accumulated": False,
                        "created_from": {},
                        "batch_routed_to": [],
                        "type_info": {
                            "module": "distilabel.pipeline.batch",
                            "name": "_Batch",
                        },
                    },
                },
                "last_batch_flag_sent_to": ["step3"],
                "type_info": {
                    "module": "distilabel.pipeline.batch_manager",
                    "name": "_BatchManager",
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_manager_path = Path(tmp_dir) / "batch_manager.json"
            batch_manager.cache(batch_manager_path)
            loaded_batch_manager = _BatchManager.load_from_cache(batch_manager_path)

        assert batch_manager.dump() == loaded_batch_manager.dump()
