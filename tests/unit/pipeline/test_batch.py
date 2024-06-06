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

from distilabel.pipeline.batch import _Batch


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
            "type_info": {"module": "distilabel.pipeline.batch", "name": "_Batch"},
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
            "type_info": {"module": "distilabel.pipeline.batch", "name": "_Batch"},
        }

    def test_from_dict(self) -> None:
        batch = _Batch.from_dict(
            {
                "seq_no": 0,
                "step_name": "step1",
                "last_batch": False,
                "data": [[{"a": 1}, {"a": 2}, {"a": 3}]],
                "accumulated": False,
                "type_info": {
                    "module": "distilabel.pipeline.batch",
                    "name": "_Batch",
                },
            }
        )

        assert isinstance(batch, _Batch)
        assert batch.seq_no == 0
        assert batch.step_name == "step1"
        assert batch.last_batch is False
        assert batch.data == [[{"a": 1}, {"a": 2}, {"a": 3}]]
        assert batch.accumulated is False
