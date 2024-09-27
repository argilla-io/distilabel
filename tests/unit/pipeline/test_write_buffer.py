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

from distilabel.distiset import Distiset, create_distiset
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.write_buffer import _WriteBuffer
from tests.unit.pipeline.utils import (
    DummyGeneratorStep,
    DummyStep1,
    DummyStep2,
    batch_gen,
)


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
            batch = batch_gen(dummy_step_2.name)  # type: ignore
            write_buffer.add_batch(batch)

            # Add 45 more rows, should write now
            for _ in range(9):
                batch = batch_gen(dummy_step_2.name)  # type: ignore
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00001.parquet").exists()

            # Add 50 more rows, we should have a new file
            for _ in range(10):
                batch = batch_gen(dummy_step_2.name)  # type: ignore
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00002.parquet").exists()

            # Add more rows and close the write buffer, we should have a new file
            for _ in range(5):
                batch = batch_gen(dummy_step_2.name)  # type: ignore
                write_buffer.add_batch(batch)

            write_buffer.close()

            assert Path(folder, "dummy_step_2", "00003.parquet").exists()

            ds = create_distiset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 1
            assert len(ds["default"]["train"]) == 125

    def test_write_buffer_multiple_leaf_steps_and_create_dataset(self) -> None:
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
                batch = batch_gen(dummy_step_2.name)  # type: ignore
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_2", "00001.parquet").exists()

            for _ in range(10):
                batch = batch_gen(dummy_step_3.name)  # type: ignore
                write_buffer.add_batch(batch)

            assert Path(folder, "dummy_step_3", "00001.parquet").exists()

            for _ in range(5):
                batch = batch_gen(dummy_step_2.name)  # type: ignore
                write_buffer.add_batch(batch)

            for _ in range(5):
                batch = batch_gen(dummy_step_3.name)  # type: ignore
                write_buffer.add_batch(batch)

            write_buffer.close()

            assert Path(folder, "dummy_step_2", "00002.parquet").exists()
            assert Path(folder, "dummy_step_3", "00002.parquet").exists()

            ds = create_distiset(write_buffer._path)
            assert isinstance(ds, Distiset)
            assert len(ds.keys()) == 2
            assert len(ds["dummy_step_2"]["train"]) == 75
            assert len(ds["dummy_step_3"]["train"]) == 75
