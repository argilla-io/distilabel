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
import pytest

from distilabel.pipeline.local import Pipeline
from distilabel.steps.generators.data import LoadDataFromDicts


class TestLoadDataFromDicts:
    data = [{"instruction": "test"}] * 10

    def test_init(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        data: list[dict[str, str]] = self.data
        task = LoadDataFromDicts(
            name="task", pipeline=pipeline, data=data, batch_size=10
        )
        assert task.data == data
        assert task.batch_size == 10

    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        data: list[dict[str, str]] = self.data
        batch_size = 1
        task = LoadDataFromDicts(
            name="task", pipeline=pipeline, data=data, batch_size=batch_size
        )

        result = task.process()
        for i in range(len(self.data) - batch_size):
            assert next(result) == ([self.data[i]], False)
        assert next(result) == ([self.data[-batch_size]], True)
        with pytest.raises(StopIteration):
            next(result)
