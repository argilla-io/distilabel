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

from distilabel.pipeline.local import Pipeline
from distilabel.steps.columns.keep import KeepColumns


class TestKeepColumns:
    def test_init(self) -> None:
        task = KeepColumns(
            name="keep-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.inputs == ["a", "b"]
        assert task.outputs == ["a", "b"]

    def test_process(self) -> None:
        combine = KeepColumns(
            name="keep-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        output = next(combine.process([{"a": 1, "b": 2, "c": 3, "d": 4}]))
        assert output == [{"a": 1, "b": 2}]

    def test_process_preserve_order(self) -> None:
        combine = KeepColumns(
            name="keep-columns",
            columns=["b", "a"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        output = next(combine.process([{"a": 1, "b": 2, "c": 3, "d": 4}]))
        assert output == [{"b": 2, "a": 1}]
