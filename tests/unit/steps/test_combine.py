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

from typing import Any, Dict, List, Optional

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.steps.columns.group import GroupColumns, MergeColumns


class TestGroupColumns:
    def test_init(self) -> None:
        task = GroupColumns(
            name="combine-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.inputs == ["a", "b"]
        assert task.outputs == ["merged_a", "merged_b"]

        task = GroupColumns(
            name="combine-columns",
            columns=["a", "b"],
            output_columns=["c", "d"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.inputs == ["a", "b"]
        assert task.outputs == ["c", "d"]

    def test_process(self) -> None:
        combine = GroupColumns(
            name="combine-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        output = next(combine.process([{"a": 1, "b": 2}], [{"a": 3, "b": 4}]))
        assert output == [{"merged_a": [1, 3], "merged_b": [2, 4]}]


class TestMergeColumns:
    @pytest.mark.parametrize(
        "output_column, expected",
        [
            (None, "combined_column"),
            ("queries", "queries"),
        ],
    )
    def test_init(self, output_column: Optional[str], expected: str) -> None:
        task = MergeColumns(columns=["query", "queries"], output_column=output_column)

        assert task.inputs == ["query", "queries"]
        assert task.outputs == [expected]

    @pytest.mark.parametrize(
        "columns",
        [
            [{"query": 1, "queries": 2}],
            [{"query": 1, "queries": [2]}],
            [{"query": [1], "queries": [2]}],
        ],
    )
    def test_process(self, columns: List[Dict[str, Any]]) -> None:
        combiner = MergeColumns(
            columns=["query", "queries"],
        )
        output = next(combiner.process(columns))
        assert output == [{"combined_column": [1, 2]}]
