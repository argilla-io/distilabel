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
from distilabel.steps.expand import ExpandColumns


class TestExpandColumns:
    def test_always_dict(self) -> None:
        expand_columns = ExpandColumns(
            name="expand_columns",
            columns=["column1", "column2"],
            pipeline=Pipeline(name="unit-test"),
        )

        assert expand_columns.columns == {"column1": "column1", "column2": "column2"}

    def test_process(self) -> None:
        expand_columns = ExpandColumns(
            name="expand_columns",
            columns=["column1", "column2"],
            pipeline=Pipeline(name="unit-test"),
        )

        result = next(
            expand_columns.process([{"column1": [1, 2, 3], "column2": ["a", "b", "c"]}])
        )

        assert result == [
            {"column1": 1, "column2": "a"},
            {"column1": 2, "column2": "b"},
            {"column1": 3, "column2": "c"},
        ]
