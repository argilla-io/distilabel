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

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.pipeline.local import Pipeline
from distilabel.steps.columns.group import CombineColumns, GroupColumns


class TestGroupColumns:
    def test_init(self) -> None:
        task = GroupColumns(
            name="group-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.inputs == ["a", "b"]
        assert task.outputs == ["grouped_a", "grouped_b"]

        task = GroupColumns(
            name="group-columns",
            columns=["a", "b"],
            output_columns=["c", "d"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        assert task.inputs == ["a", "b"]
        assert task.outputs == ["c", "d"]

    def test_process(self) -> None:
        group = GroupColumns(
            name="group-columns",
            columns=["a", "b"],
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        output = next(
            group.process(
                [{"a": 1, "b": 2, DISTILABEL_METADATA_KEY: {"model": "model-1"}}],
                [{"a": 3, "b": 4, DISTILABEL_METADATA_KEY: {"model": "model-2"}}],
            )
        )
        assert output == [
            {
                "grouped_a": [1, 3],
                "grouped_b": [2, 4],
                DISTILABEL_METADATA_KEY: {"model": ["model-1", "model-2"]},
            }
        ]


def test_CombineColumns_deprecation_warning():
    with pytest.deprecated_call():
        CombineColumns(
            name="combine_columns",
            columns=["generation", "model_name"],
        )
    from packaging.version import Version

    import distilabel

    assert Version(distilabel.__version__) <= Version("1.5.0")
