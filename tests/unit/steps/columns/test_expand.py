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

import json
from typing import Union

import pytest

from distilabel.pipeline.local import Pipeline
from distilabel.steps.columns.expand import ExpandColumns


class TestExpandColumns:
    def test_always_dict(self) -> None:
        expand_columns = ExpandColumns(
            name="expand_columns",
            columns=["column1", "column2"],
            pipeline=Pipeline(name="unit-test"),
        )

        assert expand_columns.columns == {"column1": "column1", "column2": "column2"}

    @pytest.mark.parametrize(
        "encoded, split_statistics, values, stats",
        [
            (
                False,
                False,
                [
                    {
                        "column1": [1, 2, 3],
                        "column2": ["a", "b", "c"],
                        "distilabel_metadata": {
                            "statistics_column1": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                            "statistics_column2": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                        },
                    }
                ],
                [
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                ],
            ),
            (
                ["column1", "column2"],
                False,
                [
                    {
                        "column1": json.dumps([1, 2, 3]),
                        "column2": json.dumps(["a", "b", "c"]),
                        "distilabel_metadata": {
                            "statistics_column1": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                            "statistics_column2": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                        },
                    }
                ],
                [
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                        "statistics_column2": {
                            "input_tokens": [12],
                            "output_tokens": [12],
                        },
                    },
                ],
            ),
            (
                False,
                True,
                [
                    {
                        "column1": [1, 2, 3],
                        "column2": ["a", "b", "c"],
                        "distilabel_metadata": {
                            "statistics_column1": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                            "statistics_column2": {
                                "input_tokens": [12],
                                "output_tokens": [12],
                            },
                        },
                    }
                ],
                [
                    {
                        "statistics_column1": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                        "statistics_column2": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                        "statistics_column2": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                        "statistics_column2": {
                            "input_tokens": [4],
                            "output_tokens": [4],
                        },
                    },
                ],
            ),
            (
                False,
                True,
                [
                    {
                        "column1": [1, 2, 3],
                        "column2": ["a", "b", "c"],
                        "distilabel_metadata": {
                            "statistics_column1": {
                                "input_tokens": [793],
                                "output_tokens": [361],
                            },
                            "statistics_column2": {
                                "input_tokens": [202],
                                "output_tokens": [100],
                            },
                        },
                    }
                ],
                [
                    {
                        "statistics_column1": {
                            "input_tokens": [264],
                            "output_tokens": [120],
                        },
                        "statistics_column2": {
                            "input_tokens": [67],
                            "output_tokens": [33],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [264],
                            "output_tokens": [120],
                        },
                        "statistics_column2": {
                            "input_tokens": [67],
                            "output_tokens": [33],
                        },
                    },
                    {
                        "statistics_column1": {
                            "input_tokens": [264],
                            "output_tokens": [120],
                        },
                        "statistics_column2": {
                            "input_tokens": [67],
                            "output_tokens": [33],
                        },
                    },
                ],
            ),
        ],
    )
    def test_process(
        self,
        encoded: Union[bool, list[str]],
        split_statistics: bool,
        values: list[dict[str, Union[list[int], list[str], str]]],
        stats: dict[str, dict[str, int]],
    ) -> None:
        expand_columns = ExpandColumns(
            columns=["column1", "column2"],
            encoded=encoded,
            split_statistics=split_statistics,
        )

        result = next(expand_columns.process(values))

        assert result == [
            {
                "column1": 1,
                "column2": "a",
                "distilabel_metadata": stats[0],
            },
            {
                "column1": 2,
                "column2": "b",
                "distilabel_metadata": stats[1],
            },
            {
                "column1": 3,
                "column2": "c",
                "distilabel_metadata": stats[2],
            },
        ]

    @pytest.mark.parametrize("left,right", [(2, 1), (1, 2), (0, 2), (2, 0)])
    @pytest.mark.parametrize("encoded", [False, True])
    def test_reject_unequal_lengths(self, left, right, encoded):
        step = ExpandColumns(
            columns={"answers": "answer", "scores": "score"}, encoded=encoded
        )
        row = {"answers": list(range(left)), "scores": list(range(right))}
        if encoded:
            row = {key: json.dumps(value) for key, value in row.items()}
        with pytest.raises(ValueError, match="same length") as error:
            next(step.process([row]))
        assert "answers" in str(error.value)
        assert "scores" in str(error.value)

    @pytest.mark.parametrize("split_statistics", [False, True])
    def test_empty_lists_produce_no_rows(self, split_statistics):
        step = ExpandColumns(
            columns=["answers", "scores"], split_statistics=split_statistics
        )
        row = {
            "answers": [],
            "scores": [],
            "distilabel_metadata": {
                "statistics_task": {"input_tokens": [12], "output_tokens": [6]}
            },
        }
        assert next(step.process([row])) == []

    def test_split_statistics_keeps_config_across_rows_and_calls(self):
        from copy import deepcopy

        step = ExpandColumns(
            columns={"answers": "answer", "scores": "score"}, split_statistics=True
        )
        configured = dict(step.columns)
        inputs = step.inputs
        row = {
            "answers": ["a", "b"],
            "scores": [1, 2],
            "distilabel_metadata": {
                "statistics_task": {"input_tokens": [12], "output_tokens": [6]}
            },
        }
        result = next(step.process([deepcopy(row), deepcopy(row)]))
        assert [(r["answer"], r["score"]) for r in result] == [("a", 1), ("b", 2)] * 2
        assert all(
            r["distilabel_metadata"]["statistics_task"]
            == {"input_tokens": [6], "output_tokens": [3]}
            for r in result
        )
        assert step.columns == configured
        assert step.inputs == inputs
        assert next(step.process([deepcopy(row)])) == result[:2]
