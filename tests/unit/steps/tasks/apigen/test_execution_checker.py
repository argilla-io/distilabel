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
from pathlib import Path
from typing import Any, Dict

import pytest

from distilabel.steps.tasks.apigen.execution_checker import (
    APIGenExecutionChecker,
)

SAMPLE_LIB = Path(__file__).parent / "_sample_module.py"


# TODO: THE APIGenGenerator is generating an extra nested list?


class TestAPIGenExecutionChecker:
    @pytest.mark.parametrize(
        "answers, expected",
        [
            (
                {
                    "query": json.dumps(["Whats the velocity of X?"]),
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "final_velocity",
                            }
                        ]
                    ),
                },
                [{"keep_row_after_execution_check": True, "reason": [None]}],
            ),
            (
                {
                    "query": json.dumps(["Whats the velocity of X?", "other query"]),
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "final_velocity",
                            },
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "final_velocity",
                            },
                        ]
                    ),
                },
                [{"keep_row_after_execution_check": True, "reason": [None, None]}],
            ),
            (
                {
                    "query": json.dumps(["Other query"]),
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "unknown_function",
                            }
                        ]
                    ),
                },
                [
                    {
                        "keep_row_after_execution_check": False,
                        "reason": ["Function 'unknown_function' not found."],
                    }
                ],
            ),
            (
                {
                    "query": json.dumps(["Whats the velocity of X?", "other query"]),
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                    "time": 0.5,
                                },
                                "name": "final_velocity",
                            },
                            {
                                "arguments": {
                                    "initial_velocity": 0.01,
                                    "acceleration": 0.1,
                                },
                                "name": "final_velocity",
                            },
                        ]
                    ),
                },
                [
                    {
                        "keep_row_after_execution_check": False,
                        "reason": [
                            None,
                            "final_velocity() missing 1 required positional argument: 'time'",
                        ],
                    }
                ],
            ),
        ],
    )
    def test_process(self, answers: Dict[str, str], expected: Dict[str, Any]) -> None:
        task = APIGenExecutionChecker(libpath=str(SAMPLE_LIB))
        task.load()
        result = next(task.process([answers]))
        assert result == expected
