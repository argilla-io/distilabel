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

from distilabel.steps.tasks.apigen.execution_checker import APIGenExecutionChecker

SAMPLE_LIB = Path(__file__).parent / "_sample_module.py"
SAMPLE_LIB_FOLDER = Path(__file__).parent / "_sample_lib"


class TestAPIGenExecutionChecker:
    @pytest.mark.parametrize("lib", (SAMPLE_LIB, SAMPLE_LIB_FOLDER))
    @pytest.mark.parametrize(
        "answers, expected",
        [
            (
                {
                    "query": "Whats the velocity of X?",
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.2,
                                    "acceleration": "0.1",
                                    "time": 5,
                                },
                                "name": "final_velocity",
                            }
                        ]
                    ),
                },
                [
                    {
                        "query": "Whats the velocity of X?",
                        "answers": json.dumps(
                            [
                                {
                                    "arguments": {
                                        "initial_velocity": 0.2,
                                        "acceleration": "0.1",
                                        "time": 5,
                                    },
                                    "name": "final_velocity",
                                }
                            ]
                        ),
                        "keep_row_after_execution_check": True,
                        "execution_result": ["0.7"],
                    }
                ],
            ),
            (
                {
                    "query": "Other query",
                    "answers": json.dumps(
                        [
                            {
                                "arguments": {
                                    "initial_velocity": 0.2,
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
                        "query": "Other query",
                        "answers": json.dumps(
                            [
                                {
                                    "arguments": {
                                        "initial_velocity": 0.2,
                                        "acceleration": 0.1,
                                        "time": 0.5,
                                    },
                                    "name": "unknown_function",
                                }
                            ]
                        ),
                        "keep_row_after_execution_check": False,
                        "execution_result": ["Function 'unknown_function' not found."],
                    }
                ],
            ),
        ],
    )
    def test_process(
        self, lib: str, answers: Dict[str, str], expected: Dict[str, Any]
    ) -> None:
        task = APIGenExecutionChecker(libpath=str(lib))
        task.load()
        result = next(task.process([answers]))
        assert result == expected
