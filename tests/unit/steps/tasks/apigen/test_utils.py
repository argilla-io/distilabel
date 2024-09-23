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

from pathlib import Path
from typing import Any, Dict

import pytest

from distilabel.steps.tasks.apigen.utils import (
    execute_from_response,
    load_module_from_path,
)


@pytest.mark.parametrize(
    "function_name, answer, expected_result",
    [
        (
            "final_velocity",
            {"initial_velocity": 10, "acceleration": 5, "time": 2},
            {"execution_result": "20.0", "keep": True},
        ),
        # In this case, internally we should cast the arguments
        (
            "final_velocity",
            {"initial_velocity": "10", "acceleration": "5", "time": "2"},
            {"execution_result": "20.0", "keep": True},
        ),
        # Different names for the arguments but correctly positioned
        (
            "final_velocity",
            {"v0": "10", "a": "5", "t": "2"},
            {"execution_result": "20.0", "keep": True},
        ),
        # Fail casting one of the values
        (
            "final_velocity",
            {"initial_velocity": "10", "acceleration": "5", "time": "1m/s"},
            {"execution_result": "Argument types not respected", "keep": False},
        ),
        (
            "final_velocity",
            {"initial_velocity": 10, "acceleration": 5},
            {
                "execution_result": "final_velocity() missing 1 required positional argument: 'time'",
                "keep": False,
            },
        ),
        (
            "unknwown_function",
            {"initial_velocity": 10, "acceleration": 5, "time": 2},
            {"execution_result": "Function not found", "keep": False},
        ),
    ],
)
def test_execute_from_response(
    function_name: str, answer: Dict[str, Any], expected_result: Dict[str, Any]
):
    libpath = Path(__file__).parent / "_sample_module.py"
    libpath = load_module_from_path(libpath)
    final_velocity = getattr(libpath, function_name, None)
    result = execute_from_response(final_velocity, answer)
    assert result == expected_result
