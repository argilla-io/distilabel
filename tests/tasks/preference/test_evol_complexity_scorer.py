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

from typing import List

import pytest
from distilabel.tasks.preference.evol_scorer import EvolComplexityScorerTask

prompt_1 = """Ranking the following questions according to the difficulty and complexity. Score 1-2.
You can give a score of 3 if the question is too complex for you to answer it. You should
respond with the format:
[1] Score: 1
[2] Score: 2
...

[1] instruct 1
[2] instruct 2"""

prompt_2 = """Ranking the following questions according to the difficulty and complexity. Score 1-4.
You can give a score of 5 if the question is too complex for you to answer it. You should
respond with the format:
[1] Score: 1
[2] Score: 2
...

[1] instruct 1
[2] instruct 2
[3] instruct 3
[4] instruct 4"""


@pytest.mark.parametrize(
    "input, expected",
    [
        (["instruct 1", "instruct 2"], prompt_1),
        ([f"instruct {i+1}" for i in range(4)], prompt_2),
    ],
)
def test_evol_complexity_scorer_task(input: List[str], expected: str):
    task = EvolComplexityScorerTask()
    result = task.generate_prompt(input).formatted_prompt
    assert result == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("[1] Score: 4\n[2] Score: 3\n[3] Score: 2\n[4] Score: 1", [4, 3, 2, 1]),
    ],
)
def test_evol_complexity_scorer_task_parsing(input: str, expected: str):
    task = EvolComplexityScorerTask()
    result = task.parse_output(input)
    assert result["ranks"] == expected
