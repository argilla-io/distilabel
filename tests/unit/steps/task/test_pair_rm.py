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

from unittest.mock import MagicMock, patch

from distilabel.pipeline.local import Pipeline
from distilabel.steps.task.pair_rm import PairRM


@patch("llm_blender.Blender")
class TestPairRM:
    def test_format_input(self, mocker: MagicMock) -> None:
        ranker = PairRM(name="pair_rm_ranker", pipeline=Pipeline())
        result = ranker.format_input(
            input={
                "instruction": "instruction 1",
                "responses": ["response 1", "response 2", "response 3"],
            }
        )

        assert result == {
            "input": "instruction 1",
            "candidates": ["response 1", "response 2", "response 3"],
        }

    def test_process(self, mocker: MagicMock) -> None:
        ranker = PairRM(name="pair_rm_ranker", pipeline=Pipeline())
        ranker._blender = mocker
