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

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.steps.columns.combine import CombineOutputs


class TestCombineOutputs:
    def test_process(self) -> None:
        combine = CombineOutputs()

        output = next(
            combine.process(
                [
                    {
                        "a": 1,
                        "b": 2,
                        DISTILABEL_METADATA_KEY: {"model": "model-1", "a": 1},
                    }
                ],
                [
                    {
                        "c": 3,
                        "d": 4,
                        DISTILABEL_METADATA_KEY: {"model": "model-2", "b": 1},
                    }
                ],
            )
        )

        assert output == [
            {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
                DISTILABEL_METADATA_KEY: {
                    "model": ["model-1", "model-2"],
                    "a": 1,
                    "b": 1,
                },
            }
        ]
