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
from distilabel.steps.deita import DeitaFiltering


class TestDeitaFiltering:
    def test_process(self) -> None:
        deita_filtering = DeitaFiltering(
            name="deita_filtering",
            data_budget=1,
            pipeline=Pipeline(name="unit-test"),
        )

        deita_filtering.load()

        result = next(
            deita_filtering.process(
                [
                    {
                        "evol_instruction_score": 0.5,
                        "evol_response_score": 0.5,
                        "embedding": [-8.12729941, -5.24642847, -6.34003029],
                    },
                    {
                        "evol_instruction_score": 0.6,
                        "evol_response_score": 0.6,
                        "embedding": [2.99329242, 0.7800932, 0.7799726],
                    },
                    {
                        "evol_instruction_score": 0.7,
                        "evol_response_score": 0.7,
                        "embedding": [10.29041806, 14.33088073, 13.00557506],
                    },
                ]
            )
        )

        assert result == [
            {
                "evol_instruction_score": 0.5,
                "evol_response_score": 0.5,
                "embedding": [-8.12729941, -5.24642847, -6.34003029],
                "deita_score": 0.25,
                "nearest_neighbor_distance": 1.9042812683723933,
            }
        ]
