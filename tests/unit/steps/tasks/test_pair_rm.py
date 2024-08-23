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

import numpy as np

from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.pair_rm import PairRM


@patch("llm_blender.Blender")
class TestPairRM:
    def test_process(self, mocker: MagicMock) -> None:
        ranker = PairRM(
            name="pair_rm_ranker", pipeline=Pipeline(name="unit-test-pipeline")
        )
        ranker._blender = mocker
        ranker._blender.rank = MagicMock(return_value=np.array([[2, 1, 3], [2, 1, 3]]))

        result = ranker.process(
            [
                {"input": "Hello, how are you?", "candidates": ["fine", "good", "bad"]},
                {"input": "Anybody there?", "candidates": ["get out", "yep", "nope"]},
            ]
        )
        ranked = list(result)[0]

        assert ranked == [
            {
                "input": "Hello, how are you?",
                "candidates": ["fine", "good", "bad"],
                "ranks": [2, 1, 3],
                "ranked_candidates": ["good", "fine", "bad"],
                "model_name": "llm-blender/PairRM",
            },
            {
                "input": "Anybody there?",
                "candidates": ["get out", "yep", "nope"],
                "ranks": [2, 1, 3],
                "ranked_candidates": ["yep", "get out", "nope"],
                "model_name": "llm-blender/PairRM",
            },
        ]

    def test_serialization(self, _: MagicMock) -> None:
        ranker = PairRM(
            name="pair_rm_ranker", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert ranker.dump() == {
            "name": ranker.name,
            "input_mappings": {},
            "output_mappings": {},
            "resources": {
                "cpus": None,
                "gpus": None,
                "memory": None,
                "replicas": 1,
                "resources": None,
            },
            "input_batch_size": ranker.input_batch_size,
            "model": ranker.model,
            "instructions": None,
            "runtime_parameters_info": [
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "description": "The number of replicas for the step.",
                            "name": "replicas",
                            "optional": True,
                        },
                        {
                            "description": "The number of CPUs assigned to each step replica.",
                            "name": "cpus",
                            "optional": True,
                        },
                        {
                            "description": "The number of GPUs assigned to each step replica.",
                            "name": "gpus",
                            "optional": True,
                        },
                        {
                            "description": "The memory in bytes required for each step replica.",
                            "name": "memory",
                            "optional": True,
                        },
                        {
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                            "name": "resources",
                            "optional": True,
                        },
                    ],
                },
                {
                    "description": "The number of rows that will contain the batches processed by the step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
            ],
            "type_info": {"module": "distilabel.steps.tasks.pair_rm", "name": "PairRM"},
        }
