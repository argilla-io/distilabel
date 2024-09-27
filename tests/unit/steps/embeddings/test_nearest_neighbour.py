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

from distilabel.steps.embeddings.nearest_neighbour import FaissNearestNeighbour


class TestFaissNearestNeighbour:
    def test_process(self) -> None:
        step = FaissNearestNeighbour()

        step.load()

        results = next(
            step.process(
                inputs=[
                    {"embedding": [0.1, -0.4, 0.7, 0.2]},
                    {"embedding": [-0.3, 0.9, 0.1, -0.5]},
                    {"embedding": [0.6, 0.2, -0.1, 0.8]},
                    {"embedding": [-0.2, -0.6, 0.4, 0.3]},
                    {"embedding": [0.9, 0.1, -0.3, -0.2]},
                    {"embedding": [0.4, -0.7, 0.6, 0.1]},
                    {"embedding": [-0.5, 0.3, -0.2, 0.9]},
                    {"embedding": [0.7, 0.5, -0.4, -0.1]},
                    {"embedding": [-0.1, -0.9, 0.8, 0.6]},
                ]
            )
        )

        assert results == [
            {
                "embedding": [0.1, -0.4, 0.7, 0.2],
                "nn_indices": [5],
                "nn_scores": [0.19999998807907104],
            },
            {
                "embedding": [-0.3, 0.9, 0.1, -0.5],
                "nn_indices": [7],
                "nn_scores": [1.5699999332427979],
            },
            {
                "embedding": [0.6, 0.2, -0.1, 0.8],
                "nn_indices": [7],
                "nn_scores": [1.0000001192092896],
            },
            {
                "embedding": [-0.2, -0.6, 0.4, 0.3],
                "nn_indices": [0],
                "nn_scores": [0.23000000417232513],
            },
            {
                "embedding": [0.9, 0.1, -0.3, -0.2],
                "nn_indices": [7],
                "nn_scores": [0.2200000137090683],
            },
            {
                "embedding": [0.4, -0.7, 0.6, 0.1],
                "nn_indices": [0],
                "nn_scores": [0.19999998807907104],
            },
            {
                "embedding": [-0.5, 0.3, -0.2, 0.9],
                "nn_indices": [2],
                "nn_scores": [1.2400000095367432],
            },
            {
                "embedding": [0.7, 0.5, -0.4, -0.1],
                "nn_indices": [4],
                "nn_scores": [0.2200000137090683],
            },
            {
                "embedding": [-0.1, -0.9, 0.8, 0.6],
                "nn_indices": [3],
                "nn_scores": [0.3499999940395355],
            },
        ]
